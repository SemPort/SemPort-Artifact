# import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from module.locator.anchor import pa_function_info, pb_function_info
from module.locator.units import extract_units, render_unit_diff
from utils.llm import LLM_ATTEMPTS, build_llm, invoke_llm
# from utils.llm import DEFAULT_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)

INTENT_PROMPT = """You are analyzing one hunk of a security patch (PA -> PB) for a C codebase.
The goal is a version-independent record of what the patch does, used later
to port the patch to a different version of the same code. You are given the
enclosing function BEFORE the patch (PA) and AFTER the patch (PB): use them
to understand exactly what the change does and where lines land.

## Hunk to analyze: {file} / {func}

```diff
{unit_diff}
```

## Function before the patch (PA)

```c
{pa_src}
```

## Function after the patch (PB)

```c
{pb_src}
```

## Sibling hunks of the same patch in {file}

{siblings}

## Why the patch was written (source commit message)

{pb_message}

## Task

Describe the transplantable intent of this hunk, separated into WHAT MUST
HOLD AFTER THE TRANSPLANT versus WHAT THE PATCH STRUCTURALLY DID IN PA.
Verify your reading against the PB version: every added line lands where PB
shows it, and a deleted guard/brace usually means a block was MERGED into a
neighbouring guard (state which guard now wraps the affected lines).

A hunk may pair with a sibling hunk of the same patch (for example a guard
removed here because the same patch adds the equivalent check elsewhere).
For such pairs the removal is a STRUCTURAL cleanup whose necessity on the
ported version depends on the target's own call structure — classify it as
conditional, never as required. The commit message is authoritative about
the author's intent: distill its mechanism explanation and any maintainer
notes into the behavioral/structural fields. A note like "[dropped X to
avoid Y]" means the author gave up that simplification — the corresponding
removal is conditional and a port may skip it. The message describes the
author's choices for PA; transplant decisions still follow the PC
structure, so never refuse a necessary PC adaptation by citing the message.

Output ONLY one JSON object:

{{"security_goal": "...",
  "behavioral_change": "...",
  "structural_change": "...",
  "hunk_role": "security_essential",
  "role_evidence": "...",
  "required": ["..."],
  "conditional": [{{"action": "...", "condition": "..."}}],
  "invariants": ["..."]}}

Field rules:
- "security_goal": 1 sentence — the property the patch establishes
  (e.g. "Reject descriptors whose ops table lacks a verify callback.")
- "behavioral_change": the REQUIRED behavioral outcome, stated on the
  target side and independent of which function provides it
  (e.g. "__foo_submit() must reject unsupported descriptors.")
- "structural_change": 1 sentence describing the source-side structural
  edit (e.g. "Move the capability check from foo_submit() to
  __foo_submit().")
- "hunk_role": whether THIS hunk's change is needed to fix the issue the
  commit addresses, or is incidental work the fix merely carries along:
  - "security_essential": reverting this hunk alone (keeping every other
    hunk of the patch) would leave the vulnerability or bug the commit
    message describes still reproducible — the hunk changes WHAT is
    enforced or computed (a guard, a corrected operation, a released
    resource the code failed to release, an allocation flag, a lifetime,
    a privilege check)
  - "incidental": the hunk reorganizes how existing behavior is expressed
    — extracting a helper, splitting/merging functions, renaming,
    moving code, reformatting, dead-code or comment cleanup — without
    changing what the code enforces.  Test each hunk in isolation: if
    reverting ONLY this hunk still leaves the vulnerability fixed by the
    remaining hunks, it is incidental, EVEN IF it touches resource
    handling, accounting, or error paths whose behavior the refactor
    merely preserves.  A hunk that BOTH reorganizes AND changes the
    enforced behavior (e.g. the refactor and the actual fix live in one
    hunk) is security_essential
  - When unsure, answer "security_essential" — an unnecessary transplant
    is recoverable, a dropped fix is not
  - The commit message is the primary evidence for the split: a subject
    line like "Refactor X" with the behavioral change introduced by an
    "Also ..." note means the reorganization is incidental and only the
    noted behavioral change is security_essential.  Items mentioned only
    as v2 review leftovers ("drop unused variable") are incidental
- "role_evidence": 1 sentence justifying hunk_role; MANDATORY when the
  role is incidental (name the refactor/cleanup mechanism, e.g. "splits
  the release loop into a helper the fix then calls — the fix itself is
  the GFP flag change in the other hunk")
- "required": atomic behaviors that MUST be established on PC after the
  transplant — the additions/guards the patch introduces, stated against
  the target site (e.g. "Add the capability check to __foo_submit().").
  A removal goes here ONLY when the removed code is wrong in every version
  of this code (a genuine use-after-free, double-free, race) — not merely
  redundant
- "conditional": source-side removals/relocations whose necessity depends
  on the PC structure — each as an object with the action and an explicit,
  PC-checkable condition (e.g. {{"action": "Remove the corresponding check
  from foo_submit().", "condition": "only if the target-side call
  structure makes it redundant without changing unrelated control flow"}}).
  When unsure whether a removal is required or conditional, classify it
  CONDITIONAL — preserving PC code is never wrong, an unnecessary removal is
  - "invariants": constraints that must hold after a correct transplant to
    ANY version of this code — state GUARD conditions explicitly (e.g.
    "the error path releases every descriptor allocated on the success
    path", never just "the descriptors are released"); check each one
    against the PB version. Invariants state SECURITY properties only —
    never structural facts about which call runs where or in what order
    ("X must still be called", "the loop order is unchanged" are
    structural, belong in structural_change, and MUST NOT be invariants:
    another version may legitimately satisfy the security goal with a
    different call structure). Do NOT restate a paired relocation as an
  invariant of THIS hunk: "the check no longer executes here" is only
  valid when the same patch does not add the equivalent check in a
  sibling hunk. When the patch adds or moves a PAIRED resource operation
  at a caller (lock/unlock, get/put), state the bracket invariant
  explicitly ("the resource is held across the operation and EVERY exit
  path releases it") — a port that inlines the callee into the caller
  must keep the pair on all paths
- Name concrete variables/calls; do NOT restate the diff line-by-line
- Comment and documentation lines the hunk adds or removes follow the same
  required/conditional classification as their code region
"""

PROMPT_SHA = hashlib.sha1(INTENT_PROMPT.encode()).hexdigest()[:12]

PARSE_RETRY_SUFFIX = '\n\nYour previous reply was not valid JSON. Output ONLY the JSON object.'

def parse_intent(response: str) -> dict | None:
    m = _JSON_RE.search(response)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    def _strs(value) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [s.strip() for s in value if isinstance(s, str) and s.strip()]
        return []

    behavioral = data.get('behavioral_change')
    structural = data.get('structural_change')
    legacy_summary = data.get('summary')
    behavioral = behavioral.strip() if isinstance(behavioral, str) else ''
    structural = structural.strip() if isinstance(structural, str) else ''
    legacy_summary = legacy_summary.strip() if isinstance(legacy_summary, str) else ''

    required = _strs(data.get('required'))
    conditional = []
    for c in data.get('conditional') or []:
        if isinstance(c, dict):
            action = str(c.get('action', '')).strip()
            condition = str(c.get('condition', '')).strip()
            if action:
                conditional.append({'action': action, 'condition': condition})
        elif isinstance(c, str) and c.strip():
            conditional.append({'action': c.strip(), 'condition': ''})

    if not behavioral and not structural and legacy_summary:
        required = _strs(data.get('actions'))
    if not (behavioral or structural or legacy_summary or required):
        return None

    role_raw = str(data.get('hunk_role', '')).strip().lower()
    role_evidence = str(data.get('role_evidence', '')).strip()
    incidental = role_raw in ('incidental', 'refactor', 'refactoring',
                              'cleanup', 'style', 'comment') \
        and bool(role_evidence)
    summary = behavioral or structural or legacy_summary
    actions = required + [c['action'] for c in conditional]
    return {'summary': summary,
            'security_goal': (data.get('security_goal') or '').strip()
            if isinstance(data.get('security_goal'), str) else '',
            'behavioral_change': behavioral,
            'structural_change': structural,
            'hunk_role': 'incidental' if incidental else 'security_essential',
            'role_evidence': role_evidence if incidental else '',
            'required': required,
            'conditional': conditional,
            'actions': actions,
            'invariants': _strs(data.get('invariants'))}

def _fn_source(info: dict) -> str:
    body = '\n'.join(f"{info['start_line'] + i:>6} {l}"
                     for i, l in enumerate(info['source'].splitlines()))
    return (f"(function {info['name']}, lines {info['start_line']}-"
            f"{info['end_line']})\n{body}")

def _clip(s: str, width: int = 72) -> str:
    return s if len(s) <= width else s[:width - 3] + '...'

def _siblings_brief(units: list[dict], unit: dict) -> str:
    lines = []
    for u in units:
        if u['unit_id'] == unit['unit_id'] or u['file'] != unit['file']:
            continue
        adds = [e[2] for h in u['hunks'] for e in h.get('add_lines', []) if e[2]]
        dels = [e[2] for h in u['hunks'] for e in h.get('delete_lines', []) if e[2]]
        fmt = lambda xs: ' | '.join(_clip(x) for x in xs[:3]) if xs else '(none)'
        lines.append(f"- scope {u['func']}: first added lines: {fmt(adds)}; "
                     f"first removed lines: {fmt(dels)}")
    return '\n'.join(lines) if lines \
        else '(no other hunk of this patch touches this file)'

def extract_intent(llm, task_dir: str, out_dir: str, meta: dict, unit: dict,
                   siblings: str = '', pb_message: str = '') -> dict:
    safe_id = re.sub(r'[^\w.-]', '__', unit['unit_id'])
    telemetry = {'prompt_tokens': 0, 'completion_tokens': 0, 'llm_ms': 0.0, 'attempts': 0}
    result = {'unit_id': unit['unit_id'], 'file': unit['file'], 'func': unit['func'],
              'intent': None, 'parse_ok': False, 'error': ''}
    t0 = time.monotonic()
    try:
        pa_info = pa_function_info(task_dir, meta, unit)
        pb_info = pb_function_info(task_dir, meta, unit)
        prompt = INTENT_PROMPT.format(
            file=unit['file'], func=unit['func'], unit_diff=render_unit_diff(unit),
            pa_src=_fn_source(pa_info) if pa_info
            else '(PA function source not available)',
            pb_src=_fn_source(pb_info) if pb_info
            else '(PB function source not available)',
            siblings=siblings,
            pb_message=pb_message or '(commit message not available)')
        with open(os.path.join(out_dir, f'{safe_id}.prompt.txt'), 'w') as f:
            f.write(prompt)
        intent, response = None, ''
        for attempt in range(LLM_ATTEMPTS):
            p = prompt if attempt == 0 else prompt + PARSE_RETRY_SUFFIX
            response = invoke_llm(llm, p, telemetry)
            with open(os.path.join(out_dir, f'{safe_id}.response.txt'), 'w') as f:
                f.write(response)
            intent = parse_intent(response)
            if intent is not None:
                break
        result['parse_ok'] = intent is not None
        if intent is None:
            result['error'] = 'no valid intent JSON in response'
            logger.error(f'  unit {unit["unit_id"]}: {result["error"]}')
        else:
            result['intent'] = intent
            with open(os.path.join(out_dir, f'{safe_id}.json'), 'w') as f:
                json.dump(intent, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        result['error'] = str(exc)[:300]
        logger.error(f'  unit {unit["unit_id"]} failed: {exc}')
    result.update(telemetry)
    result['wall_ms'] = (time.monotonic() - t0) * 1000
    if result['intent']:
        i = result['intent']
        logger.info(f'  unit {unit["unit_id"]}: parse_ok={result["parse_ok"]} '
                    f'required={len(i["required"])} '
                    f'conditional={len(i["conditional"])} '
                    f'invariants={len(i["invariants"])}')
    else:
        logger.info(f'  unit {unit["unit_id"]}: parse_ok=False')
    return result

def cache_valid(marker: str) -> bool:
    if not os.path.exists(marker):
        return False
    try:
        with open(marker, encoding='utf-8') as f:
            cached = json.load(f)
        return cached.get('intent_prompt_sha') == PROMPT_SHA
    except (OSError, ValueError):
        return False

def run_cve(cve: str, mode: str, model_name: str, llm_threads: int,
            config_file: str, force: bool) -> dict:
    task_dir = os.path.normpath(os.path.join(SRC_ROOT, '..', 'task', mode, cve))
    if not os.path.exists(os.path.join(task_dir, 'metadata.json')):
        raise RuntimeError(f'task artifacts missing for {cve}: run dataset.prepare first')
    with open(os.path.join(task_dir, 'metadata.json')) as f:
        meta = json.load(f)

    out_dir = os.path.join(task_dir, 'intent')
    marker = os.path.join(out_dir, 'intents.json')
    if cache_valid(marker) and not force:
        logger.info(f'[{cve}] intent cache hit (prompt sha {PROMPT_SHA}), skip')
        return json.load(open(marker))
    if os.path.exists(marker):
        logger.info(f'[{cve}] intent prompt changed since the cached run, re-extracting')
    os.makedirs(out_dir, exist_ok=True)

    llm = build_llm(model_name, config_file)
    units = extract_units(task_dir)
    logger.info(f'[{cve}] extracting intent for {len(units)} unit(s) with {model_name} '
                f'(llm_threads={llm_threads})')
    if llm_threads > 1 and len(units) > 1:
        with ThreadPoolExecutor(max_workers=llm_threads) as pool:
            results = list(pool.map(
                lambda u: extract_intent(llm, task_dir, out_dir, meta, u,
                                         _siblings_brief(units, u),
                                         meta.get('pb_message', '')),
                units))
    else:
        results = [extract_intent(llm, task_dir, out_dir, meta, u,
                                  _siblings_brief(units, u),
                                  meta.get('pb_message', ''))
                   for u in units]

    s = {
        'cve': cve, 'mode': mode, 'model': model_name, 'stage': 'intent',
        'intent_prompt_sha': PROMPT_SHA,
        'units': results,
        'summary': {
            'units_total': len(results),
            'parse_ok': sum(r['parse_ok'] for r in results),
            'prompt_tokens': sum(r['prompt_tokens'] for r in results),
            'completion_tokens': sum(r['completion_tokens'] for r in results),
            'llm_ms': sum(r['llm_ms'] for r in results),
        },
    }
    if s['summary']['parse_ok'] > 0 or not results:
        with open(marker, 'w') as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
    else:
        logger.error(f'[{cve}] intent extraction failed for every unit; '
                     f'marker NOT written')
    logger.info(f'[{cve}] intent done: {s["summary"]["parse_ok"]}/{len(units)} parsed, '
                f'tokens p={s["summary"]["prompt_tokens"]} '
                f'c={s["summary"]["completion_tokens"]}')
    return s

# def main(argv: list[str] | None = None) -> int:
#     p = argparse.ArgumentParser(prog='module.locator.intent',
#                                 description='Extract per-hunk patch intent (PA->PB).')
#     p.add_argument('--cve', required=True, help='comma-separated CVE ids')
#     p.add_argument('--mode', default='backport', choices=['backport', 'upstream-to-downstream'])
#     p.add_argument('--model', default='GPT-4o-test', help='config.ini section (default: GPT-4o-test)')
#     p.add_argument('--config', default=DEFAULT_CONFIG, help='generator config.ini path')
#     p.add_argument('--llm-threads', type=int, default=4, help='parallel unit LLM calls per CVE')
#     p.add_argument('--force', action='store_true', help='rerun even if intent/intents.json exists')
#     args = p.parse_args(argv)

#     for cve in [c.strip() for c in args.cve.split(',') if c.strip()]:
#         run_cve(cve, args.mode, args.model, args.llm_threads, args.config, args.force)
#     return 0

