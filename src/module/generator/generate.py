import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

from dataset.prepare import _get_repos
from module.generator.prompt import build_prompt
from module.locator import anchor as anchor_mod, intent as intent_mod
from module.locator.anchor import _git_show
from module.locator.symbols import for_group
from module.locator.units import (_pa_context_blocks, extract_units,
                                  fuzzy_pa_line_map, snapshot_path,
                                  span_containment, substantive_lines)
from module.types import (Anchor, Intent, PatchGroup, PatchedFunction,
                          Telemetry, Unit)
from module.validator.validate import (account_add_lines, check_compile,
                                       file_arity_mismatch_issues,
                                       placement_issues, validate_function,
                                       _hunk_bodies, _unit_dispositions)
from utils.checks import check_apply
from utils.llm import build_llm, invoke_llm, invoke_llm_messages
from utils.logger import get_logger

logger = get_logger(__name__)

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_C_FENCE_RE = re.compile(r'```(?:c)?\s*\n(.*?)```', re.DOTALL)
_SR_RE = re.compile(r'<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE',
                    re.DOTALL)
_LINENO_RE = re.compile(r'^ *\d+ ')
EMPTY_RETRY_SUFFIX = ("\n\nYour previous reply was identical to the current PC "
                      "function — that is NOT an acceptable transplant. Apply "
                      "every intent action to the PC function as real code "
                      "changes; an empty change-set can never be the answer "
                      "because the source patch demonstrably modifies code.")

SEARCH_MISMATCH_SUFFIX = ("\n\nYour previous SEARCH block does not match the PC "
                          "file. Copy the SEARCH lines EXACTLY as they appear "
                          "in the PC file shown in the anchor section (without "
                          "the line-number prefixes) — never source them from "
                          "the PA section, which contains drifted code. Keep "
                          "the block as small as possible.")

def _feedback_suffix(group: dict, issues: list[str]) -> str:
    bullets = '\n'.join(f'- {issue}' for issue in issues)
    if (group.get('anchor') or {}).get('status') == 'FILE_SCOPE':
        tail = ('Return the corrected complete search/replace block '
                '(`<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`), with the '
                'SEARCH lines copied exactly from the PC file.')
    else:
        tail = 'Return the corrected complete function.'
    constraint = ('Modify only what these flags ask for (a flagged line, or '
                  'the #include a flag names); every other line of your '
                  'previous output must remain byte-identical. A flag with '
                  'a line number points at the exact line to fix — do not '
                  're-send your previous output unchanged.')
    cot = ('Before writing the answer, reason step by step, briefly:\n'
           '1. For each flag above: which of your output lines does it '
           'point at, and what exactly must change there?\n'
           '2. Walk the intent actions and invariants one by one: is each '
           'already satisfied by the PC code, or must it land in this '
           'output?\n'
           '3. Self-check: no line the source patch deletes survives; '
           'every added line sits at the hunk context shown in the anchor; '
           'no line that neither the anchor nor the patch accounts for; '
           'every exit path releases a resource bracket the patch '
           'introduced.')
    directive = ''
    if any(i.startswith(('the output is identical to the PC original',
                         'PA and PC are identical',
                         'source patch deleted line still present'))
           for i in issues):
        directive = _directive_block(group)
    return ('\n\nYour previous output has these problems:\n' + bullets
            + '\n\n' + constraint + '\n\n' + cot
            + '\n\n' + tail + directive)

_DIRECTIVE_PREFIXES = ('the output is identical to the PC original',
                       'PA and PC are identical',
                       'source patch deleted line still present')

def _directive_block(group: dict) -> str:
    parts = []
    for u in group['units']:
        for body in _hunk_bodies(u.get('raw_diff')):
            if not any(k in '-+' for k, _ in body):
                continue
            adds = [t for k, t in body if k == '+']
            dels = [t for k, t in body if k == '-']
            first_change = next(
                (i for i, (k, _) in enumerate(body) if k in '-+'), 0)
            ctx_before = [t for k, t in body[:first_change] if k == ' ']
            anchor = ctx_before[-1] if ctx_before else '(the start of the region)'
            lines = []
            if dels:
                lines.append('  remove these lines:')
                lines += [f'    - {d}' for d in dels[:12]]
            if adds:
                lines.append(f'  insert these lines immediately after '
                             f'`{anchor}`:')
                lines += [f'    + {a}' for a in adds[:12]]
            parts.append('\n'.join(lines))
    if not parts:
        return ''
    return ('\n\nConcretely, the source patch does exactly this — apply it '
            'to the PC function as-is:\n' + '\n'.join(parts))

def _candidate_key(issues: list[str], addressed: int = 0
                   ) -> tuple[int, int, int]:
    structural = sum(1 for i in issues
                     if i.startswith(_DIRECTIVE_PREFIXES + (
                         'no output:', 'assigns to `', 'symbol `', 'type `',
                         'line of the SEARCH block',
                         'comment line of the SEARCH', 'syntax error',
                         'arity mismatch', 'unit not addressed',
                         'the patch adds this line',
                         'pre-existing line deleted', '`')))
    return -addressed, structural, len(issues)

def _empty_retry_suffix(group: dict) -> str:
    raws = [u['raw_diff'] for u in group['units'] if u.get('raw_diff')]
    if not raws:
        return EMPTY_RETRY_SUFFIX
    return (EMPTY_RETRY_SUFFIX
            + '\n\n### Source patch hunks the output must account for (apply '
              'them to the PC function as real code changes)\n'
            + '\n\n'.join(raws))

RESIDUAL_DEFECT_SUFFIX = (
    "\n\nYour previous reply matched the current PC function, and the source "
    "patch's substantive added lines ALREADY EXIST in the PC file.  That "
    "echo means the literal master fix reached this branch already — and "
    "since a correct backport for THIS branch demonstrably exists and "
    "changes the function, the branch-specific fix must take a DIFFERENT "
    "shape than the source patch's lines.  Hunt the residual defect the "
    "way the branch maintainer would:\n"
    "1. Walk the patch's required actions and invariants one by one "
    "against the anchor function. An invariant held only partially (the "
    "existing guard covers fewer lines or fewer call sites than needed) "
    "names the gap.\n"
    "2. Look for SIBLING sites in the anchor function with the same "
    "weakness the patch fixes at its own site — an unguarded call that "
    "dereferences the same NULL pointer, a parallel path missing the same "
    "check, a cleanup step the guard forgot to cover.\n"
    "3. Return the complete function with that residual defect fixed "
    "MINIMALLY (typically: extend the existing guard to the sibling "
    "site). Do not reorganize anything else.")

def _adds_already_present(units: list[dict], pc_text: str) -> bool:
    pc_lines = {l.strip() for l in pc_text.splitlines() if l.strip()}
    adds = [t[2].strip() for u in units for h in (u.get('hunks') or [])
            for t in h.get('add_lines', []) if len(t[2].strip()) >= 8]
    return bool(adds) and all(a in pc_lines for a in adds)

def extract_function(response: str) -> str | None:
    blocks = _C_FENCE_RE.findall(response)
    text = blocks[-1] if blocks else response
    text = text.strip('\n')
    if '{' not in text or '}' not in text:
        return None
    text = _strip_number_prefixes(text)
    if not text.endswith('\n'):
        text += '\n'
    return text

def _strip_number_prefixes(text: str) -> str:
    lines = text.splitlines()
    nonblank = [l for l in lines if l.strip()]
    if len(nonblank) < 4:
        return text
    numbered = sum(1 for l in nonblank if _LINENO_RE.match(l))
    if numbered >= max(3, (len(nonblank) + 1) // 2):
        return '\n'.join(_LINENO_RE.sub('', l, count=1) for l in lines) + '\n'
    return text

def extract_search_replace(response: str) -> str | None:
    m = _SR_RE.search(response)
    return m.group(0) if m else None

_DEF_LINE_RE = re.compile(r'^[A-Za-z_][\w\s\*]*?([A-Za-z_]\w*)\s*\(')
_KEYWORDS_SKIP = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'else',
                  'do', 'ifdef', 'ifndef', 'define', 'include'}
_MECH_ISSUE_PREFIXES = ('syntax error', 'arity mismatch', 'no output:',
                        'line of the SEARCH block',
                        'comment line of the SEARCH',
                        'symbol `', 'type `', 'assigns to `',
                        'inserted into wrong function',
                        'file-scope statement',
                        'definition of `', 'line-number echo')

def _is_mechanical(issue: str) -> bool:
    return issue.startswith(_MECH_ISSUE_PREFIXES)

def _fence_fallback_sr(response: str, pc_text: str) -> str | None:
    blocks = re.findall(r'```[a-zA-Z]*\n(.*?)```', response, re.DOTALL)
    pc_lines = pc_text.splitlines()
    for block in blocks:
        sig = next((l for l in block.splitlines()
                    if l.strip() and not l.lstrip().startswith(('#', '/*', '*', '//'))),
                   None)
        if not sig:
            continue
        m = _DEF_LINE_RE.match(sig.strip())
        if not m or len(m.group(1)) < 3 or m.group(1) in _KEYWORDS_SKIP:
            continue
        name = m.group(1)
        spans = []
        for i, ln in enumerate(pc_lines):
            dm = _DEF_LINE_RE.match(ln)
            if dm and dm.group(1) == name:
                depth, started, end = 0, False, None
                for j in range(i, min(i + 400, len(pc_lines))):
                    depth += pc_lines[j].count('{') - pc_lines[j].count('}')
                    if '{' in pc_lines[j]:
                        started = True
                    if started and depth <= 0:
                        end = j
                        break
                if end is not None and end - i >= 2:
                    spans.append((i, end))
        if len(spans) == 1:
            i, end = spans[0]
            search = '\n'.join(pc_lines[i:end + 1])
            return (f'<<<<<<< SEARCH\n{search}\n=======\n'
                    f'{block.rstrip()}\n>>>>>>> REPLACE')
    return None

def _find_block(lines: list[str], pat: list[str]) -> int | None:
    if not pat or len(pat) > len(lines):
        return None
    keys = [l.strip() for l in lines]
    target = [l.strip() for l in pat]
    for i in range(len(lines) - len(pat) + 1):
        if keys[i:i + len(pat)] == target:
            return i
    return None

def _file_scope_splice(src: str, record: dict,
                       patch_add_lines: set[str] | None = None) -> dict | None:
    m = _SR_RE.search(record.get('_func_text') or '')
    lines = src.splitlines(keepends=True)
    idx = None
    s_lines = m.group(1).splitlines() if m else None
    if s_lines:
        idx = _find_block(lines, s_lines)
        if idx is None:
            idx = _find_block(lines, [_LINENO_RE.sub('', l, count=1)
                                      for l in s_lines])
    if idx is None:
        record['splice'] = 'file_scope_no_match'
        return None
    replace = m.group(2)
    if patch_add_lines:
        rep_set = {l.strip() for l in replace.splitlines() if l.strip()}
        net_removed = {l.strip() for l in s_lines
                       if l.strip() and l.strip() not in rep_set}
        if net_removed & patch_add_lines:
            record['splice'] = 'file_scope_direction_violation'
            return None
    if replace and not replace.endswith('\n'):
        replace += '\n'
    return {'group_id': record['group_id'], 'start': idx + 1,
            'end': idx + len(s_lines), 'text': replace}

def splice(lines: list[str], splices: list[dict],
           want_map: bool = False):
    out, skipped = [], []
    regions: list[dict] = []
    cursor = 0
    for s in sorted(splices, key=lambda x: (x['start'], x['end'])):
        start0, end0 = s['start'] - 1, s['end']
        if start0 < cursor:
            skipped.append(s['group_id'])
            continue
        out.extend(lines[cursor:start0])
        if want_map:
            regions.append({'new_start': len(out) + 1,
                            'new_end': len(out)
                            + len(s['text'].splitlines(keepends=True)),
                            'group_id': s['group_id']})
        out.extend(s['text'].splitlines(keepends=True))
        cursor = end0
    out.extend(lines[cursor:])
    return (out, skipped, regions) if want_map else (out, skipped)

def _git_diff(orig: str, mod: str, rel: str) -> str:
    with tempfile.TemporaryDirectory(prefix='gen_diff_') as tmp:
        po, pm = os.path.join(tmp, 'orig.c'), os.path.join(tmp, 'mod.c')
        with open(po, 'w', encoding='utf-8') as f:
            f.write(orig)
        with open(pm, 'w', encoding='utf-8') as f:
            f.write(mod)
        r = subprocess.run(['git', 'diff', '--no-index', '--no-color', po, pm],
                           capture_output=True, text=True)
        out = []
        for line in r.stdout.splitlines():
            if line.startswith(('diff --git', 'index ')):
                continue
            if line.startswith('--- '):
                out.append(f'--- a/{rel}')
            elif line.startswith('+++ '):
                out.append(f'+++ b/{rel}')
            else:
                out.append(line)
        return '\n'.join(out) + '\n' if out else ''

def _unit_dicts(group: PatchGroup) -> list[dict]:
    return [u.to_dict() for u in group.units]

def build_groups(units: list[Unit], anchors: dict[str, Anchor]
                 ) -> tuple[list[PatchGroup], list[dict]]:
    by_key: dict[tuple, PatchGroup] = {}
    groups: list[PatchGroup] = []
    skipped: list[dict] = []
    for unit in units:
        anchor = anchors.get(unit.unit_id)
        chosen = anchor.chosen if anchor else None
        if not chosen:
            skipped.append({'unit_id': unit.unit_id, 'file': unit.file,
                            'func': unit.func,
                            'anchor_status': anchor.status_name if anchor
                            else 'NO_ARTIFACT'})
            continue
        if anchor.is_file_scope:
            key = (chosen.file, 1, None, unit.unit_id)
        else:
            key = (chosen.file, chosen.start_line, chosen.end_line)
        if key not in by_key:
            by_key[key] = PatchGroup(key=key, file=chosen.file, anchor=anchor,
                                     safe_id=re.sub(r'[^\w.-]', '__',
                                                    unit.unit_id))
            groups.append(by_key[key])
        by_key[key].units.append(unit)
    return groups, skipped

DRIFT_MAP_CAP = 12

def _drift_briefing(group: dict, ctx: dict) -> str:
    anchor = group.get('anchor') or {}
    chosen = anchor.get('chosen') or {}
    lo, hi = chosen.get('start_line'), chosen.get('end_line')
    status = anchor.get('status')
    pc_rel = group['units'][0].get('pc_file') or group['file']
    pc_path = snapshot_path(ctx['task_dir'], 'pc',
                            ctx['meta']['commits']['pc'], pc_rel)
    hay: list[str] = []
    if lo and hi and os.path.exists(pc_path):
        with open(pc_path, encoding='utf-8', errors='replace') as f:
            hay = f.read().splitlines()
    parts = []
    for u in group['units']:
        lines = []
        level = u.get('pc_file_src')
        if level and level != 'same_path':
            lines.append(f'- PC-side file resolved by content evidence '
                         f'(`{level}`): {u["file"]} -> '
                         f'{u.get("pc_file") or u["file"]}')
        if hay and lo and hi:
            needles = substantive_lines(_pa_context_blocks(u.get('raw_diff')))
            if needles:
                hits, total = span_containment(hay, lo, hi, needles)
                lines.append(f'- source-hunk context inside the anchor '
                             f'function: {hits}/{total} substantive lines '
                             f'matched')
                shown = 0
                for s, ln, mode in fuzzy_pa_line_map(hay, lo, hi, needles):
                    if shown >= DRIFT_MAP_CAP:
                        break
                    if ln and mode == 'similar':
                        lines.append(f'  - `{s}` -> anchor line {ln} '
                                     f'(similar but DRIFTED — adapt the '
                                     f'shape, do not copy this PA line)')
                    else:
                        lines.append(f'  - `{s}` -> ' +
                                     (f'anchor line {ln}' if ln else
                                      'NOT PRESENT in this shape'))
                    shown += 1
                unmatched = sum(1 for _s, ln, _v in
                                fuzzy_pa_line_map(hay, lo, hi, needles)
                                if ln is None)
                if unmatched:
                    lines.append(f'  - {unmatched} context line(s) have no '
                                 f'counterpart in the anchor function: port '
                                 f'the semantic change, do not force the PA '
                                 f'text onto PC; a region that is fully '
                                 f'absent need not be ported at all')
        if status == 'MERGED':
            lines.append(f'- PA function `{u["func"]}` does not exist in the '
                         f'PC file; its body aligns INSIDE the anchor '
                         f'function above (merged by content evidence). '
                         f'Where the patch adds a paired operation (lock/'
                         f'unlock, get/put) at the CALLER, the inlined '
                         f'structure must keep that pair on every exit '
                         f'path of the region')
        if lines:
            parts.append(f"### {u['file']} / {u['func']}\n"
                         + '\n'.join(lines))
    return '\n\n'.join(parts)

def _mechanical_sites(group: dict, base_lines: list[str]) \
        -> list[tuple[int, int, list[str]]] | None:
    core_idx = [i for i, l in enumerate(base_lines) if l.strip()]
    core = [base_lines[i].strip() for i in core_idx]
    edits: list[tuple[int, int, list[str]]] = []
    for order, u in enumerate(group['units']):
        for h in (u.get('hunks') or []):
            adds = [t[2].rstrip('\n') for t in h.get('add_lines', [])]
            if h.get('delete_lines'):
                return None
            if not adds:
                continue
            before = [c for c in h.get('context_before', []) if c.strip()]
            after = [c for c in h.get('context_after', []) if c.strip()]
            if not before:
                return None
            pat = [c.strip() for c in before + after]
            sites = [i for i in range(len(core) - len(pat) + 1)
                     if core[i:i + len(pat)] == pat]
            if len(sites) != 1:
                return None
            insert_after = core_idx[sites[0] + len(before) - 1] + 1
            edits.append((insert_after, order, adds))
    return edits or None

def _apply_sites(base_lines: list[str],
                 edits: list[tuple[int, int, list[str]]]) -> str:
    out: list[str] = []
    cursor = 0
    for at, _order, adds in sorted(edits):
        out.extend(base_lines[cursor:at])
        out.extend(adds)
        cursor = at
    out.extend(base_lines[cursor:])
    return '\n'.join(out) + '\n'

def _mechanical_transplant(group: dict, original_src: str | None) -> str | None:
    if not original_src:
        return None
    edits = _mechanical_sites(group, original_src.splitlines())
    if edits is None:
        return None
    return _apply_sites(original_src.splitlines(), edits)

def _mechanical_sr_block(group: dict, pc_lines: list[str]) -> str | None:
    edits = _mechanical_sites(group, pc_lines)
    if edits is None or len(edits) != 1:
        return None
    at, _order, adds = edits[0]
    lo = max(0, at - 3)
    search = pc_lines[lo:at]
    block = ['<<<<<<< SEARCH'] + search + ['======='] + search + adds \
        + ['>>>>>>> REPLACE']
    return '\n'.join(block)

def _verify_search_block(llm, block: str | None, prompt: str, group: dict,
                         ctx: dict, telemetry: dict,
                         previous_response: str = '',
                         max_retries: int = 1) -> tuple[str | None, str]:
    pc_path = snapshot_path(ctx['task_dir'], 'pc',
                            ctx['meta']['commits']['pc'], group['file'])
    if block is None or not os.path.exists(pc_path):
        return block, ''
    with open(pc_path, encoding='utf-8', errors='replace') as f:
        pc_lines = f.read().splitlines()

    def _matches(text: str | None) -> bool:
        m = _SR_RE.search(text) if text else None
        if not m:
            return True
        s_lines = m.group(1).splitlines()
        if _find_block(pc_lines, s_lines) is not None:
            return True
        stripped = [_LINENO_RE.sub('', l, count=1) for l in s_lines]
        return _find_block(pc_lines, stripped) is not None

    if _matches(block):
        return block, ''
    messages = [('human', prompt), ('ai', previous_response)]
    response = previous_response
    for attempt in range(1, max_retries + 1):
        logger.warning(f'  group {group["safe_id"]}: SEARCH block does not '
                       f'match the PC file, retrying with feedback '
                       f'({attempt}/{max_retries})')
        messages.append(('human', SEARCH_MISMATCH_SUFFIX))
        response = invoke_llm_messages(llm, messages, telemetry)
        retry_block = extract_search_replace(response)
        if retry_block is not None and _matches(retry_block):
            return retry_block, response
        messages.append(('ai', response))
    return block, response

def generate_group(llm, out_dir: str, group: PatchGroup, ctx: dict,
                   empty_retry: int = 1,
                   validate_retry: int = 3, semantic_retry: int = 1,
                   extra_issues: list[str] | None = None) -> PatchedFunction:
    safe_id = group.safe_id
    telemetry = {'prompt_tokens': 0, 'completion_tokens': 0, 'llm_ms': 0.0, 'attempts': 0}
    pf = PatchedFunction(group_id=safe_id, unit_ids=group.unit_ids,
                         file=group.file, funcs=[u.func for u in group.units],
                         anchor_status=group.anchor.status_name)
    t0 = time.monotonic()
    try:
        legacy = group.to_legacy_dict()
        for_group(legacy['units'], ctx)
        ctx['drift'] = _drift_briefing(legacy, ctx)
        prompt = build_prompt(ctx)
        with open(os.path.join(out_dir, 'units', f'{safe_id}.prompt.txt'), 'w') as f:
            f.write(prompt)
        original_src = None
        ctx_path = os.path.join(ctx['task_dir'], 'anchor', f'{safe_id}.ctx.c')
        if os.path.exists(ctx_path):
            with open(ctx_path, encoding='utf-8', errors='replace') as f:
                original_src = f.read().strip()
        response = ''
        func_text = None
        pc_text = ''
        pc_path = snapshot_path(ctx['task_dir'], 'pc',
                                ctx['meta']['commits']['pc'], group.file)
        if os.path.exists(pc_path):
            with open(pc_path, encoding='utf-8', errors='replace') as f:
                pc_text = f.read()
        mech = _mechanical_transplant(legacy, original_src)
        if mech is None and group.anchor.is_file_scope and pc_text:
            sr = _mechanical_sr_block(legacy, pc_text.splitlines())
            if sr is not None:
                func_text = sr
                response = '(mechanical transplant; no LLM call was made)'
                pf.mechanical = True
                logger.info(f'  group {safe_id}: FILE_SCOPE mechanical '
                            f'fast path (pure-add, unique site)')
        if mech is not None:
            group.func_text = mech
            if validate_function(group.to_legacy_dict(), ctx)['ok']:
                func_text = mech
                response = '(mechanical transplant; no LLM call was made)'
                pf.mechanical = True
                logger.info(f'  group {safe_id}: mechanical fast path '
                            f'accepted (pure-add hunks, unique context sites)')
        if not pf.mechanical and func_text is None:
            response = invoke_llm(llm, prompt, telemetry)
            pf.llm_ok = True
        if not pf.mechanical:
            if group.anchor.is_file_scope:
                sr = extract_search_replace(response)
                if sr is None and pc_text:
                    fb = _fence_fallback_sr(response, pc_text)
                    if fb is not None:
                        logger.info(f'  group {safe_id}: fence fallback '
                                    f'synthesized SEARCH/REPLACE')
                        sr = fb
                func_text, retry_response = _verify_search_block(
                    llm, sr, prompt, legacy,
                    ctx, telemetry, previous_response=response,
                    max_retries=validate_retry)
                if retry_response:
                    response = retry_response
                    pf.memory_retries += 1
            else:
                func_text = extract_function(response)
        already_present = _adds_already_present(_unit_dicts(group), pc_text)
        noop_budget = max(empty_retry, semantic_retry) if already_present \
            else empty_retry
        while original_src is not None and func_text is not None \
                and func_text.strip() == original_src \
                and pf.empty_retries < noop_budget:
            pf.empty_retries += 1
            suffix = (RESIDUAL_DEFECT_SUFFIX if already_present
                      else _empty_retry_suffix(legacy))
            logger.warning(f'  group {safe_id}: output identical to original'
                           f'{" (adds already in PC: hunting the residual defect)" if already_present else ""}, '
                           f'retrying with feedback '
                           f'({pf.empty_retries}/{noop_budget})')
            response = invoke_llm_messages(
                llm, [('human', prompt), ('ai', response),
                      ('human', suffix)], telemetry)
            pf.memory_retries += 1
            func_text = extract_function(response)
        if func_text is not None and original_src is not None \
                and func_text.strip() == original_src:
            pf.noop_with_changes = True
        if func_text is not None and not pf.mechanical:
            group.func_text = func_text
            verdict = validate_function(group.to_legacy_dict(), ctx)
            if extra_issues:
                verdict = {'ok': False,
                           'issues': list(verdict['issues']) + list(extra_issues),
                           'notes': verdict.get('notes', []),
                           'addressed_count': verdict.get('addressed_count', 0),
                           'dispositions': verdict.get('dispositions', [])}
            candidates = [(func_text, verdict['issues'], response,
                           verdict.get('addressed_count', 0),
                           verdict.get('dispositions', []))]
            peak_addr, peak_disp = candidates[0][3], candidates[0][4]
            regressions, regress_note = [], ''
            mech_r = sem_r = 0
            while not verdict['ok']:
                has_mech = any(_is_mechanical(i) for i in verdict['issues'])
                if has_mech and mech_r < validate_retry:
                    mech_r += 1
                elif sem_r < semantic_retry:
                    sem_r += 1
                else:
                    break
                pf.validator_retries = mech_r + sem_r
                logger.warning(f'  group {safe_id}: validator retry '
                               f'(mech {mech_r}/{validate_retry}, '
                               f'sem {sem_r}/{semantic_retry}): '
                               f'{verdict["issues"]}')
                suffix = _feedback_suffix(legacy, verdict['issues'])
                if regress_note:
                    suffix += '\n\n' + regress_note
                response = invoke_llm_messages(
                    llm, [('human', prompt), ('ai', response),
                          ('human', suffix)],
                    telemetry)
                retry_text = extract_function(response)
                pf.memory_retries += 1
                if retry_text is None:
                    break
                group.func_text = retry_text
                verdict = validate_function(group.to_legacy_dict(), ctx)
                candidates.append((retry_text, verdict['issues'], response,
                                   verdict.get('addressed_count', 0),
                                   verdict.get('dispositions', [])))
                if candidates[-1][3] > peak_addr:
                    peak_addr, peak_disp = candidates[-1][3], candidates[-1][4]
                elif candidates[-1][3] < peak_addr:
                    kept = {d['unit_id'] for d in peak_disp
                            if d['status'] in ('addressed', 'addressed_with_gaps',
                                               'exempt')}
                    lost = sorted({d['unit_id'] for d in candidates[-1][4]
                                   if d['status'] == 'unaddressed'
                                   and d['unit_id'] in kept})
                    if lost:
                        regressions.append({'lost_units': lost})
                        regress_note = ('your previous version already '
                                        'addressed these units but this '
                                        'version dropped them — restore: '
                                        + ', '.join(lost))
            func_text, best_issues, response, _addr, best_disp = min(
                candidates, key=lambda c: _candidate_key(c[1], c[3]))
            group.func_text = func_text
            pf.validator_issues = best_issues
            pf.dispositions = best_disp
            pf.unit_regressions = regressions
        with open(os.path.join(out_dir, 'units', f'{safe_id}.response.txt'), 'w') as f:
            f.write(response)
        pf.parse_ok = func_text is not None
        if not pf.parse_ok:
            pf.error = ('no search/replace block in response'
                        if group.anchor.is_file_scope
                        else 'no function body in response')
            logger.error(f'  group {safe_id}: {pf.error}')
        else:
            with open(os.path.join(out_dir, 'units', f'{safe_id}.func.c'), 'w') as f:
                f.write(func_text)
            pf.text = func_text
    except Exception as exc:
        pf.error = str(exc)[:300]
        logger.error(f'  group {safe_id} failed: {exc}')
    pf.telemetry = Telemetry(prompt_tokens=telemetry['prompt_tokens'],
                             completion_tokens=telemetry['completion_tokens'],
                             llm_ms=telemetry['llm_ms'],
                             attempts=telemetry['attempts'],
                             wall_ms=(time.monotonic() - t0) * 1000)
    return pf

_HUNK_OLD_RE = re.compile(r'@@ -(\d+)')

def _split_hunks(raw_diff: str) -> list[str]:
    hunks, cur = [], None
    for line in (raw_diff or '').splitlines():
        if line.startswith('@@'):
            if cur:
                hunks.append('\n'.join(cur))
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur:
        hunks.append('\n'.join(cur))
    return hunks

def _apply_one_tier(diff_text: str, pristine: str, rel: str,
                    cmd: list[str], git_init: bool) -> str | None:
    with tempfile.TemporaryDirectory(prefix='direct_apply_') as tmp:
        if git_init:
            r = subprocess.run(['git', 'init', '-q'], cwd=tmp,
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return None
        path = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(path) or tmp, exist_ok=True)
        with open(path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(pristine)
        try:
            r = subprocess.run(cmd, input=diff_text, cwd=tmp,
                               capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or glob_rej(tmp):
            return None
        with open(path, encoding='utf-8', errors='replace') as f:
            return f.read()

def glob_rej(tmp: str) -> bool:
    for root, _dirs, files in os.walk(tmp):
        for f in files:
            if f.endswith(('.rej', '.orig')):
                return True
    return False

def _splices_from_diff(pristine: str, mod: str) -> list[dict]:
    a = pristine.splitlines(keepends=True)
    b = mod.splitlines(keepends=True)
    entries = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        entries.append({'start': i1 + 1, 'end': i2,
                        'text': ''.join(b[j1:j2])})
    return entries

_TIER_COMMANDS = (
    ('exact', ['git', 'apply', '-p1', '-'], True),
    ('context1', ['git', 'apply', '-p1', '-C1', '-'], True),
    ('fuzz3', ['patch', '-p1', '-l', '--fuzz=3', '--forward',
               '--no-backup-if-mismatch'], False),
)

def try_apply_group(group: dict, pristine_by_file: dict[str, str]
                    ) -> tuple[dict[str, list[dict]], str] | None:
    files: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for u in group['units']:
        if not u.get('raw_diff'):
            return None
        rel = u.get('pc_file') or u['file']
        if rel not in pristine_by_file or not pristine_by_file[rel]:
            return None
        hunks = _split_hunks(u['raw_diff'])
        if not hunks:
            return None
        for h in hunks:
            m = _HUNK_OLD_RE.match(h)
            files[rel].append((int(m.group(1)) if m else 0, h))
    splices: dict[str, list[dict]] = {}
    tolerance = 'exact'
    for rel, hs in files.items():
        hs.sort(key=lambda x: x[0])
        diff_text = (f'--- a/{rel}\n+++ b/{rel}\n'
                     + '\n'.join(h[1] for h in hs) + '\n')
        mod = None
        for tier, cmd, git_init in _TIER_COMMANDS:
            mod = _apply_one_tier(diff_text, pristine_by_file[rel], rel,
                                  cmd, git_init)
            if mod is not None:
                tolerance = tier
                break
        if mod is None:
            return None
        splices[rel] = _splices_from_diff(pristine_by_file[rel], mod)
    if not any(splices.values()):
        return None
    logger.info(f'  [{group["safe_id"]}] direct-apply ({tolerance}): '
                f'{sum(len(v) for v in splices.values())} region(s) applied')
    return splices, tolerance

# _UNIT_REGEN_PROMPT = """You are backporting ONE unit of a security fix to a target kernel branch.

# ## Source patch hunk (this unit only)
# {raw_diff}

# ## What this unit is for (extracted intent)
# {intent_text}

# ## Current target file content (post-splice, relevant region)
# ```c
# {spliced_region}
# ```

# ## Symbols the patch uses and their PC-tree status
# {evidence_text}

# ## Contract
# Return ONE SEARCH/REPLACE block: SEARCH must copy target lines VERBATIM from
# the file above (keep it minimal — just enough context to be unique); REPLACE
# is those lines with this unit's change applied, adapted to the PC tree where
# the symbol evidence says the PA-world form does not exist.
# """

# def regenerate_unit(llm, group: PatchGroup, unit_dict: dict,
#                     spliced_text: str, ctx: dict, evidence: list[dict],
#                     budget: int = 2, telemetry: dict | None = None,
#                     missing_lines: list[str] | None = None) -> dict:
#     telemetry = telemetry or {'prompt_tokens': 0, 'completion_tokens': 0,
#                               'llm_ms': 0.0, 'attempts': 0}
#     rec = {'unit_id': unit_dict['unit_id'], 'fallback_used': True,
#            'merged': False, 'rejected_reason': ''}
#     raw = unit_dict.get('raw_diff') or ''
#     intent = (ctx.get('intents') or {}).get(unit_dict['unit_id']) or {}
#     intent_text = json.dumps(
#         {k: intent.get(k) for k in ('required', 'conditional', 'invariants')
#          if intent.get(k)}, indent=1) if intent else '(no intent extracted)'
#     absent = absent_symbols(evidence)
#     evidence_text = '\n'.join(
#         f"- `{e['sym']}` ({e['kind']}): absent from PC tree"
#         + (f", nearest: {', '.join(e['nearest_top3'])}"
#            if e.get('nearest_top3') else '')
#         for e in absent) or '- all patch symbols resolve in the PC tree'
#     missing_text = ''
#     if missing_lines:
#         missing_text = ('\n\n## Named missing lines (source patch lines that '
#                         'must land in this rewrite)\n'
#                         + '\n'.join(f'- {l}' for l in missing_lines))
#     lines = spliced_text.splitlines()
#     prompt = _UNIT_REGEN_PROMPT.format(
#         raw_diff=raw or '(see hunks)', intent_text=intent_text,
#         spliced_region='\n'.join(lines[:400]), evidence_text=evidence_text) \
#         + missing_text
#     response = ''
#     for attempt in range(budget):
#         response = invoke_llm_messages(llm, [('human', prompt)], telemetry) \
#             if attempt == 0 else \
#             invoke_llm_messages(llm, [
#                 ('human', prompt), ('ai', response),
#                 ('human', 'That SEARCH block did not match the file verbatim '
#                           'or the merge failed the checks. Return a corrected '
#                           'SEARCH/REPLACE block.')], telemetry)
#         block = extract_search_replace(response) \
#             or _fence_fallback_sr(response, spliced_text)
#         if block is None:
#             rec['rejected_reason'] = 'no parseable block'
#             continue
#         m = _SR_RE.search(block)
#         s_lines = m.group(1).splitlines()
#         idx = _find_block(lines, s_lines) \
#             or _find_block(lines, [_LINENO_RE.sub('', l, count=1)
#                                    for l in s_lines])
#         if idx is None:
#             rec['rejected_reason'] = 'SEARCH not in spliced file'
#             continue
#         merged = lines[:idx] + m.group(2).splitlines() + lines[idx + len(s_lines):]
#         merged_text = '\n'.join(merged) + '\n'
#         repo = ctx.get('_repo') or ''
#         bad = [i for i, _l in (check_compile(merged_text, spliced_text, repo,
#                                              rel=group.file) if repo else [])]
#         bad += [i for i, _a, _b in file_arity_mismatch_issues(merged_text, spliced_text)]
#         if bad:
#             rec['rejected_reason'] = f'checks failed: {bad[0][:80]}'
#             continue
#         accounted, total = account_add_lines(
#             [t for h in (unit_dict.get('hunks') or [])
#              for t in h.get('add_lines', [])], merged)
#         if not total or len(accounted) / total < 0.8:
#             rec['rejected_reason'] = 'unit still unaddressed after merge'
#             continue
#         rec.update({'merged': True, 'rejected_reason': '',
#                     'merged_text': merged_text})
#         return rec
#     return rec

def run_cve(cve: str, mode: str, name: str, model_name: str,
            llm_threads: int, config_file: str, force: bool,
            empty_retry: int = 1, validate_retry: int = 3,
            semantic_retry: int = 1) -> dict:
    task_dir = os.path.normpath(os.path.join(SRC_ROOT, '..', 'task', mode, cve))
    if not os.path.exists(os.path.join(task_dir, 'metadata.json')):
        raise RuntimeError(f'task artifacts missing for {cve}: run dataset.prepare first')
    with open(os.path.join(task_dir, 'metadata.json')) as f:
        meta = json.load(f)

    units = extract_units(task_dir)
    def _pc_snapshot_missing(rel: str) -> bool:
        return not os.path.exists(snapshot_path(task_dir, 'pc',
                                                meta['commits']['pc'], rel))

    dropped = [u for u in units if _pc_snapshot_missing(u.get('pc_file') or u['file'])]
    for u in dropped:
        logger.warning(f'[{cve}] {u["unit_id"]}: target file absent at PC '
                       f'({u.get("pc_file") or u["file"]}) — unit dropped')
    units = [Unit.from_dict(u) for u in units
             if not _pc_snapshot_missing(u.get('pc_file') or u['file'])]
    unit_ids = {u.unit_id for u in units}

    def _cached_unit_ids(path: str) -> set:
        try:
            with open(path, encoding='utf-8') as f:
                return {u['unit_id'] for u in json.load(f)['units']}
        except (OSError, ValueError, KeyError):
            return set()

    intent_path = os.path.join(task_dir, 'intent', 'intents.json')
    intent_stale = not intent_mod.cache_valid(intent_path)
    intent_gap = not unit_ids <= _cached_unit_ids(intent_path)
    if intent_stale or intent_gap:
        logger.info(f'[{cve}] intent artifacts missing or stale, running locator.intent')
        intent_mod.run_cve(cve, mode, model_name, 1, config_file,
                           force=intent_gap)
    anchor_path = os.path.join(task_dir, 'anchor', 'anchors.json')
    if not unit_ids <= _cached_unit_ids(anchor_path) \
            or not anchor_mod.cache_valid(anchor_path):
        logger.info(f'[{cve}] anchor artifacts missing or stale, running locator.anchor')
        anchor_mod.run_cve(cve, mode, anchor_mod.DIM_WEIGHTS, 5, force=True)
    with open(intent_path) as f:
        intents = {u['unit_id']: (Intent.from_dict(u['intent'])
                                  if u.get('intent') else None)
                   for u in json.load(f)['units']}
    with open(anchor_path) as f:
        anchors = {u['unit_id']: Anchor.from_dict(u)
                   for u in json.load(f)['units']}

    out_dir = os.path.join(task_dir, 'runs', name)
    marker = os.path.join(out_dir, 'metadata.json')
    if os.path.exists(marker) and not force:
        logger.info(f'[{cve}] run {name} already exists, skip (use --force to rerun)')
        return json.load(open(marker))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, 'units'), exist_ok=True)
    _, repo = _get_repos(mode)
    pc_commit = meta['commits']['pc']
    groups, skipped_units = build_groups(units, anchors)
    llm = build_llm(model_name, config_file)
    logger.info(f'[{cve}] run {name}: {len(units)} unit(s) -> {len(groups)} '
                f'group(s), {len(skipped_units)} skipped (no anchor)')

    func_cache: dict[str, str] = {}
    direct_splices: dict[str, dict[str, list[dict]]] = {}
    direct_units: set[str] = set()
    for g in groups:
        rels = {u.target_file for u in g.units}
        pristine_by_file = {}
        for rel in rels:
            if rel not in func_cache:
                func_cache[rel] = _git_show(repo, pc_commit, rel, 10 ** 9) or ''
            pristine_by_file[rel] = func_cache[rel]
        res = try_apply_group(g.to_legacy_dict(), pristine_by_file)
        if res is None:
            continue
        sp, tol = res
        direct_splices[g.safe_id] = sp
        direct_units.update(g.unit_ids)
        logger.info(f'[{cve}] {g.safe_id}: direct-apply ({tol}) — '
                    f'LLM call skipped')
    logger.info(f'[{cve}] direct-apply: {len(direct_splices)}/{len(groups)} '
                f'group(s)')

    da_dispositions: dict[str, list[dict]] = {}
    if direct_splices:
        for g in groups:
            if g.safe_id not in direct_splices:
                continue
            disp: list[dict] = []
            for rel, entries in direct_splices[g.safe_id].items():
                pristine_t = func_cache.get(rel) or ''
                mod_text = ''.join(splice(pristine_t.splitlines(keepends=True),
                                          entries)[0])
                uds = [u.to_dict() for u in g.units if u.target_file == rel]
                if not uds:
                    continue
                g_intents = {u.unit_id: (intents[u.unit_id].to_dict()
                                         if intents.get(u.unit_id) else None)
                             for u in g.units if u.target_file == rel}
                d_list, _iss = _unit_dispositions(uds, mod_text, g_intents,
                                          {'task_dir': task_dir, 'meta': meta,
                                           'mode': mode, 'file': rel})
                disp.extend(d_list)
            da_dispositions[g.safe_id] = disp

    def _ctx(group):
        ints = intents
        return {'units': _unit_dicts(group),
                'intents': {u.unit_id: (ints[u.unit_id].to_dict()
                                        if ints.get(u.unit_id) else None)
                            for u in group.units},
                'anchor': group.anchor.to_dict(), 'task_dir': task_dir,
                'meta': meta, 'safe_id': group.safe_id, 'mode': mode,
                'pb_message': meta.get('pb_message', '')}

    def _one(group):
        if group.safe_id in direct_splices:
            return group, PatchedFunction(
                group_id=group.safe_id,
                unit_ids=group.unit_ids,
                file=group.file,
                funcs=[u.func for u in group.units],
                anchor_status=group.anchor.status_name,
                parse_ok=True, direct_apply=True, splice='ok',
                dispositions=da_dispositions.get(group.safe_id, []))
        return group, generate_group(llm, out_dir, group, _ctx(group),
                                     empty_retry, validate_retry,
                                     semantic_retry)

    if llm_threads > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=llm_threads) as pool:
            pairs = list(pool.map(_one, groups))
    else:
        pairs = [_one(g) for g in groups]

    def build_splices() -> dict[str, list[dict]]:
        splices_by_file: dict[str, list[dict]] = defaultdict(list)
        for group, pf in pairs:
            if pf.direct_apply:
                pf.splice = 'ok'
                for rel, entries in direct_splices[group.safe_id].items():
                    for i, e in enumerate(entries):
                        e.setdefault('group_id', f"{group.safe_id}__da{i}")
                        e.setdefault('unit_id', group.units[0].unit_id)
                    splices_by_file.setdefault(rel, []).extend(entries)
                continue
            if not pf.parse_ok:
                pf.splice = 'skipped_parse_fail'
                continue
            rel = group.file
            if rel not in func_cache:
                func_cache[rel] = _git_show(repo, pc_commit, rel, 10 ** 9) or ''
            if not func_cache[rel]:
                pf.splice = 'skipped_no_pc_file'
                continue
            if pf.anchor_status == 'FILE_SCOPE':
                patch_adds = {t[2].strip() for u2 in group.units
                              for h in u2.hunks
                              for t in h.add_lines if len(t[2].strip()) >= 8}
                rec_view = {'group_id': pf.group_id, '_func_text': pf.text}
                entry = _file_scope_splice(func_cache[rel], rec_view,
                                           patch_adds)
                if rec_view.get('splice'):
                    pf.splice = rec_view['splice']
                if entry is None:
                    if _adds_already_present(_unit_dicts(group),
                                             func_cache[rel]):
                        pf.splice = 'file_scope_no_match_already'
                    continue
            else:
                entry = {'group_id': pf.group_id,
                         'start': group.key[1], 'end': group.key[2],
                         'text': pf.text}
            pf.splice = 'ok'
            splices_by_file[rel].append(entry)
        return splices_by_file

    # placement_on = os.environ.get('SEMPORT_PLACEMENT', '1') != '0'
    placement_on = True

    def _compile_round(sbf: dict[str, list[dict]]):
        issues_by_group: dict[str, list[str]] = defaultdict(list)
        file_level: list[str] = []
        for rel, splices in sbf.items():
            if not splices or rel not in func_cache:
                continue
            try:
                mod_lines, _, regions = splice(
                    func_cache[rel].splitlines(keepends=True), splices,
                    want_map=True)
            except Exception:
                continue
            mod = ''.join(mod_lines)
            found = [(i, a, a) for i, a in
                     check_compile(mod, func_cache[rel], repo, tag=rel, rel=rel)]
            found += file_arity_mismatch_issues(mod, func_cache[rel])
            for issue, ln, dln in found:
                r = next((r for r in regions
                          if r['new_start'] <= ln <= r['new_end']), None)
                if r is None and dln != ln:
                    r = next((r for r in regions
                              if r['new_start'] <= dln <= r['new_end']), None)
                if r is None:
                    file_level.append(f'{rel}: {issue}')
                    continue
                gid = re.sub(r'__da\d+$', '', r['group_id'])
                issues_by_group[gid].append(f'{rel}: {issue}')
            by_gid: OrderedDict[str, list[dict]] = OrderedDict()
            for r in regions:
                by_gid.setdefault(r['group_id'], []).append(r)
            for rid, regs in by_gid.items():
                gid = re.sub(r'__da\d+$', '', rid)
                hit = next(((g, pf) for g, pf in pairs
                            if pf.group_id == gid), None)
                if hit is None or not placement_on:
                    continue
                group, pf = hit
                chosen = getattr(group.anchor, 'chosen', None)
                if group.anchor.is_file_scope or chosen is None:
                    continue
                depth_check = not (pf.mechanical or pf.direct_apply)
                for issue, ln in placement_issues(mod, func_cache[rel], regs,
                                                  chosen.name, check_depth=depth_check):
                    issues_by_group[gid].append(f'{rel}: {issue}')
        return issues_by_group, file_level

    warnings: list[str] = []
    splices_by_file = build_splices()
    for attempt in range(validate_retry + 1):
        issues_by_group, file_level = _compile_round(splices_by_file)
        warnings.extend(file_level)
        for w in file_level:
            logger.warning(f'[{cve}] {w}')
        if not issues_by_group or attempt == validate_retry:
            for gid, g_issues in issues_by_group.items():
                for _, pf in pairs:
                    if pf.group_id == gid:
                        pf.validator_issues = pf.validator_issues + g_issues
            break
        logger.warning(f'[{cve}] post-splice blocking round '
                       f'{attempt + 1}/{validate_retry}: regenerating '
                       f'{sorted(issues_by_group)}')
        for i, (group, pf) in enumerate(pairs):
            g_issues = issues_by_group.get(pf.group_id)
            if not g_issues:
                continue
            if pf.direct_apply:
                pf.direct_apply = False
                direct_splices.pop(group.safe_id, None)
            pairs[i] = (group, generate_group(
                llm, out_dir, group, _ctx(group), empty_retry,
                validate_retry, extra_issues=g_issues))
        splices_by_file = build_splices()

    for i, (group, pf) in enumerate(pairs):
        if pf.parse_ok or pf.direct_apply or pf.mechanical:
            continue
        for attempt in range(validate_retry):
            logger.warning(f'[{cve}] parse-fail regen {attempt + 1}/'
                           f'{validate_retry}: {pf.group_id} ({pf.error})')
            extra = []
            for u in group.units:
                named = [t[2].strip() for h in u.hunks for t in h.add_lines
                         if len(t[2].strip()) >= 8][:4]
                extra.append(
                    f"unit {u.unit_id}: the previous response could not be "
                    f"parsed as code — output the complete function body / "
                    f"SEARCH-REPLACE block and incorporate this unit's "
                    f"change into it"
                    + ("; its source-added lines that must land: "
                       + " | ".join(named) if named else ""))
            pf2 = generate_group(llm, out_dir, group, _ctx(group),
                                 empty_retry, validate_retry, semantic_retry,
                                 extra_issues=extra)
            if pf2.parse_ok:
                pairs[i] = (group, pf2)
                splices_by_file = build_splices()
                break
            pairs[i] = (group, pf2)
            pf = pf2

    for group, pf in pairs:
        unaddr = [d for d in (pf.dispositions or [])
                  if d['status'] == 'unaddressed' and not d.get('surviving_deletions')]
        if not unaddr:
            continue
        splices = splices_by_file.get(group.file)
        if not splices or group.file not in func_cache:
            continue
        mod = ''.join(splice(func_cache[group.file].splitlines(keepends=True),
                             splices)[0])
        final_lines = mod.splitlines()
        by_id = {u.unit_id: u for u in group.units}
        flipped = []
        for d in unaddr:
            u = by_id.get(d['unit_id'])
            if not u:
                continue
            accounted, total = account_add_lines(
                [t for h in u.hunks for t in h.add_lines], final_lines)
            if total and len(accounted) / total >= 0.8:
                d['status'] = 'addressed'
                d['reconciled'] = 'file-level adoption'
                flipped.append(d['unit_id'])
        if flipped:
            logger.info(f'[{cve}] {pf.group_id}: reconciled to addressed '
                        f'(landed via other path): {flipped}')

#     unit_regen_log: list[dict] = []
#     if os.environ.get('SEMPORT_UNIT_REGEN') == '1':
#         for group, pf in pairs:
#             disp = pf.dispositions or []
#             unaddr = {d['unit_id'] for d in disp
#                       if d['status'] == 'unaddressed'}
#             gaps = {d['unit_id']: d.get('missing_lines') or []
#                     for d in disp
#                     if d['status'] == 'addressed_with_gaps'}
#             if not unaddr and not gaps:
#                 continue
#             splices = splices_by_file.get(group.file)
#             if splices and group.file in func_cache:
#                 spliced = ''.join(splice(
#                     func_cache[group.file].splitlines(keepends=True),
#                     splices)[0])
#             elif group.file in func_cache:
#                 spliced = func_cache[group.file]
#             else:
#                 continue
#             gctx = _ctx(group)
#             gctx['_repo'] = repo
#             ev = for_group(_unit_dicts(group), gctx)
#             for u in group.units:
#                 if u.unit_id in unaddr:
#                     missing = None
#                 elif u.unit_id in gaps:
#                     missing = gaps[u.unit_id]
#                 else:
#                     continue
#                 rec = regenerate_unit(llm, group, u.to_dict(), spliced,
#                                       gctx, ev, budget=2,
#                                       missing_lines=missing)
#                 unit_regen_log.append(rec)
#                 if rec['merged']:
#                     spliced = rec['merged_text']
#                     func_cache[group.file] = spliced
#                     splices_by_file[group.file] = [{
#                         'group_id': pf.group_id, 'start': 1,
#                         'end': len(spliced.splitlines()), 'text': spliced}]
#                     for d in pf.dispositions or []:
#                         if d['unit_id'] == u.unit_id:
#                             d['status'] = 'addressed'
#                             d['reconciled'] = ('unit-regen merge (gap lines)'
#                                                if missing else 'unit-regen merge')
#                     logger.info(f'[{cve}] unit-regen merged {u.unit_id}')
#                 else:
#                     logger.warning(f'[{cve}] unit-regen rejected '
#                                    f'{u.unit_id}: {rec["rejected_reason"]}')

    file_diffs: list[str] = []
    for rel, splices in splices_by_file.items():
        orig = func_cache[rel]
        mod, overlap_skipped = splice(orig.splitlines(keepends=True), splices)
        for gid in overlap_skipped:
            for group, pf in pairs:
                if pf.group_id == gid:
                    pf.splice = 'skipped_overlap'
        if any(pf.splice == 'skipped_overlap' for _, pf in pairs):
            mod, _ = splice(orig.splitlines(keepends=True),
                            [s for s in splices if s['group_id'] not in overlap_skipped])
        for s in splices:
            unit_mod, _ = splice(orig.splitlines(keepends=True), [s])
            safe_id = s['group_id']
            diff_text = _git_diff(orig, ''.join(unit_mod), rel)
            with open(os.path.join(out_dir, 'units', f'{safe_id}.diff'), 'w') as f:
                f.write(diff_text)
        file_diffs.append(_git_diff(orig, ''.join(mod), rel))
    output_diff = ''.join(file_diffs)
    with open(os.path.join(out_dir, 'output.diff'), 'w') as f:
        f.write(output_diff)
    final_ok, final_msg = check_apply(output_diff, task_dir, meta)

    for group, pf in pairs:
        gid = pf.group_id
        diff_path = os.path.join(out_dir, 'units', f'{gid}.diff')
        if pf.splice == 'ok' and os.path.exists(diff_path):
            with open(diff_path, encoding='utf-8') as f:
                pf.unit_apply_ok, pf.unit_apply_msg = \
                    check_apply(f.read(), task_dir, meta)
        else:
            pf.unit_apply_ok = False
            pf.unit_apply_msg = pf.splice or 'not spliced'

    group_records = [pf for _, pf in pairs]
    unit_records = []
    for group, pf in pairs:
        for u in group.units:
            unit_records.append({'unit_id': u.unit_id, 'file': u.file,
                                 'func': u.func, 'group_id': pf.group_id,
                                 'anchor_status': pf.anchor_status,
                                 'splice': pf.splice or 'not spliced',
                                 'unit_apply_ok': pf.unit_apply_ok is True})
    unit_records += [{'unit_id': s['unit_id'], 'file': s['file'], 'func': s['func'],
                      'group_id': None, 'anchor_status': s['anchor_status'],
                      'splice': 'skipped_no_anchor', 'unit_apply_ok': False}
                     for s in skipped_units]
    units_lost = [{'unit_id': r['unit_id'], 'reason': r['splice']}
                  for r in unit_records if r['splice'] != 'ok']
    unit_regressions = [e for _, pf in pairs for e in pf.unit_regressions]

    for s in skipped_units:
        warnings.append(f'{s["unit_id"]}: no PC anchor found — its source '
                        f'change is NOT transplanted')
    for group, pf in pairs:
        gid = pf.group_id
        if pf.splice == 'skipped_overlap':
            warnings.append(f'{gid}: splice skipped (overlaps another group) '
                            f'— its source change is NOT in the output')
        elif pf.splice == 'file_scope_no_match':
            warnings.append(f'{gid}: FILE_SCOPE SEARCH never matched the PC '
                            f'file — its source change is NOT in the output')
        elif pf.splice == 'file_scope_direction_violation':
            warnings.append(f'{gid}: FILE_SCOPE output reverted patch-added '
                            f'lines — splice rejected, region left at PC state')
        elif pf.splice == 'skipped_no_pc_file':
            warnings.append(f'{gid}: {pf.splice} — its source change is '
                            f'NOT in the output')
        if pf.noop_with_changes:
            warnings.append(f'{gid}: model returned the PC function unchanged '
                            f'while the source patch has changes')
        if pf.validator_issues:
            warnings.append(f'{gid}: validator issues unresolved after retry: '
                            f'{"; ".join(pf.validator_issues)}')
    for w in warnings:
        logger.warning(f'[{cve}] {w}')

    summary = {
        'cve': cve, 'mode': mode, 'run': name, 'model': model_name,
        'warnings': warnings,
        'units_lost': units_lost,
        'unit_regressions': unit_regressions,
#         'unit_regen': unit_regen_log,
        'groups': [pf.to_record() for pf in group_records],
        'units': unit_records,
        'summary': {
            'units_total': len(units),
            'groups_total': len(group_records),
            'llm_ok': sum(pf.llm_ok for pf in group_records),
            'parse_ok': sum(pf.parse_ok for pf in group_records),
            'spliced': sum(pf.splice == 'ok' for pf in group_records),
            'unit_apply_ok': sum(bool(pf.unit_apply_ok) for pf in group_records),
            'skipped_no_anchor': len(skipped_units),
            'memory_retries': sum(pf.memory_retries for pf in group_records),
            'prompt_tokens': sum(pf.telemetry.prompt_tokens for pf in group_records),
            'completion_tokens': sum(pf.telemetry.completion_tokens
                                     for pf in group_records),
        },
        'final_apply_check': {'ok': final_ok, 'msg': final_msg},
        'output_diff': f'runs/{name}/output.diff',
    }
    with open(marker, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    s = summary['summary']
    logger.info(f'[{cve}] run {name} done: groups {s["groups_total"]}, '
                f'spliced {s["spliced"]}, unit-apply {s["unit_apply_ok"]}, '
                f'final={final_ok}, tokens p={s["prompt_tokens"]} c={s["completion_tokens"]}')
    return summary
