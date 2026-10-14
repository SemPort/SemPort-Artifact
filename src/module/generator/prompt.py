import os

from module.locator.anchor import pa_function_info
from module.locator.units import pa_source_window, render_unit_diff, snapshot_path

SECTION_ORDER = ('role', 'source_patch', 'rationale', 'intent', 'anchor',
                 'drift', 'pa_context', 'output_contract')

PA_FUNC_SPAN_CAP = 400
PA_WINDOW_PAD = 60
PC_FILE_CHAR_CAP = 150_000
PC_WINDOW_PAD = 100

ROLE_TEXT = """
## Role

You are a patch-transplant engineer. A security patch was developed against a source version (PA) and needs to be 
ported to a target version (PC), which may be an older stable branch or a downstream variant of the same project.
The source and target versions may have diverged in code structure, APIs, names, and surrounding context. 
Your task is to reproduce the security behavior of the source patch in PC while adapting the transformation to the
existing target code. The provided patch intent, source changes, and semantically located target context guide the transplant.
"""

OUTPUT_CONTRACT_TEXT = """
## Output requirements

Return ONLY the complete modified PC function inside a ```c code block:
- Return the full function definition, including its signature, ready to replace the target function shown in the anchor section.
- Apply the security-relevant intent of the source patch to PC. Adapt names, types, APIs, control flow, and coding style to the code that actually exists in PC.
- TRANSPLANT FIDELITY has two modes, determined by the nature of the source hunk:
  (a) if the hunk represents a coherent security fix whose structural transformation is part of the fix 
  (e.g., a rewritten parser body, a new validation block, or a restructured loop), 
  preserve the essential transformation and control-flow intent while adapting it to the existing PC structure;
  (b) if the hunk bundles the security fix with unrelated reorganization (e.g., helper extraction, function splitting, or renaming), 
  port only the security-relevant transformation and preserve PC's existing structure. When PC can express the required 
  security behavior with a smaller change, prefer the smaller change rather than reproducing source-side reorganization.
- Prefer the smallest PC-local transformation that fully satisfies the required security behavior. 
  Do not reproduce source-side restructuring merely for structural similarity.
- The source patch's actions describe what was changed in PA, not literal edits that must be reproduced in PC. 
  Map each security-relevant action to the semantically corresponding construct in PC. The target construct may use 
  different names, APIs, expressions, or control-flow structure. 
- If the corresponding security behavior is already provided by PC, do not add a duplicate check, guard, early return, 
  or other redundant change. An invariant that already holds in PC should be preserved, not reimplemented. 
- If a source action has no direct syntactic counterpart in PC, do not simply reproduce the source structure. 
  Adapt the action to the closest existing PC construct that can provide the required security behavior. 
- If satisfying an action or invariant would require unrelated restructuring of PC, prefer a smaller PC-local transformation 
  that preserves the required security behavior. Do not force source-side control-flow structure onto PC. 
- Changes marked INCIDENTAL represent refactoring, cleanup, or other non-security changes that accompanied the fix. 
  Do not port them unless they are necessary to implement a required security behavior. 
- Do not make changes solely to avoid returning the original PC function. If PC already satisfies the required security behavior, 
  preserve it unchanged rather than introducing a redundant, unrelated, or cosmetic modification. 
- Do not introduce unrelated refactoring, speculative API changes, or compatibility changes that are not required by the source patch. 
- Do not add or remove blank lines solely to imitate the source patch. Preserve the formatting and coding style of the surrounding PC code. 
- Wrap lines longer than 80 columns following the kernel style used by the surrounding PC code.
- No unified diff, no SEARCH/REPLACE markers, no commentary.
"""

FILE_SCOPE_CONTRACT_TEXT = """
## Output requirements

The change applies at file scope, such as a top-level struct, table, macro, declaration, or other non-function region. 
Return exactly ONE search/replace block inside a ```c code block:

<<<<<<< SEARCH
<exact contiguous lines from the numbered PC file to replace, copied verbatim from the anchor section without line-number prefixess>
=======
<replacement lines implementing the security-relevant patch intent in PC>
>>>>>>> REPLACE

- Keep the SEARCH block as small as possible (only the lines the change touches plus the minimum context needed to be unambiguous)
- Wrap lines longer than 80 columns following the kernel style used by the surrounding PC code.
- No unified diff, no SEARCH/REPLACE markers, no commentary.
"""

def render_source_patch(ctx: dict) -> str:
    blocks = []
    for unit in ctx['units']:
        block = (f"### Hunk scope: {unit['file']} / {unit['func']}\n\n"
                 f"```diff\n{render_unit_diff(unit)}\n```")
        blocks.append(block)
    return ('## Source patch to transplant (PA -> PB)\n\n'
            + '\n\n'.join(blocks))

def render_rationale(ctx: dict) -> str:
    message = (ctx.get('pb_message') or '').strip()
    if not message:
        return ''
    return ('## Why the patch was written (source commit message)\n\n'
            'The author\'s account of the change — context for the port, not\n'
            'a template to force onto PC:\n\n'
            f'{message}')

def render_intent(ctx: dict) -> str:
    blocks = []
    for unit in ctx['units']:
        intent = (ctx.get('intents') or {}).get(unit['unit_id'])
        if not intent:
            continue
        lines = [f"### Intent of {unit['file']} / {unit['func']}", '']
        if intent.get('security_goal'):
            lines.append(f"- Security goal: {intent['security_goal']}")
        if intent.get('hunk_role') == 'incidental':
            evidence = (intent.get('role_evidence')
                        or 'classified as refactor/cleanup, not part of the fix')
            lines.append(f"- Hunk role: INCIDENTAL — {evidence}; "
                         f"omit unless the security goal needs it")
        if intent.get('behavioral_change'):
            lines.append(f"- Behavioral change (PC must): "
                         f"{intent['behavioral_change']}")
        if intent.get('structural_change'):
            lines.append(f"- Structural change (what the patch did in PA): "
                         f"{intent['structural_change']}")
        if intent.get('required'):
            lines.append('- Required after transplant:')
            lines += [f'  {i + 1}. {a}'
                      for i, a in enumerate(intent['required'])]
        if intent.get('conditional'):
            lines.append('- Conditional — apply ONLY if the condition holds '
                         'in the PC code; otherwise keep the PC code '
                         'unchanged:')
            for i, c in enumerate(intent['conditional']):
                cond = f" — but only {c['condition']}" if c.get('condition') else ''
                lines.append(f"  {i + 1}. {c['action']}{cond}")
        if intent.get('invariants'):
            lines.append('- Invariants that must hold after the transplant:')
            lines += [f'  - {s}' for s in intent['invariants']]
        if not intent.get('required') and not intent.get('conditional') \
                and intent.get('actions'):
            lines.append('- Actions:')
            lines += [f'  {i + 1}. {a}' for i, a in enumerate(intent['actions'])]
        if intent.get('summary') and not intent.get('behavioral_change'):
            lines.append(f"- Summary: {intent['summary']}")
        blocks.append('\n'.join(lines))
    if not blocks:
        return '(patch intent unavailable for this unit)'
    return '## Patch intent (what the hunks do, version-independent)\n\n' \
        + '\n\n'.join(blocks)

def _numbered(source: str, start_line: int) -> str:
    return '\n'.join(f'{start_line + i:>6} {line}'
                     for i, line in enumerate(source.splitlines()))

def _pc_hunk_windows(pc_lines: list[str], ctx: dict, pad: int = PC_WINDOW_PAD,
                     max_windows: int = 4) -> str | None:
    keys = [l.strip() for l in pc_lines]
    spans = []
    cursor = 0
    for unit in ctx['units']:
        for h in unit.get('hunks', []):
            ctx_lines = [l for l in h.get('context_before', []) if l.strip()] \
                or [l for l in h.get('context_after', []) if l.strip()]
            pat = [l.strip() for l in ctx_lines]
            if not pat:
                continue
            seed = h.get('old_start')
            idx = None
            for i in range(cursor, len(keys) - len(pat) + 1):
                if keys[i:i + len(pat)] == pat:
                    if seed is None or idx is None \
                            or abs(i - seed) < abs(idx - seed):
                        idx = i
                    if seed is None:
                        break
            if idx is None:
                continue
            cursor = max(cursor, idx + 1)
            spans.append((max(0, idx - pad), min(len(pc_lines),
                                                 idx + len(pat) + pad)))
    if not spans:
        return None
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
        if len(merged) == max_windows:
            break
    blocks = []
    for lo, hi in merged:
        if blocks:
            blocks.append(f'   ... [lines {merged[-1][1] + 1}-{lo} not shown] ...')
        blocks.append('\n'.join(f'{lo + 1 + i:>6} {pc_lines[lo + i]}'
                                for i in range(hi - lo)))
    note = (f'{len(pc_lines)} lines total; only the regions near the change '
            f'are shown — line numbers are real file line numbers')
    return f'({note})\n' + '\n'.join(blocks)

def render_anchor(ctx: dict) -> str:
    anchor = ctx.get('anchor')
    task_dir, meta = ctx['task_dir'], ctx['meta']
    u0 = ctx['units'][0]
    rel = u0.get('pc_file') or u0['file']
    if anchor and anchor.get('chosen'):
        chosen = anchor['chosen']
        ctx_path = os.path.join(task_dir, 'anchor',
                                f"{ctx['safe_id']}.ctx.c")
        if os.path.exists(ctx_path):
            with open(ctx_path, encoding='utf-8', errors='replace') as f:
                src = f.read()
            return (f"## Anchor: where the change applies in PC\n\n"
                    f"Target function `{chosen['name']}` at {chosen['file']}:"
                    f"{chosen['start_line']}-{chosen['end_line']} "
                    f"(anchor status: {anchor['status']}, "
                    f"confidence {anchor['confidence']:.2f}).\n"
                    f"Line numbers below are real file line numbers; the "
                    f"function starts at line {chosen['start_line']}.\n\n"
                    f"```c\n{_numbered(src, chosen['start_line'])}\n```")
    pc_path = snapshot_path(task_dir, 'pc', meta['commits']['pc'], rel)
    if os.path.exists(pc_path):
        with open(pc_path, encoding='utf-8', errors='replace') as f:
            src = f.read()
        why = ('change is at file scope: a top-level struct/table/macro region'
               if anchor and anchor.get('status') == 'FILE_SCOPE'
               else 'anchor UNRESOLVED')
        if anchor and anchor.get('status') == 'FILE_SCOPE' \
                and len(src) > PC_FILE_CHAR_CAP:
            windowed = _pc_hunk_windows(src.splitlines(), ctx)
            if windowed:
                return (f"## Anchor: target file in PC ({why}; file exceeds "
                        f"the size budget, hunk-centred excerpts provided)"
                        f"\n\nFile: {rel}\n\n```c\n{windowed}\n```")
        return (f"## Anchor: target file in PC ({why}, whole "
                f"file provided)\n\nFile: {rel}\n\n"
                f"```c\n{_numbered(src, 1)}\n```")
    return '(PC target source unavailable for this unit)'

def _pa_unit_context(task_dir: str, meta: dict, unit: dict) -> str:
    info = pa_function_info(task_dir, meta, unit)
    if info is None:
        return pa_source_window(task_dir, meta, unit)
    name, lo, hi = info['name'], info['start_line'], info['end_line']
    if hi - lo + 1 <= PA_FUNC_SPAN_CAP:
        return (f'(function {name}, lines {lo}-{hi})\n'
                + _numbered(info['source'], lo))
    linenos = [n for h in unit['hunks'] for n in h['delete_lineno']] or [lo]
    wlo = max(lo, min(linenos) - PA_WINDOW_PAD)
    whi = min(hi, max(linenos) + PA_WINDOW_PAD)
    excerpt = '\n'.join(info['source'].splitlines()[wlo - lo: whi - lo + 1])
    return (f'(function {name} is large; excerpt of lines {wlo}-{whi} '
            f'within {lo}-{hi})\n' + _numbered(excerpt, wlo))

def render_drift(ctx: dict) -> str:
    text = (ctx.get('drift') or '').strip()
    if not text:
        return ''
    return ('## Structural alignment (computed: source hunk context vs the '
            'PC anchor)\n\n'
            'Line-level facts computed over the PC file. A line marked NOT '
            'PRESENT does not exist in the anchor function in that shape — '
            'port the semantic change instead of forcing the PA text onto '
            'PC.\n\n' + text)

def render_pa_context(ctx: dict) -> str:
    blocks = []
    seen: set[str] = set()
    for unit in ctx['units']:
        block = _pa_unit_context(ctx['task_dir'], ctx['meta'], unit)
        if block in seen:
            continue
        seen.add(block)
        blocks.append(block)
    return ('## PA source around the change\n\n'
            'For orientation only: use PA to understand what the source\n'
            'patch changed. Differences between PA and PC beyond the patch\n'
            'hunks themselves are pre-existing drift between the branches —\n'
            'never port PA-only statements into PC.\n\n```c\n'
            + '\n\n'.join(blocks) + '\n```')

def render_output_contract(ctx: dict) -> str:
    anchor = ctx.get('anchor')
    if anchor and anchor.get('status') == 'FILE_SCOPE':
        return FILE_SCOPE_CONTRACT_TEXT
    return OUTPUT_CONTRACT_TEXT

def build_prompt(ctx: dict) -> str:
    renderers = {
        'role': lambda c: ROLE_TEXT,
        'source_patch': render_source_patch,
        'rationale': render_rationale,
        'intent': render_intent,
        'anchor': render_anchor,
        'drift': render_drift,
        'pa_context': render_pa_context,
        'output_contract': render_output_contract,
    }
    parts = [text for s in SECTION_ORDER
             if s in renderers
             for text in [renderers[s](ctx)] if text]
    return '\n\n'.join(parts) + '\n'
