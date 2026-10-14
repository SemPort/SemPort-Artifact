# import argparse
import difflib
import json
import os
import re
import subprocess
from collections import Counter

from dataset.prepare import _get_repos
from module.locator.units import (_pa_context_blocks, extract_units,
                                  pa_position_seeds, snapshot_path,
                                  span_containment, substantive_lines)
from utils.logger import get_logger
from utils.parser.ast_parser import (ASTParser, declarator_identifier,
                                     iter_function_definitions)

logger = get_logger(__name__)

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIM_WEIGHTS = {'calls': 0.30, 'name': 0.25, 'text': 0.25, 'ast': 0.10, 'sig': 0.10}
ALGO_VERSION = 'lean-2-disc'
SCORE_MIN = 0.15
EXPAND_SCORE = 0.45
MERGE_MIN = 0.5
REHOME_CONT_MAX = 0.15
REHOME_SCORE_MIN = 0.25
REHOME_RATIO = 2.0
# TOP_K = 5
MAX_NEIGHBOUR_LINES = 4000
MAX_GREP_FILES = 10
# CROSS_INDEX_FILES = 2

_KEYWORD_CALLS = {'if', 'for', 'while', 'switch', 'return', 'sizeof',
                  'likely', 'unlikely'}
_SYSCALL_RE = re.compile(r'^(?:COMPAT_)?SYSCALL_DEFINE[0-9]?\s*\(\s*(\w+)\s*[,)]')
_SYSCALL_HEAD_RE = re.compile(r'^(?:COMPAT_)?SYSCALL_DEFINE[0-9]*$')
_SYS_PREFIX_RE = re.compile(r'^(?:__se_|__do_|__compat_)?(?:compat_)?sys_')
_CALL_RE = re.compile(r'([A-Za-z_]\w*)\s*\(')

def normalize_name(name: str) -> str:
    m = _SYSCALL_RE.match(name.strip())
    if m:
        return m.group(1).lower()
    return _SYS_PREFIX_RE.sub('', name.strip()).lower()

def name_sim(a: str, b: str) -> float:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.5
    return difflib.SequenceMatcher(None, na, nb).ratio()

def _calls_in(src: str) -> set[str]:
    return {m for m in _CALL_RE.findall(src) if m not in _KEYWORD_CALLS}

def _token_stream(src: str) -> list[str]:
    return re.findall(r'\w+', src)

def _ast_sequence(fn_node) -> list[str]:
    out = []
    stack = [fn_node]
    while stack:
        n = stack.pop()
        if n.type != 'comment':
            out.append(n.type)
        stack.extend(reversed(n.children))
    return out

def _param_count(fn_node) -> int:
    decl = fn_node.child_by_field_name('declarator')
    while decl is not None and decl.type != 'function_declarator':
        decl = next((c for c in decl.children if 'declarator' in c.type), None)
    if decl is None:
        return -1
    params = decl.child_by_field_name('parameters')
    if params is None:
        return -1
    return max(0, len([c for c in params.children if c.type != 'comment']))

_MACRO_FUNC_RE = re.compile(r'^([A-Z][A-Z0-9_]+)\s*\(')

def _macro_functions(lines: list[str]) -> list[dict]:
    funcs = []
    for i, line in enumerate(lines):
        m = _MACRO_FUNC_RE.match(line)
        if not m:
            continue
        start = None
        for j in range(i, min(i + 8, len(lines))):
            scan = lines[j] if j != i else line[m.end():]
            brace, semi = scan.find('{'), scan.find(';')
            if brace != -1 and (semi == -1 or brace < semi):
                start = j
                break
            if semi != -1:
                break
        if start is None:
            continue
        depth, end = 0, None
        for j in range(start, len(lines)):
            depth += lines[j].count('{') - lines[j].count('}')
            if depth <= 0:
                end = j
                break
        if end is None:
            continue
        body = '\n'.join(lines[i:end + 1])
        funcs.append({'name': line.strip(),
                      'start_line': i + 1, 'end_line': end + 1,
                      'source': body, 'calls': _calls_in(body),
                      'ast_seq': [], 'params': -1})
    return funcs

def _recompute_features(fn: dict, lines: list[str]) -> dict:
    body = '\n'.join(lines[fn['start_line'] - 1:fn['end_line']])
    fn['source'] = body
    fn['calls'] = _calls_in(body)
    try:
        fn['ast_seq'] = _ast_sequence(ASTParser.parse_code(body).root_node)
    except Exception:
        fn['ast_seq'] = []
    return fn

def index_functions(src: str) -> list[dict]:
    try:
        tree = ASTParser.parse_code(src)
    except Exception:
        return []
    funcs = []
    lines = src.splitlines()
    for fn in iter_function_definitions(tree.root_node):
        name = declarator_identifier(fn.child_by_field_name('declarator'))
        if not name:
            continue
        start, end = fn.start_point[0] + 1, fn.end_point[0] + 1
        body = '\n'.join(lines[start - 1:end])
        funcs.append({'name': name, 'start_line': start, 'end_line': end,
                      'source': body, 'calls': _calls_in(body),
                      'ast_seq': _ast_sequence(fn), 'params': _param_count(fn)})
    macros = _macro_functions(lines)
    for fn in funcs:
        inside = [m for m in macros
                  if fn['start_line'] < m['start_line'] and m['end_line'] <= fn['end_line']]
        if inside:
            fn['end_line'] = min(m['start_line'] for m in inside) - 1
            _recompute_features(fn, lines)
    known = {normalize_name(f['name']) for f in funcs}
    for mf in macros:
        if normalize_name(mf['name']) not in known:
            funcs.append(mf)
    return funcs

def _function_context(path: str, linenos: list[int]) -> dict | None:
    if not os.path.exists(path) or not linenos:
        return None
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    lo, hi = min(linenos), max(linenos)
    for m in _macro_functions(lines):
        if m['start_line'] <= lo and hi <= m['end_line']:
            return m
    try:
        tree = ASTParser.parse_file(path)
    except Exception:
        return None
    for fn in iter_function_definitions(tree.root_node):
        if fn.start_point[0] + 1 <= lo and hi <= fn.end_point[0] + 1:
            name = declarator_identifier(fn.child_by_field_name('declarator'))
            if not name:
                continue
            start, end = fn.start_point[0] + 1, fn.end_point[0] + 1
            info = {'name': name, 'start_line': start, 'end_line': end,
                    'source': ''.join(lines[start - 1:end]),
                    'calls': set(), 'ast_seq': _ast_sequence(fn),
                    'params': _param_count(fn)}
            info['calls'] = _calls_in(info['source'])
            inside = [m for m in _macro_functions(lines)
                      if start < m['start_line'] and m['end_line'] <= end]
            if inside and hi < min(m['start_line'] for m in inside):
                info['end_line'] = min(m['start_line'] for m in inside) - 1
                _recompute_features(info, lines)
            return info
    return None

def pa_function_info(task_dir: str, meta: dict, unit: dict) -> dict | None:
    path = snapshot_path(task_dir, 'pa', meta['commits']['pa'], unit['file'])
    return _function_context(path, pa_position_seeds(unit))

def pb_function_info(task_dir: str, meta: dict, unit: dict) -> dict | None:
    path = snapshot_path(task_dir, 'pb', meta['commits']['pb'], unit['file'])
    linenos = [n for h in unit['hunks'] for n in h['add_lineno']] or \
        [n for h in unit['hunks'] for n in h['delete_lineno']]
    return _function_context(path, linenos)

def score_candidate(pa: dict, cand: dict, hunk_calls: set[str],
                    dims: dict) -> tuple[float, dict]:
    signals = {}
    if 'name' in dims:
        signals['name'] = name_sim(pa['name'], cand['name'])
    if 'calls' in dims:
        pa_calls, pc_calls = pa['calls'], cand['calls']
        jac = (len(pa_calls & pc_calls) / len(pa_calls | pc_calls)
               if pa_calls | pc_calls else 0.0)
        cov = (len(hunk_calls & pc_calls) / len(hunk_calls)
               if hunk_calls else 0.0)
        signals['calls'] = 0.7 * jac + 0.3 * cov
    if 'text' in dims:
        sm = difflib.SequenceMatcher(None, _token_stream(pa['source']),
                                     _token_stream(cand['source']))
        signals['text'] = sm.quick_ratio() * 0.5 + sm.ratio() * 0.5
    if 'ast' in dims:
        sm = difflib.SequenceMatcher(None, pa['ast_seq'], cand['ast_seq'])
        signals['ast'] = sm.ratio()
    if 'sig' in dims:
        a, b = pa['params'], cand['params']
        signals['sig'] = (min(a, b) / max(a, b)
                          if a >= 0 and b > 0 else (1.0 if a == b else 0.0))
    score = sum(DIM_WEIGHTS[d] * s for d, s in signals.items())
    total = sum(DIM_WEIGHTS[d] for d in signals)
    return (score / total if total else 0.0), signals

def _git_show(repo: str, commit: str, path: str, max_lines: int) -> str | None:
    r = subprocess.run(['git', '-C', repo, 'show', f'{commit}:{path}'],
                       capture_output=True, text=True, errors='replace', timeout=30)
    if r.returncode != 0:
        return None
    lines = r.stdout.splitlines()
    return '\n'.join(lines[:max_lines]) if len(lines) > max_lines else r.stdout

def _grep_name(pa_name: str) -> str | None:
    head = pa_name.split('(')[0].split()
    name = head[-1] if head else ''
    return None if not name or _SYSCALL_HEAD_RE.match(name) else name

def _function_elsewhere(repo: str, pc_commit: str, pa: dict, pc_file: str,
                        searched: set[str], hunk_calls: set[str], dims: dict,
                        best_in: float, cap: int = 3) -> list[dict]:
    gname = _grep_name(pa['name'])
    if not gname:
        return []
    hits = _cross_file_hits(repo, pc_commit, gname, pc_file, set(searched))
    out = []
    for cf in hits[:cap]:
        if cf == pc_file:
            continue
        csrc = _git_show(repo, pc_commit, cf, MAX_NEIGHBOUR_LINES)
        if not csrc:
            continue
        for cand in index_functions(csrc):
            score, signals = score_candidate(pa, cand, hunk_calls, dims)
            if score < EXPAND_SCORE or score <= best_in:
                continue
            out.append({'file': cf,
                        **{k: v for k, v in cand.items() if k not in ('source', 'calls', 'ast_seq')},
                        'score': round(score, 4),
                        'signals': {d: round(v, 4) for d, v in signals.items()},
                        '_source': cand['source']})
    out.sort(key=lambda c: c['score'], reverse=True)
    return out

def _cross_file_hits(repo: str, commit: str, name: str, unit_file: str,
                     searched: set[str]) -> list[str]:
    r = subprocess.run(['git', '-C', repo, 'grep', '-l', name, commit, '--', '*.c'],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return []
    prefix = f'{commit}:'
    pa_dir = os.path.dirname(unit_file)

    def path_rank(f: str) -> float:
        sim = difflib.SequenceMatcher(None, os.path.dirname(f), pa_dir).ratio()
        return sim + (0.3 if name.lower() in f.lower() else 0.0)

    files = [p[len(prefix):] if p.startswith(prefix) else p
             for p in r.stdout.splitlines() if p.endswith('.c')]
    files = sorted((f for f in files if f not in searched), key=path_rank, reverse=True)
    return files[:MAX_GREP_FILES]

def locate_unit(task_dir: str, meta: dict, unit: dict, repo: str,
                pc_commit: str, dims: dict,
                topk: int) -> dict:
    unit_id = unit['unit_id']
    hunk_calls = {c for h in unit['hunks']
                  for _, t, _ in h['delete_lines'] + h['add_lines']
                  for c in _calls_in(t)}
    pa = pa_function_info(task_dir, meta, unit)
    result = {'unit_id': unit_id, 'file': unit['file'], 'func': unit['func'],
              'pa_func': pa['name'] if pa else None,
              'status': 'UNRESOLVED', 'confidence': 0.0,
              'candidates': [], 'chosen': None}

    pc_file = unit.get('pc_file') or unit['file']

    if pa is None:
        pc_src = _git_show(repo, pc_commit, pc_file, 10 ** 9)
        logger.warning(f'  [{unit_id}] no PA function span; '
                       f'anchor at file scope of {pc_file}')
        result['status'] = 'FILE_SCOPE'
        result['chosen'] = {'file': pc_file, 'name': '(file scope)',
                            'start_line': 1,
                            'end_line': len(pc_src.splitlines()) if pc_src else 0}
        return result

    pc_src = _git_show(repo, pc_commit, pc_file, 10 ** 9)
    searched = {pc_file}
    if pc_src is None:
        logger.warning(f'  [{unit_id}] pc file missing in repo: {pc_file}')
        pool = []
    else:
        pool = [(pc_file, c) for c in index_functions(pc_src)]
        if not pool:
            logger.warning(f'  [{unit_id}] no PC functions indexed in '
                           f'{pc_file}; anchoring at file scope of the '
                           f'resolved counterpart')
            result['status'] = 'FILE_SCOPE'
            result['chosen'] = {'file': pc_file, 'name': '(file scope)',
                                'start_line': 1,
                                'end_line': len(pc_src.splitlines())}
            return result

    scored = []
    for cf, cand in pool:
        score, signals = score_candidate(pa, cand, hunk_calls, dims)
        scored.append({'file': cf,
                       **{k: v for k, v in cand.items() if k not in ('source', 'calls', 'ast_seq')},
                       'score': round(score, 4),
                       'signals': {d: round(v, 4) for d, v in signals.items()},
                       '_source': cand['source']})
    scored.sort(key=lambda c: c['score'], reverse=True)

    if not scored or scored[0]['score'] < EXPAND_SCORE:
        best_in = scored[0]['score'] if scored else -1.0
        elsewhere = _function_elsewhere(repo, pc_commit, pa, pc_file,
                                        searched, hunk_calls, dims,
                                        best_in)
        if elsewhere:
            scored.extend(elsewhere)
            scored.sort(key=lambda c: c['score'], reverse=True)
            logger.info(f'  [{unit_id}] narrowed cross-file check: '
                        f'{elsewhere[0]["file"]} scores '
                        f'{elsewhere[0]["score"]} over in-file {best_in}')

    scored = scored[:topk]
    best = scored[0] if scored else None
    result['candidates'] = [{k: v for k, v in c.items() if k != '_source'}
                            for c in scored]
    if best is None or best['score'] < SCORE_MIN:
        result['status'] = 'UNRESOLVED'
        return result

    same_file = best['file'] == pc_file
    if same_file and best['signals'].get('name') == 1.0:
        result['status'] = 'EXACT'
    elif same_file and best['signals'].get('name', 0) >= 0.8:
        result['status'] = 'FUZZY'
    elif same_file:
        result['status'] = 'STRUCTURAL'
    else:
        result['status'] = 'CROSS_FILE'
    result['confidence'] = best['score']
    result['chosen'] = {k: v for k, v in best.items() if k != '_source'}
    result['_chosen_source'] = best['_source']

    if result['status'] in ('STRUCTURAL', 'FUZZY') and pc_src:
        hay = pc_src.splitlines()
        lo, hi = best['start_line'], best['end_line']
        needles = substantive_lines(_pa_context_blocks(unit.get('raw_diff')))
        if needles and hi >= lo:
            span_set = {l.strip() for l in hay[lo - 1:hi]}
            cur_cont = (sum(1 for s in dict.fromkeys(needles) if s in span_set)
                        / len(needles))
            if cur_cont < REHOME_CONT_MAX:
                sigs = _hunk_call_signatures(unit.get('raw_diff'))
                cur_score = _span_post_score(hay, lo, hi, needles, sigs)
                best_alt, best_alt_score = None, cur_score
                for cf, cand in pool:
                    if cf != pc_file:
                        continue
                    if cand['start_line'] == lo and cand['end_line'] == hi:
                        continue
                    s = _span_post_score(hay, cand['start_line'],
                                         cand['end_line'], needles, sigs)
                    if s > best_alt_score:
                        best_alt, best_alt_score = cand, s
                if best_alt is not None \
                        and best_alt_score >= REHOME_SCORE_MIN \
                        and best_alt_score >= REHOME_RATIO * max(cur_score, 1e-6):
                    result['rehome'] = {
                        'prev_status': result['status'],
                        'prev_name': best['name'],
                        'prev_containment': round(cur_cont, 4),
                        'new_score': round(best_alt_score, 4)}
                    result['status'] = 'REHOME'
                    result['confidence'] = round(best_alt_score, 4)
                    alt = {'file': cf,
                           **{k: v for k, v in best_alt.items()
                              if k not in ('source', 'calls', 'ast_seq')}}
                    result['chosen'] = alt
                    result['_chosen_source'] = best_alt['source']
                    logger.info(
                        f'  [{unit_id}] post-verify rehome: {best["name"]} '
                        f'(containment {cur_cont:.2f}) -> '
                        f'{best_alt["name"]} (score {best_alt_score:.2f})')
    return result

def _containment_merge(results: list[dict], units: list[dict], meta: dict,
                       repo: str,
                       task_dir: str | None = None) -> None:
    commit = meta['commits']['pc']
    hay_cache: dict[str, list[str] | None] = {}
    spans_cache: dict[str, list[tuple[str, int, int]]] = {}
    for unit, res in zip(units, results):
        if res['status'] not in ('UNRESOLVED', 'CROSS_FILE'):
            continue
        pc_file = unit.get('pc_file') or unit['file']
        if pc_file not in hay_cache:
            src = _git_show(repo, commit, pc_file, 10 ** 9)
            hay_cache[pc_file] = src.splitlines() if src else None
            spans_cache[pc_file] = ([(f['name'], f['start_line'], f['end_line'])
                                     for f in index_functions(src)]
                                    if src else [])
        hay = hay_cache[pc_file]
        if not hay:
            continue
        pa_name = res.get('pa_func') or unit['func']
        if re.search(r'\b' + re.escape(pa_name) + r'\b', '\n'.join(hay)):
            continue
        needles = substantive_lines(_pa_context_blocks(unit.get('raw_diff')))
        if not needles:
            continue

        def _ratio(hits_total: tuple[int, int]) -> float:
            hits, total = hits_total
            return hits / total if total else 0.0

        target = None
        for other, ores in zip(units, results):
            if other is unit or ores['status'] == 'FILE_SCOPE':
                continue
            if not ores.get('chosen'):
                continue
            oc = ores['chosen']
            if (oc.get('file') or other.get('pc_file') or other['file']) != pc_file:
                continue
            hits, total = span_containment(hay, oc['start_line'], oc['end_line'],
                                           needles)
            ratio = _ratio((hits, total))
            if total and ratio >= MERGE_MIN:
                target = (oc['name'], oc['start_line'], oc['end_line'], ratio)
                break
        if target is None:
            best_ratio, best_span = 0.0, None
            for name, lo, hi in spans_cache[pc_file]:
                hits, total = span_containment(hay, lo, hi, needles)
                ratio = _ratio((hits, total))
                if ratio > best_ratio:
                    best_ratio, best_span = ratio, (name, lo, hi)
            if best_span and best_ratio >= MERGE_MIN:
                target = (best_span[0], best_span[1], best_span[2], best_ratio)
        if target is None:
            continue
        name, lo, hi, ratio = target
        res['merge'] = {'prev_status': res['status'],
                        'containment': round(ratio, 4)}
        res['status'] = 'MERGED'
        res['confidence'] = round(ratio, 4)
        res['chosen'] = {'file': pc_file, 'name': name, 'start_line': lo,
                         'end_line': hi, 'score': round(ratio, 4),
                         'signals': {'containment': round(ratio, 4)}}
        res['_chosen_source'] = '\n'.join(hay[lo - 1:hi]) + '\n'
        logger.info(f'  [{res["unit_id"]}] `{pa_name}` absent from {pc_file}: '
                    f'containment {ratio:.2f} -> MERGED into {name} ({lo}-{hi})')

_CALLSIG_RE = re.compile(r'([A-Za-z_]\w*)\s*\(([^;{}]*?)\)')

def _hunk_call_signatures(raw_diff: str | None) -> list[tuple[str, frozenset]]:
    sigs: list[tuple[str, frozenset]] = []
    for line in substantive_lines(_pa_context_blocks(raw_diff)):
        for m in _CALLSIG_RE.finditer(line):
            name = m.group(1)
            if name in _KEYWORD_CALLS or len(name) < 5:
                continue
            args = frozenset(t for t in re.split(r'[(),\s*&]+', m.group(2))
                             if len(t) >= 2 and t != name)
            if args:
                sigs.append((name, args))
    return sigs

def _span_post_score(hay: list[str], lo: int, hi: int,
                     needles: list[str],
                     sigs: list[tuple[str, frozenset]]) -> float:
    if hi < lo:
        return 0.0
    span = hay[lo - 1:hi]
    span_set = {l.strip() for l in span}
    cont = (sum(1 for s in dict.fromkeys(needles) if s in span_set)
            / len(needles)) if needles else 0.0
    if not sigs:
        return cont
    sig_score = 0.0
    for name, args in sigs:
        best = 0.0
        for l in span:
            if name + '(' in l:
                toks = set(re.findall(r'\w+', l))
                best = max(best, len(args & toks) / len(args))
                if best >= 1.0:
                    break
        sig_score += best
    sig_score /= len(sigs)
    return sig_score + 0.5 * cont

def cache_valid(marker: str) -> bool:
    if not os.path.exists(marker):
        return False
    try:
        with open(marker, encoding='utf-8') as f:
            cached = json.load(f)
        return cached.get('anchor_algo_version') == ALGO_VERSION
    except (OSError, ValueError):
        return False

def run_cve(cve: str, mode: str, dims: dict, topk: int, force: bool) -> dict:
    task_dir = os.path.normpath(os.path.join(SRC_ROOT, '..', 'task', mode, cve))
    if not os.path.exists(os.path.join(task_dir, 'metadata.json')):
        raise RuntimeError(f'task artifacts missing for {cve}: run dataset.prepare first')
    with open(os.path.join(task_dir, 'metadata.json')) as f:
        meta = json.load(f)
    _, repo = _get_repos(mode)

    out_dir = os.path.join(task_dir, 'anchor')
    marker = os.path.join(out_dir, 'anchors.json')
    if os.path.exists(marker) and not force and cache_valid(marker):
        logger.info(f'[{cve}] anchor already exists, skip (use --force to rerun)')
        return json.load(open(marker))
    if os.path.exists(marker):
        logger.info(f'[{cve}] anchor cache stale (algorithm version changed), '
                    f'recomputing')
    os.makedirs(out_dir, exist_ok=True)

    units = extract_units(task_dir)
    logger.info(f'[{cve}] locating anchors for {len(units)} unit(s), dims={sorted(dims)}')

    results = []
    for unit in units:
        r = locate_unit(task_dir, meta, unit, repo, meta['commits']['pc'],
                        dims, topk)
        results.append(r)
    _containment_merge(results, units, meta, repo, task_dir)
    for unit, r in zip(units, results):
        safe_id = re.sub(r'[^\w.-]', '__', unit['unit_id'])
        if r.get('_chosen_source'):
            with open(os.path.join(out_dir, f'{safe_id}.ctx.c'), 'w') as f:
                f.write(r.pop('_chosen_source'))
        else:
            r.pop('_chosen_source', None)
        with open(os.path.join(out_dir, f'{safe_id}.json'), 'w') as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        logger.info(f'  [{unit["unit_id"]}] {r["status"]} conf={r["confidence"]:.3f} '
                    f'chosen={r["chosen"]["name"] if r["chosen"] else "-"}')

    summary = {
        'cve': cve, 'mode': mode, 'dims': sorted(dims), 'topk': topk,
        'anchor_algo_version': ALGO_VERSION,
        'units': results,
        'summary': {
            'units_total': len(results),
            'status_dist': dict(Counter(r['status'] for r in results)),
        },
    }
    with open(marker, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f'[{cve}] anchor done: {summary["summary"]["status_dist"]}')
    return summary

# def main(argv: list[str] | None = None) -> int:
#     p = argparse.ArgumentParser(prog='module.locator.anchor',
#                                 description='Locate PC anchors for patch units (deterministic).')
#     p.add_argument('--cve', required=True, help='comma-separated CVE ids')
#     p.add_argument('--mode', default='backport', choices=['backport', 'upstream-to-downstream'])
#     p.add_argument('--topk', type=int, default=TOP_K, help='candidates kept per unit')
#     p.add_argument('--force', action='store_true', help='rerun even if anchor/anchors.json exists')
#     args = p.parse_args(argv)

#     for cve in [c.strip() for c in args.cve.split(',') if c.strip()]:
#         run_cve(cve, args.mode, DIM_WEIGHTS, args.topk, args.force)
#     return 0

