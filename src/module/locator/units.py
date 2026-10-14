import difflib
import hashlib
import json
import os
import re
import subprocess
import threading
from collections import OrderedDict

from dataset.prepare import _get_repos
from utils.logger import get_logger
from utils.parser.diff_parser import Diff

logger = get_logger(__name__)

PA_WINDOW_PAD = 60
_HUNK_HEAD_RE = re.compile(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?')
_ABSORB_RATIO = 0.6

_RESOLVE_CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), 'tmp', 'resolve-cache')
_TREE_CACHE: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
_TREE_CACHE_MAX = 4
_SHOW_CACHE: OrderedDict[tuple[str, str, str], str | None] = OrderedDict()
_SHOW_CACHE_MAX = 256
MAX_RESOLVE_CANDIDATES = 24
_CACHE_LOCK = threading.Lock()

def _cache_put(cache: OrderedDict, key, value, cap: int) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > cap:
            cache.popitem(last=False)

def _git_show(repo: str, commit: str, path: str) -> str | None:
    key = (repo, commit, path)
    if key in _SHOW_CACHE:
        return _SHOW_CACHE[key]
    try:
        r = subprocess.run(['git', '-C', repo, 'show', f'{commit}:{path}'],
                           capture_output=True, text=True, errors='replace',
                           timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        logger.warning(f'  git show failed for {path}@{commit[:12]}')
        return None
    _cache_put(_SHOW_CACHE, key, r.stdout if r.returncode == 0 else None,
               _SHOW_CACHE_MAX)
    return _SHOW_CACHE[key]

def _tree_files(repo: str, commit: str) -> list[str]:
    key = (repo, commit)
    if key in _TREE_CACHE:
        return _TREE_CACHE[key]
    try:
        r = subprocess.run(['git', '-C', repo, 'ls-tree', '-r',
                            '--name-only', commit],
                           capture_output=True, text=True, timeout=120)
        files = r.stdout.splitlines() if r.returncode == 0 else []
        if r.returncode != 0:
            logger.warning(f'  ls-tree failed for {commit[:12]}: '
                           f'basename search degraded')
    except (OSError, subprocess.TimeoutExpired):
        logger.warning(f'  ls-tree failed for {commit[:12]}: '
                       f'basename search degraded')
        files = []
    _cache_put(_TREE_CACHE, key, files, _TREE_CACHE_MAX)
    return files

def _disk_cache_path(repo: str, commit: str, name: str) -> str:
    slug = hashlib.sha1(repo.encode()).hexdigest()[:12]
    return os.path.join(_RESOLVE_CACHE_ROOT, slug, commit, name)

def _read_cache_lines(path: str) -> list[str] | None:
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().splitlines()
    except OSError:
        return None

def _write_cache_lines(path: str, lines: list[str]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
    except OSError:
        pass

def _grep_hits(repo: str, commit: str,
               needles: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    todo: list[str] = []
    for nd in dict.fromkeys(needles):
        cached = _read_cache_lines(_disk_cache_path(repo, commit,
                                                    f'grep-{nd}.lst'))
        if cached is None:
            todo.append(nd)
        else:
            out[nd] = cached
    if not todo:
        return out
    cmd = (['git', '-C', repo, 'grep', '-o', '-w']
           + [a for nd in todo for a in ('-e', nd)] + [commit, '--', '*.c'])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        logger.warning(f'  git grep degraded at {commit[:12]} '
                       f'({len(todo)} needle(s))')
        return out
    hits: dict[str, set[str]] = {nd: set() for nd in todo}
    if r.returncode == 0:
        for ln in r.stdout.splitlines():
            parts = ln.split(':', 2)
            if len(parts) == 3 and parts[2] in hits and parts[1].endswith('.c'):
                hits[parts[2]].add(parts[1])
    for nd in todo:
        files = sorted(hits[nd])
        out[nd] = files
        _write_cache_lines(_disk_cache_path(repo, commit, f'grep-{nd}.lst'),
                           files)
    return out

def _materialize_pc_snapshot(repo: str, pc_commit: str, task_dir: str,
                             rel: str) -> None:
    dst = snapshot_path(task_dir, 'pc', pc_commit, rel)
    if os.path.exists(dst):
        return
    src = _git_show(repo, pc_commit, rel)
    if src is None:
        return
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(src)

def _block_affinity(hay_set: set[str], block: list[str]) -> float:
    core = {l.strip() for l in block if len(l.strip()) >= 8}
    if not core:
        return 0.0
    return len(core & hay_set) / len(core)

def extract_raw_hunks(diff_path: str) -> dict[str, dict[tuple, str]]:
    raw: dict[str, list[dict]] = {}
    cur_file, cur = None, None
    with open(diff_path, encoding='utf-8', errors='replace') as f:
        content = f.read()
    for line in content.splitlines():
        if line.startswith(('diff --git', '--- ', '+++ ', 'index ', 'new file',
                            'deleted file', 'old mode', 'new mode')):
            if cur is not None and cur_file:
                raw[cur_file].append(cur)
            cur = None
            if line.startswith('+++ '):
                p = line[4:].strip()
                cur_file = p[2:] if p.startswith('b/') else p
                raw.setdefault(cur_file, [])
            continue
        if line.startswith('@@'):
            m = _HUNK_HEAD_RE.match(line)
            cur = {'old_start': int(m.group(1)), 'new_start': int(m.group(2)),
                   'lines': [line]}
            raw.setdefault(cur_file, []).append(cur)
            continue
        if cur is not None and cur_file:
            cur['lines'].append(line)
    if cur is not None and cur_file:
        raw[cur_file].append(cur)

    out: dict[str, dict[tuple, str]] = {}
    for f, hunks in raw.items():
        out[f] = {}
        for h in hunks:
            old, new = h['old_start'], h['new_start']
            del_first = add_first = None
            for ln in h['lines'][1:]:
                if ln.startswith('\\'):
                    continue
                if ln.startswith('-'):
                    if del_first is None:
                        del_first = old
                    old += 1
                elif ln.startswith('+'):
                    if add_first is None:
                        add_first = new
                    new += 1
                else:
                    old += 1
                    new += 1
            out[f][(del_first, add_first)] = '\n'.join(h['lines'])
    return out

def _map_content_ok(unit: dict, mapped_src: str) -> bool:
    blocks = _pa_context_blocks(unit.get('raw_diff'))
    if not blocks:
        return True
    hay = [l.strip() for l in mapped_src.splitlines() if l.strip()]
    for b in blocks:
        needle = [l.strip() for l in b if l.strip()]
        if needle and _contains_block(hay, needle):
            return True
        if _block_affinity(set(hay), b) >= _ABSORB_RATIO:
            return True
    return False

def extract_units(task_dir: str) -> list[dict]:
    diff_path = os.path.join(task_dir, 'pa_pb.diff')
    chunks = Diff(diff_path).diff_chunks
    raw_map = extract_raw_hunks(diff_path)
    units = []
    for file, funcs in chunks.items():
        for func, data in funcs.items():
            hunks = [h for h in data.get('hunks', []) if h['delete_lines'] or h['add_lines']]
            if not hunks:
                continue
            raw_parts = []
            for h in hunks:
                key = (h['delete_lineno'][0] if h['delete_lineno'] else None,
                       h['add_lineno'][0] if h['add_lineno'] else None)
                raw_text = raw_map.get(file, {}).get(key)
                if raw_text:
                    raw_parts.append(raw_text)
                    head = _HUNK_HEAD_RE.match(raw_text.splitlines()[0])
                    if head:
                        h['old_start'] = int(head.group(1))
            units.append({'unit_id': f'{file}::{func}', 'file': file, 'func': func,
                          'hunks': hunks,
                          'raw_diff': '\n'.join(raw_parts) if raw_parts else None})
    try:
        with open(os.path.join(task_dir, 'metadata.json'), encoding='utf-8') as f:
            meta = json.load(f)
        pc_commit = meta['commits']['pc']
        _, repo = _get_repos(meta.get('mode') or 'backport')
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(f'  metadata unavailable ({exc}); keeping units unfiltered')
        logger.info(f'  pa_pb.diff -> {len(units)} unit(s)')
        return units
    keep: list[dict] = []
    pending: list[dict] = []
    same_path: list[dict] = []
    for u in units:
        if os.path.exists(snapshot_path(task_dir, 'pc', pc_commit, u['file'])) \
                or _git_show(repo, pc_commit, u['file']) is not None:
            if not os.path.exists(snapshot_path(task_dir, 'pc', pc_commit,
                                                u['file'])):
                _materialize_pc_snapshot(repo, pc_commit, task_dir, u['file'])
            u['pc_file'] = u['file']
            keep.append(u)
            same_path.append(u)
            continue
        pending.append(u)
    if keep:
        needles: list[str] = []
        for u in keep + pending:
            if re.fullmatch(r'[A-Za-z_]\w*', u.get('func') or ''):
                needles.append(u['func'])
            needles += [t for t in re.findall(r'[A-Za-z_]\w*', u['func'])
                        if len(t) >= 5 and t not in ('struct', 'union', 'enum',
                                                     'static', 'inline', 'const')]
        grep_map = _grep_hits(repo, pc_commit, needles)
        for u in same_path:
            others = [p for p in grep_map.get(u['func'], [])
                      if p != u['file']][:3]
            if others:
                u['pc_multi_copy'] = others
                logger.info(f'  [{u["unit_id"]}] function also defined at '
                            f'{", ".join(others)} — multi-copy ambiguity')
    if pending:
        pc_map: dict[str, str] = {}
        try:
            pc_map = meta.get('paths', {}).get('pc_map') or {}
        except AttributeError:
            pc_map = {}
        needles = []
        for u in pending:
            if re.fullmatch(r'[A-Za-z_]\w*', u.get('func') or ''):
                needles.append(u['func'])
            needles += [t for t in re.findall(r'[A-Za-z_]\w*', u['func'])
                        if len(t) >= 5 and t not in ('struct', 'union', 'enum',
                                                     'static', 'inline', 'const')]
        grep_map = _grep_hits(repo, pc_commit, needles)
        for u in pending:
            mapped = pc_map.get(u['file'])
            msrc = _git_show(repo, pc_commit, mapped) if mapped else None
            if mapped is not None and msrc is not None \
                    and _map_content_ok(u, msrc):
                _materialize_pc_snapshot(repo, pc_commit, task_dir, mapped)
                u['pc_file'] = mapped
                u['pc_file_src'] = 'prepare_map'
                logger.info(f'  [{u["unit_id"]}] file moved since PC '
                            f'(prepare_map): {u["file"]} -> {mapped}')
                keep.append(u)
                continue
            pc_file, src_level = _resolve_pc_file(u['file'], u, pc_commit,
                                                  repo, grep_map)
            if pc_file is None and mapped is not None and msrc is not None:
                pc_file, src_level = mapped, 'prepare_map_last_resort'
                logger.info(f'  [{u["unit_id"]}] content gates found nothing '
                            f'better; keeping prepare mapping {mapped}')
            if pc_file is None:
                logger.warning(f'  unit dropped, target file absent at PC: '
                               f'{u["unit_id"]}')
                continue
            _materialize_pc_snapshot(repo, pc_commit, task_dir, pc_file)
            u['pc_file'] = pc_file
            u['pc_file_src'] = src_level
            logger.info(f'  [{u["unit_id"]}] file moved since PC '
                        f'({src_level}): {u["file"]} -> {pc_file}')
            keep.append(u)
    units = keep
    logger.info(f'  pa_pb.diff -> {len(units)} unit(s)')
    return units

def _pa_context_blocks(raw_diff: str | None) -> list[list[str]]:
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for ln in (raw_diff or '').splitlines():
        if ln.startswith('@@'):
            if cur:
                blocks.append(cur)
            cur = []
        elif cur is None or ln.startswith(('diff --git', '--- ', '+++ ', 'index ')):
            continue
        elif ln.startswith('\\'):
            continue
        elif not ln.startswith('+'):
            cur.append((ln[1:] if ln.startswith((' ', '-')) else ln).rstrip())
    if cur:
        blocks.append(cur)
    return blocks

def _contains_block(hay: list[str], block: list[str]) -> bool:
    if not block or len(block) > len(hay):
        return False
    n = len(block)
    return any(hay[i:i + n] == block for i in range(len(hay) - n + 1))

def substantive_lines(blocks: list[list[str]]) -> list[str]:
    out = []
    for b in blocks:
        for l in b:
            s = l.strip()
            if len(s) >= 8:
                out.append(s)
    return out

def span_containment(hay: list[str], lo: int, hi: int,
                     needles: list[str]) -> tuple[int, int]:
    uniq = list(dict.fromkeys(needles))
    if hi < lo or not uniq:
        return 0, len(uniq)
    span = {l.strip() for l in hay[lo - 1:hi]}
    return sum(1 for s in uniq if s in span), len(uniq)

SIM_MIN = 0.75
SIM_MARGIN = 0.05

_WORD_RE = re.compile(r'\w+')

def fuzzy_pa_line_map(hay: list[str], lo: int, hi: int, needles: list[str]
                      ) -> list[tuple[str, int | None, str]]:
    uniq = list(dict.fromkeys(needles))
    if hi < lo or not uniq:
        return [(x, None, '') for x in uniq]
    lo_i, hi_i = max(0, lo - 1), min(len(hay), hi)
    exact: dict[str, int] = {}
    for i in range(lo_i, hi_i):
        exact.setdefault(hay[i].strip(), i + 1)
    span_core = [(i + 1, hay[i].strip()) for i in range(lo_i, hi_i)
                 if len(hay[i].strip()) >= 8]
    out = []
    for x in uniq:
        ln = exact.get(x)
        mode = 'exact' if ln else ''
        if ln is None and span_core:
            best_n, best_r, runner = None, 0.0, 0.0
            xt = _WORD_RE.findall(x)
            for n, s in span_core:
                r = difflib.SequenceMatcher(None, xt, _WORD_RE.findall(s)).ratio()
                if r > best_r:
                    best_n, best_r, runner = n, r, best_r
                elif r > runner:
                    runner = r
            if best_r >= SIM_MIN and best_r - runner >= SIM_MARGIN:
                ln, mode = best_n, 'similar'
        out.append((x, ln, mode))
    return out

def _resolve_pc_file(pa_file: str, unit: dict, pc_commit: str, repo: str,
                     grep_map: dict[str, list[str]]
                     ) -> tuple[str | None, str]:
    if _git_show(repo, pc_commit, pa_file) is not None:
        return pa_file, 'same_path'
    blocks = _pa_context_blocks(unit.get('raw_diff'))
    base = os.path.basename(pa_file)
    pa_dir = os.path.dirname(pa_file)
    tokens = [t for t in re.findall(r'[A-Za-z_]\w*', unit['func'])
              if len(t) >= 5 and t not in ('struct', 'union', 'enum',
                                           'static', 'inline', 'const')]

    def _dir_rank(p: str) -> float:
        return difflib.SequenceMatcher(None, os.path.dirname(p),
                                       pa_dir).ratio()

    def _content_pick(cands: list[str], prefix: str
                      ) -> tuple[str | None, str]:
        hay: dict[str, list[str]] = {}
        for p in sorted(cands, key=_dir_rank,
                        reverse=True)[:MAX_RESOLVE_CANDIDATES]:
            src = _git_show(repo, pc_commit, p)
            if src:
                hay[p] = [l.rstrip() for l in src.splitlines()]
        exact = [p for p in cands if p in hay
                 and any(_contains_block(hay[p], b) for b in blocks)]
        if len(exact) == 1:
            return exact[0], prefix + 'content_exact'
        fuzzy = [p for p in cands if p in hay
                 and any(_block_affinity(set(hay[p]), b) >= _ABSORB_RATIO
                         for b in blocks)]
        if len(fuzzy) == 1:
            return fuzzy[0], prefix + 'content_fuzzy'
        return None, ''

    cands = [p for p in _tree_files(repo, pc_commit)
             if os.path.basename(p) == base]
    if cands:
        got, level = _content_pick(cands, '')
        if got:
            return got, level
    needles = []
    if re.fullmatch(r'[A-Za-z_]\w*', unit.get('func') or ''):
        needles.append(unit['func'])
    needles += [t for t in tokens if t not in needles][:2]
    for nd in needles:
        ghits = [p for p in grep_map.get(nd, []) if p != pa_file]
        if not ghits:
            continue
        if len(ghits) == 1:
            src = _git_show(repo, pc_commit, ghits[0])
            if src and any(re.search(r'\b' + re.escape(t) + r'\b', src)
                           for t in tokens):
                return ghits[0], 'grep_unique'
            continue
        got, level = _content_pick(ghits, 'grep_')
        if got:
            return got, level
    return None, ''

def render_unit_diff(unit: dict) -> str:
    if unit.get('raw_diff'):
        return unit['raw_diff']
    blocks = []
    for h in unit['hunks']:
        lines = [f'@@ {h["signature"] or unit["func"]} @@']
        lines += h['context_before']
        lines += [f'-{text.rstrip(chr(10))}' for _, text, _ in h['delete_lines']]
        lines += [f'+{text.rstrip(chr(10))}' for _, text, _ in h['add_lines']]
        lines += h['context_after']
        blocks.append('\n'.join(lines))
    return '\n'.join(blocks)

def pa_position_seeds(unit: dict) -> list[int]:
    linenos = [n for h in unit['hunks'] for n in h['delete_lineno']]
    if linenos:
        return linenos
    seeds = []
    for h in unit['hunks']:
        if h.get('delete_lineno') or 'old_start' not in h:
            continue
        seeds.append(h['old_start'] + len(h.get('context_before', [])))
    return seeds

def snapshot_path(task_dir: str, role: str, commit: str, rel_path: str) -> str:
    name = f'{role}_{commit[:12]}_{rel_path.replace("/", "_")}'
    return os.path.join(task_dir, name)

def pa_source_window(task_dir: str, meta: dict, unit: dict) -> str:
    path = snapshot_path(task_dir, 'pa', meta['commits']['pa'], unit['file'])
    if not os.path.exists(path):
        logger.warning(f'  pa snapshot missing for {unit["file"]}, window skipped')
        return '(PA source not available)'
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    linenos = pa_position_seeds(unit)
    if not linenos:
        return ''.join(lines)
    lo = max(1, min(linenos) - PA_WINDOW_PAD)
    hi = min(len(lines), max(linenos) + PA_WINDOW_PAD)
    window = ''.join(f'{i + 1:>6} {lines[i]}' for i in range(lo - 1, hi))
    return f'(lines {lo}-{hi} of {len(lines)})\n{window}'
