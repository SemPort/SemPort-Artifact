# import argparse
import difflib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger

logger = get_logger(__name__)

SRC_ROOT = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

DATASETS = {
    'backport': os.path.join(PROJECT_ROOT, 'dataset', 'backport'),
    'upstream-to-downstream': os.path.join(PROJECT_ROOT, 'dataset', 'upstream-to-downstream'),
}

REPOS_BACKPORT = {
    'source': os.path.join(PROJECT_ROOT, 'repos', 'backport', 'linux'),
    'target': os.path.join(PROJECT_ROOT, 'repos', 'backport', 'linux-stable'),
}
REPOS_UP2DOWN = {
    'source': os.path.join(PROJECT_ROOT, 'repos', 'backport', 'linux'),
    'target': os.path.join(PROJECT_ROOT, 'repos', 'upstream-to-downstream', 'android-common'),
}
REPOS = {
    'backport': REPOS_BACKPORT,
    'upstream-to-downstream': REPOS_UP2DOWN,
}

# TMP_ROOT = os.path.join(PROJECT_ROOT, 'tmp')
TASK_ROOT = os.path.join(PROJECT_ROOT, 'task')

_SHORT = 12
PREPARE_VERSION = 3

def _git(repo: str, *args) -> str:
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'git {args} failed in {repo}: {r.stderr[:500]}')
    return r.stdout.rstrip('\n')

def _get_repos(mode: str) -> tuple[str, str]:
    if mode not in REPOS:
        raise ValueError(f'unknown mode {mode!r}; expected one of {sorted(REPOS)}')
    repos = REPOS[mode]
    if not repos['source'] or not repos['target']:
        raise RuntimeError(f"repos for mode {mode!r} not configured yet "
                           f"(source={repos['source']!r}, target={repos['target']!r})")
    return repos['source'], repos['target']

def path2filename(path: str) -> str:
    return path.replace('/', '_')

def _find_metadata(cve_id: str, mode: str) -> str:
    cve_dir = os.path.join(DATASETS[mode], cve_id)
    if not os.path.isdir(cve_dir):
        raise RuntimeError(f'dataset entry missing: {cve_dir}')
    candidates = [os.path.join(cve_dir, 'metadata.json')]
    sub_dir = sorted(d for d in os.listdir(cve_dir) if os.path.isdir(os.path.join(cve_dir, d)))
    if sub_dir:
        candidates.append(os.path.join(cve_dir, sub_dir[0], 'metadata.json'))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise RuntimeError(f'metadata.json missing for {cve_id}')

def _pb_pe_hashes(meta: dict, mode: str, meta_path: str) -> tuple[str, str]:
    if mode == 'backport':
        pb_hash = (meta.get('master') or {}).get('hash') or meta.get('upstream_key')
        pe_hash = (meta.get('oldest_stable') or {}).get('hash')
    elif mode == 'upstream-to-downstream':
        pb_hash = (meta.get('linux') or {}).get('hash')
        pe_hash = (meta.get('android') or {}).get('hash') \
            or (meta.get('android') or {}).get('commit')
    else:
        pb_hash = pe_hash = None
    if not pb_hash or not pe_hash:
        raise RuntimeError(f'PB/PE hash fields missing in {meta_path}')
    return pb_hash, pe_hash

def resolve_commits(cve_id: str, mode: str) -> dict[str, str]:
    meta_path = _find_metadata(cve_id, mode)
    pb_hash, pe_hash = _pb_pe_hashes(json.load(open(meta_path)), mode, meta_path)
    source, target = _get_repos(mode)
    pb = _git(source, 'rev-parse', '--verify', pb_hash)
    pe = _git(target, 'rev-parse', '--verify', pe_hash)
    return {'pa': _git(source, 'rev-parse', pb + '^'), 'pb': pb,
            'pc': _git(target, 'rev-parse', pe + '^'), 'pe': pe}

def _git_ok(repo: str, *args) -> str | None:
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None

def _blob_lines(repo: str, commit: str, path: str) -> list[str] | None:
    out = subprocess.run(['git', '-C', repo, 'show', f'{commit}:{path}'],
                         capture_output=True, text=True)
    return out.stdout.splitlines() if out.returncode == 0 else None

_SIM_THRESHOLD = 0.4

def _blob_match(source: str, pa: str, pa_path: str,
                target: str, pc: str, cand: str) -> bool:
    a = _blob_lines(source, pa, pa_path)
    b = _blob_lines(target, pc, cand)
    if not a or not b:
        return False
    norm = lambda ls: {l.strip() for l in ls if len(l.strip()) >= 6}
    sa, sb = norm(a), norm(b)
    if not sa:
        return False
    if len(sb) <= 4000 \
            and difflib.SequenceMatcher(None, a[:3000], b[:3000]).ratio() \
            >= _SIM_THRESHOLD:
        return True
    return len(sa & sb) / len(sa) >= 0.5

def _creation_commit(source: str, pa: str, pa_path: str) -> str | None:
    adds = _git_ok(source, 'log', '--reverse', '--diff-filter=A',
                   '--format=%H', pa, '--', pa_path)
    return adds.splitlines()[0] if adds else None

def _creation_pairing(source: str, pa: str, pa_path: str) -> list[str]:
    commit = _creation_commit(source, pa, pa_path)
    if not commit:
        return []
    out = _git_ok(source, 'diff-tree', '-C', '--name-status', '-r',
                  '--no-commit-id', commit)
    srcs: list[str] = []
    for line in (out or '').splitlines():
        f = line.split('\t')
        if len(f) == 3 and f[0][:1] in 'RC' and f[2] == pa_path:
            if f[1] not in srcs:
                srcs.append(f[1])
    return srcs

def _creation_siblings(source: str, pa: str, pa_path: str) -> list[str]:
    commit = _creation_commit(source, pa, pa_path)
    if not commit:
        return []
    out = _git_ok(source, 'diff-tree', '--name-status', '-r',
                  '--no-commit-id', commit)
    parent = pa_path.rsplit('/', 1)[0] if '/' in pa_path else ''
    sib: list[str] = []
    for line in (out or '').splitlines():
        f = line.split('\t')
        if len(f) == 2 and f[0] == 'M' and f[1].rsplit('/', 1)[0] == parent \
                and f[1] != pa_path and f[1] not in sib:
            sib.append(f[1])
    return sib

def _rename_history(source: str, pa: str, pa_path: str,
                    limit: int = 60) -> list[str]:
    out = _git_ok(source, 'log', '--follow', f'--max-count={limit}',
                  '--name-status', '--format=%x01', pa, '--', pa_path)
    older: list[str] = []
    for line in (out or '').splitlines():
        if '\t' not in line:
            continue
        fields = line.split('\t')
        if fields[0][0] in 'RC' and len(fields) == 3:
            old = fields[1]
            if old != pa_path and old not in older:
                older.append(old)
    return older

def _pc_path_fallback(source: str, pa: str, target: str, pc: str,
                      pa_paths: list[str], exact_hits: list[str]) -> tuple[dict[str, str], list[str]]:
    missed = [p for p in pa_paths if p not in set(exact_hits)]
    if not missed:
        return {}, []
    by_sha: dict[str, str] = {}
    by_base: dict[str, list[str]] = {}
    for line in _git(target, 'ls-tree', '-r', pc).splitlines():
        meta, _, path = line.partition('\t')
        fields = meta.split()
        if len(fields) >= 3 and path:
            by_sha.setdefault(fields[2], path)
            by_base.setdefault(path.rsplit('/', 1)[-1], []).append(path)

    def _affinity(pa_path: str, cand: str) -> tuple:
        pa_parts, c_parts = pa_path.split('/'), cand.split('/')
        same_parent = int(len(pa_parts) > 1 and len(c_parts) > 1
                          and pa_parts[-2] == c_parts[-2])
        lcp = 0
        for a, b in zip(pa_parts, c_parts):
            if a != b:
                break
            lcp += 1
        return (same_parent, lcp, -len(c_parts))

    def _unique(cands: list[str], pa_path: str) -> list[str]:
        if len(cands) <= 1:
            return cands
        ordered = sorted(cands, key=lambda c: (*_affinity(pa_path, c), c))
        return ordered[:1] if _affinity(pa_path, ordered[0]) != _affinity(pa_path, ordered[1]) else []

    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for pa_path in missed:
        cands: list[str] = []
        how = ''
        sha = _git_ok(source, 'rev-parse', f'{pa}:{pa_path}')
        if sha and sha in by_sha:
            cands, how = [by_sha[sha]], 'blob-identical'
        if not cands:
            for old in _rename_history(source, pa, pa_path):
                hit = _git_ok(target, 'ls-tree', '--name-only', pc, '--', old)
                if hit:
                    cands, how = [hit.splitlines()[0]], 'rename-history'
                    break
        if not cands:
            for old in _creation_pairing(source, pa, pa_path):
                hit = _git_ok(target, 'ls-tree', '--name-only', pc, '--', old)
                if hit:
                    cands, how = [hit.splitlines()[0]], 'creation-pairing'
                    break
        if not cands:
            for sib in _creation_siblings(source, pa, pa_path):
                hit = _git_ok(target, 'ls-tree', '--name-only', pc, '--', sib)
                if hit and _blob_match(source, pa, pa_path, target, pc,
                                       hit.splitlines()[0]):
                    cands, how = [hit.splitlines()[0]], 'creation-sibling'
                    break
        if not cands:
            base = pa_path.rsplit('/', 1)[-1]
            named = list(by_base.get(base, []))
            if not named:
                named = [p for b, ps in by_base.items()
                         if b.endswith(base) and b != base for p in ps]
            u = _unique(named, pa_path)
            if u and _blob_match(source, pa, pa_path, target, pc, u[0]):
                cands, how = u, 'name+content'
        if not cands:
            unresolved.append(pa_path)
            logger.info(f'  pc path unresolved: {pa_path}')
            continue
        mapping[pa_path] = cands[0]
        logger.info(f'  pc path fallback ({how}): {pa_path} -> {cands[0]}')
    return mapping, unresolved

def resolve_cve(cve_id: str, mode: str = 'backport') -> dict:
    if mode not in DATASETS:
        raise ValueError(f'unknown mode {mode!r}, expected one of {sorted(DATASETS)}')
    pre = _resolve_cache_get(cve_id)
    if pre is not None:
        return pre
    return _resolve_cve_slow(cve_id, mode)

_RESOLVE_CACHE: dict[str, dict] | None = None

def _resolve_cache_get(cve_id: str) -> dict | None:
    global _RESOLVE_CACHE
    path = os.environ.get('SEMPORT_RESOLVE_CACHE')
    if not path:
        return None
    if _RESOLVE_CACHE is None:
        try:
            with open(path) as f:
                _RESOLVE_CACHE = json.load(f)
        except (OSError, ValueError):
            _RESOLVE_CACHE = {}
    return _RESOLVE_CACHE.get(cve_id)

def _resolve_cve_slow(cve_id: str, mode: str = 'backport') -> dict:
    if mode not in DATASETS:
        raise ValueError(f'unknown mode {mode!r}, expected one of {sorted(DATASETS)}')
    meta_path = _find_metadata(cve_id, mode)
    pb_hash, pe_hash = _pb_pe_hashes(json.load(open(meta_path)), mode, meta_path)

    source, target = _get_repos(mode)
    pb = _git(source, 'rev-parse', '--verify', pb_hash)
    pe = _git(target, 'rev-parse', '--verify', pe_hash)
    pa = _git(source, 'rev-parse', pb + '^')
    pc = _git(target, 'rev-parse', pe + '^')

    pa_paths = _git(source, 'diff', '--name-only', pa, pb).splitlines()
    pc_paths = _git(target, 'ls-tree', '--name-only', pc, '--',
                    *pa_paths).splitlines() if pa_paths else []
    pc_map = {p: p for p in pc_paths}
    if pa_paths and len(pc_paths) < len(pa_paths):
        fallback_map, unresolved = _pc_path_fallback(source, pa, target, pc,
                                                     pa_paths, pc_paths)
        pc_paths = pc_paths + list(fallback_map.values())
        pc_map.update(fallback_map)
    else:
        unresolved = []

    return {
        'cve': cve_id,
        'mode': mode,
        'metadata': meta_path,
        'pa_commit': pa,
        'pb_commit': pb,
        'pc_commit': pc,
        'pe_commit': pe,
        'pa_paths': pa_paths,
        'pc_paths': pc_paths,
        'pc_map': pc_map,
        'pc_unresolved': unresolved,
    }

def _task_commits(task_dir: str) -> dict[str, str] | None:
    try:
        with open(os.path.join(task_dir, 'metadata.json')) as f:
            meta = json.load(f)
        if meta.get('prepare_version') != PREPARE_VERSION:
            return None
        return meta.get('commits')
    except (OSError, ValueError):
        return None

def _save_snapshots(repo: str, commit: str, role: str, paths: list[str], task_dir: str):
    short = commit[:_SHORT]
    for path in paths:
        r = subprocess.run(['git', '-C', repo, 'show', f'{commit}:{path}'],
                           capture_output=True)
        if r.returncode != 0:
            logger.warning(f'  {role}:{path} missing at {short}, skipped')
            continue
        dst = os.path.join(task_dir, f'{role}_{short}_{path2filename(path)}')
        with open(dst, 'wb') as f:
            f.write(r.stdout)

def _invalidate_locator_caches(task_dir: str, info: dict) -> None:
    old_meta_path = os.path.join(task_dir, 'metadata.json')
    try:
        with open(old_meta_path, encoding='utf-8') as f:
            old_pc = json.load(f).get('paths', {}).get('pc') or []
    except (OSError, ValueError):
        return
    if old_pc == info['pc_paths']:
        return
    for sub, name in (('intent', 'intents.json'), ('anchor', 'anchors.json')):
        path = os.path.join(task_dir, sub, name)
        if os.path.exists(path):
            os.remove(path)
    logger.info(f'  pc path mapping changed '
                f'({len(old_pc)} -> {len(info["pc_paths"])} files); '
                'invalidated intent/anchor caches')

def prepare_task(cve_id: str, mode: str, info: dict | None = None) -> str:
    if info is None:
        info = resolve_cve(cve_id, mode)
    source, target = _get_repos(mode)
    task_dir = os.path.join(TASK_ROOT, mode, cve_id)
    os.makedirs(task_dir, exist_ok=True)
    _invalidate_locator_caches(task_dir, info)

    pa_pb = _git(source, 'diff', info['pa_commit'], info['pb_commit'])
    with open(os.path.join(task_dir, 'pa_pb.diff'), 'w') as f:
        f.write(pa_pb + '\n')
    pc_pe = _git(target, 'diff', info['pc_commit'], info['pe_commit'])
    with open(os.path.join(task_dir, 'pc_pe.diff'), 'w') as f:
        f.write(pc_pe + '\n')

    _save_snapshots(source, info['pa_commit'], 'pa', info['pa_paths'], task_dir)
    _save_snapshots(source, info['pb_commit'], 'pb', info['pa_paths'], task_dir)
    _save_snapshots(target, info['pc_commit'], 'pc', info['pc_paths'], task_dir)
    _save_snapshots(target, info['pe_commit'], 'pe', info['pc_paths'], task_dir)

    pb_message = subprocess.run(
        ['git', '-C', source, 'log', '-1', '--format=%B', info['pb_commit']],
        capture_output=True, text=True).stdout.strip()

    meta = {
        'cve': cve_id,
        'mode': mode,
        'prepare_version': PREPARE_VERSION,
        'commits': {r: info[f'{r}_commit'] for r in ('pa', 'pb', 'pc', 'pe')},
        'paths': {'pa': info['pa_paths'], 'pc': info['pc_paths'],
                  'pc_map': info.get('pc_map', {}),
                  'pc_unresolved': info.get('pc_unresolved', [])},
        'pb_message': pb_message,
    }
    with open(os.path.join(task_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f'  task artifacts -> {task_dir} '
                f'({len(info["pa_paths"])} pa-side, {len(info["pc_paths"])} pc-side files)')
    return task_dir

class StageError(RuntimeError):

    def __init__(self, stage: str, msg: str):
        super().__init__(f'[{stage}] {msg}')
        self.stage = stage

def run_cve_queue(cves: list[str], mode: str, threads: int,
                  run_hook=None, reprepare: bool = False) -> list[dict]:
    if not cves:
        return []
    env_count = min(threads, len(cves))
    free_envs: queue.Queue = queue.Queue()
    for x in range(1, env_count + 1):
        free_envs.put(f'thread-{x}')
    statuses: dict[str, dict] = {}
    lock = threading.Lock()

    def _record(status: dict):
        with lock:
            statuses[status['cve']] = status

    def _run_task(cve: str, env_token: str, prepare_s: float | None) -> None:
        env_name = env_token
        t0 = time.monotonic()
        try:
            result = run_hook(cve, mode) or {}
            test_s = time.monotonic() - t0
            logger.info(f'[{cve}] generate done on env {env_name} in {test_s:.1f}s')
            _record({'cve': cve, 'ok': True, 'stage': None, 'env': env_name,
                     'prepare_s': round(prepare_s, 1) if prepare_s else None,
                     'test_s': round(test_s, 1),
                     'error': None, 'result': result})
        except Exception as exc:
            logger.error(f'[{cve}] generate failed on env {env_name}: {exc}')
            _record({'cve': cve, 'ok': False,
                     'stage': getattr(exc, 'stage', 'generate'), 'env': env_name,
                     'prepare_s': round(prepare_s, 1) if prepare_s else None,
                     'test_s': None,
                     'error': str(exc), 'result': None})
        finally:
            free_envs.put(env_token)

    logger.info(f'queue: {len(cves)} CVE(s), {env_count} env copy(ies), '
                'serial prepare' + (', runs in parallel' if run_hook else ''))
    with ThreadPoolExecutor(max_workers=env_count) as test_pool:
        futures = []
        for cve in cves:
            env_token = free_envs.get()
            env_name = env_token
            t0 = time.monotonic()
            try:
                task_dir = os.path.join(TASK_ROOT, mode, cve)
                if not reprepare and _task_commits(task_dir) \
                        == resolve_commits(cve, mode):
                    prepare_s = time.monotonic() - t0
                    logger.info(f'[{cve}] task artifacts fresh, prepare '
                                f'skipped ({prepare_s:.1f}s)')
                else:
                    info = resolve_cve(cve, mode=mode)
                    prepare_task(cve, mode, info)
                    prepare_s = time.monotonic() - t0
                    logger.info(f'[{cve}] env {env_name} prepared in {prepare_s:.1f}s')
            except Exception as exc:
                prepare_s = time.monotonic() - t0
                logger.error(f'[{cve}] preparation failed on env {env_name}: {exc}')
                _record({'cve': cve, 'ok': False, 'stage': 'prepare',
                         'env': env_name, 'prepare_s': round(prepare_s, 1),
                         'test_s': None,
                         'error': str(exc), 'result': None})
                free_envs.put(env_token)
                continue
            if run_hook is None:
                _record({'cve': cve, 'ok': True, 'stage': None,
                         'env': env_name, 'prepare_s': round(prepare_s, 1),
                         'test_s': None,
                         'error': None, 'result': None})
                free_envs.put(env_token)
            else:
                futures.append(test_pool.submit(_run_task, cve, env_token, prepare_s))
        for future in futures:
            future.result()
    return [statuses[cve] for cve in cves]

# def main(argv: list[str] | None = None) -> int:
#     p = argparse.ArgumentParser(
#         prog='dataset.prepare',
#         description='Prepare per-thread repo environments and git-derived task artifacts.')
#     p.add_argument('--cve', default=None, help='comma-separated CVE ids')
#     p.add_argument('--mode', default='backport', choices=sorted(DATASETS),
#                    help='dataset mode (default: backport)')
#     p.add_argument('--threads', type=int, default=1,
#                    help='number of env copies under <repo-root>/tmp/<mode>/thread-x/ '
#                         '(min(N, #CVEs) created or reused); prepares always run '
#                         'serially in queue order (default: 1)')
#     p.add_argument('--force', action='store_true',
#                    help='rewrite task artifacts even when their commits match')
#     p.add_argument('--resolve-only', action='store_true',
#                    help='only resolve commits/paths; no env copies, no artifacts (serial)')
#     args = p.parse_args(argv)

#     if not args.cve:
#         p.error('--cve is required')

#     cves = [c.strip() for c in args.cve.split(',') if c.strip()]
#     if not cves:
#         p.error('--cve is required')
#     if args.resolve_only:
#         for cve in cves:
#             info = resolve_cve(cve, mode=args.mode)
#             logger.info(f'[{cve}] PA={info["pa_commit"][:_SHORT]} PB={info["pb_commit"][:_SHORT]} '
#                         f'PC={info["pc_commit"][:_SHORT]} PE={info["pe_commit"][:_SHORT]}')
#             logger.info(f'  pa_paths={info["pa_paths"]}')
#             logger.info(f'  pc_paths={info["pc_paths"]}')
#         return 0

#     statuses = run_cve_queue(cves, args.mode, args.threads,
#                              reprepare=args.force)
#     failed = [s['cve'] for s in statuses if not s['ok']]
#     if failed:
#         logger.error(f'{len(failed)}/{len(cves)} CVE(s) failed preparation: {", ".join(failed)}')
#         return 1
#     return 0

