import os
import re
import subprocess

from dataset.prepare import _get_repos
from module.locator.anchor import name_sim
from module.locator.units import snapshot_path
from utils.logger import get_logger
from utils.parser.ctags_parser import ctags_parse

logger = get_logger(__name__)

_CALL_RE = re.compile(r'([A-Za-z_]\w*)\s*\(')
_DEF_LINE_RE = re.compile(r'^[A-Za-z_][\w\s\*]*?([A-Za-z_]\w*)\s*\(')
_MACRO_RE = re.compile(r'\b([A-Z][A-Z0-9_]{2,})\b')
_TYPE_RE = re.compile(r'\b(?:struct|union|enum)\s+([A-Za-z_]\w*)')
_FIELD_RE = re.compile(r'(?:->|\.\s*)([A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)')
_WORD_RE = re.compile(r'[A-Za-z_]\w*')
_KEYWORDS = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'else',
             'do', 'goto', 'break', 'continue', 'case', 'default', 'defined',
             'NULL', 'EINVAL', 'ENOMEM', 'GFP_KERNEL', 'true', 'false'}

_SYM_CACHE: dict[tuple, str] = {}

def _grep_hit(repo: str, commit: str, patterns: list[str],
              pathspec: str) -> str | None:
    key = (repo, commit, pathspec, tuple(patterns))
    if key in _SYM_CACHE:
        return _SYM_CACHE[key] if _SYM_CACHE[key] != '' else ''
    try:
        r = subprocess.run(
            ['git', '-C', repo, 'grep', '-o', *patterns, commit, '--',
             pathspec],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        _SYM_CACHE[key] = None
        return None
    if r.returncode not in (0, 1):
        _SYM_CACHE[key] = None
        return None
    _SYM_CACHE[key] = r.stdout
    return r.stdout

def _needle_for(sym: str, kind: str) -> list[str]:
    if kind == 'field':
        return ['-F', '-e', f'->{sym}', '-e', f'.{sym}']
    return ['-w', '-e', sym]

def nearby_presence(symbols: list[dict], repo: str, commit: str,
                    anchored_file: str, anchored_text: str) -> dict[str, str]:
    status: dict[str, str] = {}
    dirp = os.path.dirname(anchored_file)
    todo = []
    for s in symbols:
        sym = s['sym']
        if s['kind'] == 'field':
            pat = re.compile(r'(?:->|\.\s*)' + re.escape(sym) + r'\b')
            if pat.search(anchored_text):
                status[sym] = 'present'
                continue
        elif re.search(r'\b' + re.escape(sym) + r'\b', anchored_text):
            status[sym] = 'present'
            continue
        todo.append(s)
    if not todo:
        return status
    for kind in ('plain', 'field'):
        group = [s for s in todo
                 if (s['kind'] == 'field') == (kind == 'field')]
        if not group:
            continue
        pathspec = f'{dirp}/' if dirp else '.'
        ok = True
        for s in group:
            got = _grep_hit(repo, commit, _needle_for(s['sym'], kind),
                            pathspec)
            if got is None:
                ok = False
            elif any(ln.rstrip().endswith(s['sym'])
                     for ln in got.splitlines()):
                status[s['sym']] = 'present'
        for s in group:
            status.setdefault(s['sym'], 'nearby_absent' if ok else 'unknown')
    return status

def confirm_absent(sym: str, kind: str, ctx: dict) -> bool:
    key = (ctx['meta']['commits']['pc'], sym, kind)
    if key in _SYM_CACHE:
        return _SYM_CACHE[key] == 'absent'
    meta = ctx['meta']
    _, repo = _get_repos(meta.get('mode') or 'backport')
    unit0 = ctx['units'][0] if ctx.get('units') else {}
    dirp = os.path.dirname(ctx.get('file') or unit0.get('file', ''))
    patterns = _needle_for(sym, kind)
    seen = False
    exhausted = True
    for pathspec in ([f'{dirp}/'] if dirp else ['.']) + ['*.c', '*.h']:
        got = _grep_hit(repo, meta['commits']['pc'], patterns, pathspec)
        if got is None:
            exhausted = False
            break
        if any(ln.rstrip().endswith(sym) for ln in got.splitlines()):
            seen = True
            break
    verdict = 'absent' if (exhausted and not seen) else 'unknown'
    _SYM_CACHE[key] = verdict
    return verdict == 'absent'

def extract_hunk_symbols(units: list[dict]) -> list[dict]:
    seen: dict[tuple[str, str], dict] = {}
    intro: set[str] = set()
    for u in units:
        for h in (u.get('hunks') or []):
            for t in h.get('add_lines', []):
                s = t[2].strip()
                if re.match(r'^(?:static\s+|const\s+|inline\s+)*[\w\*\s]+?\b'
                            r'[A-Za-z_]\w*\s*\(', s) \
                        and not re.match(r'^(?:if|for|while|switch|return|else)\b',
                                         s):
                    dm = _DEF_LINE_RE.match(s)
                    if dm and len(dm.group(1)) >= 3:
                        intro.add(dm.group(1))
                dm = re.match(r'#define\s+([A-Za-z_]\w*)', s)
                if dm:
                    intro.add(dm.group(1))
                dt = _TYPE_RE.search(s)
                if dt and re.match(r'^(?:struct|union|enum)\s', s) and '{' in s:
                    intro.add(dt.group(1))
        for h in (u.get('hunks') or []):
            for t in h.get('add_lines', []):
                line = t[2]
                for m in _CALL_RE.findall(line):
                    if m not in _KEYWORDS and len(m) >= 3:
                        seen.setdefault(('call', m), {'sym': m, 'kind': 'call'})
                for m in _MACRO_RE.findall(line):
                    if m not in _KEYWORDS and not m.startswith('CONFIG_'):
                        seen.setdefault(('macro', m),
                                        {'sym': m, 'kind': 'macro'})
                for m in _TYPE_RE.findall(line):
                    seen.setdefault(('type', m), {'sym': m, 'kind': 'type'})
                for m in _FIELD_RE.findall(line):
                    m = re.sub(r'\s+', '', m)
                    if '.' in m or len(m) >= 3:
                        seen.setdefault(('field', m), {'sym': m, 'kind': 'field'})
    for (kind, sym), entry in seen.items():
        if sym in intro:
            entry['kind'] = 'patch_introduced'
    return list(seen.values())

def pa_definitions(pa_path: str, task_dir: str) -> set[str]:
    try:
        tags = ctags_parse(pa_path, tag_dir=task_dir)
    except Exception:
        return set()
    out = set()
    for t in tags.values():
        name = t.get('name') or b''
        out.add(name.decode('utf-8', 'replace') if isinstance(name, bytes)
                else name)
    return out

def nearest_candidates(sym: str, pc_path: str, task_dir: str,
                       k: int = 3) -> list[str]:
    try:
        tags = ctags_parse(pc_path, tag_dir=task_dir)
    except Exception:
        return []
    names = []
    for t in tags.values():
        kind = t.get('kind') or b''
        kind = kind.decode('utf-8', 'replace') if isinstance(kind, bytes) else kind
        name = t.get('name') or b''
        name = name.decode('utf-8', 'replace') if isinstance(name, bytes) else name
        if kind in ('f', 'function') and name:
            names.append(name)
    scored = sorted(((name_sim(sym, n), n) for n in set(names)), reverse=True)
    return [n for sc, n in scored[:k] if sc > 0.5]

def for_group(units: list[dict], ctx: dict) -> list[dict]:
    cached = ctx.get('symbol_evidence')
    if cached is not None:
        return cached
    meta = ctx['meta']
    task_dir = ctx['task_dir']
    unit0 = units[0] if units else {}
    pc_path = snapshot_path(task_dir, 'pc', meta['commits']['pc'],
                            ctx.get('file') or unit0.get('file', ''))
    pa_path = snapshot_path(task_dir, 'pa', meta['commits']['pa'],
                            unit0.get('file', ''))
    if not (os.path.exists(pc_path) and os.path.exists(pa_path)):
        ctx['symbol_evidence'] = []
        return []
    with open(pc_path, encoding='utf-8', errors='replace') as f:
        pc_text = f.read()
    syms = extract_hunk_symbols(units)
    if not syms:
        ctx['symbol_evidence'] = []
        return []
    pa_defs = pa_definitions(pa_path, task_dir)
    words = {w for u in units for h in (u.get('hunks') or [])
             for t in h.get('add_lines', [])
             for w in _WORD_RE.findall(t[2]) if len(w) >= 5}
    for w in sorted(words & pa_defs):
        if not any(s['sym'] == w for s in syms):
            syms.append({'sym': w, 'kind': 'global'})
    _, repo = _get_repos(meta.get('mode') or 'backport')
    presence = nearby_presence(
        [s for s in syms if s['kind'] != 'patch_introduced'],
        repo, meta['commits']['pc'],
        ctx.get('file') or unit0.get('file', ''), pc_text)
    pa_usage: dict[str, str] = {}
    with open(pa_path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            for s in syms:
                if s['sym'] not in pa_usage and re.search(
                        r'\b' + re.escape(s['sym']) + r'\b', ln):
                    pa_usage[s['sym']] = ln.strip()[:120]
    evidence = []
    for s in syms:
        st = presence.get(s['sym'], 'present')
        ev = {'sym': s['sym'], 'kind': s['kind'], 'pc_status': st}
        if st == 'nearby_absent' and s['kind'] in ('call', 'global'):
            ev['nearest_top3'] = nearest_candidates(s['sym'], pc_path, task_dir)
        if s['sym'] in pa_usage:
            ev['pa_usage'] = pa_usage[s['sym']]
        evidence.append(ev)
    ctx['symbol_evidence'] = evidence
    return evidence

# def absent_symbols(evidence: list[dict]) -> list[dict]:
#     return [e for e in evidence if e['pc_status'] == 'absent']
