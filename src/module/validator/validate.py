import difflib
import os
import re
import subprocess
import shutil
import tempfile
from collections import Counter

from dataset.prepare import _get_repos
from module.locator.anchor import index_functions, name_sim, normalize_name
from module.locator.symbols import confirm_absent
from module.locator.units import snapshot_path
from utils.logger import get_logger
from utils.parser.ast_parser import ASTParser
from utils.parser.ctags_parser import ctags_parse

logger = get_logger(__name__)

if shutil.which('gcc') is None:
    logger.warning('gcc not found — compile blocking disabled '
                   '(arity checks still active)')

BRANCH_KINDS = ('if_statement', 'for_statement', 'while_statement',
                'switch_statement')
_COMMENT_RE = re.compile(r'^\s*(?:/\*|//|\*(?!\w))')
_WORD_RE = re.compile(r'[A-Za-z_]\w*')
_CALL_RE = re.compile(r'([A-Za-z_]\w*)\s*\(')
_WS_RE = re.compile(r'\s+')
_KEYWORDS = {'if', 'for', 'while', 'switch', 'return', 'sizeof'}
_MIN_LINE = 8
_MAX_ISSUES = 5
_MAX_ERRORS = 2000
_GCC_TIMEOUT = 60

def _branch_counts(src: str) -> dict[str, int] | None:
    if not src or '{' not in src:
        return None
    try:
        root = ASTParser.parse_code(src).root_node
    except Exception:
        return None
    counts = dict.fromkeys(BRANCH_KINDS, 0)
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in counts:
            counts[n.type] += 1
        stack.extend(n.children)
    return counts

def _stmt_kind(line: str) -> str:
    m = _WORD_RE.match(line)
    return m.group(0) if m else ''

def _line_no(gen_lines: list[str], needle: str) -> int | None:
    for i, l in enumerate(gen_lines):
        if l.strip() == needle:
            return i + 1
    return None

def _at(gen_lines: list[str], needle: str) -> str:
    n = _line_no(gen_lines, needle)
    return f' (your output line {n})' if n else ''

def _deletion_issues(units: list[dict], generated: str,
                     intents: dict | None = None) -> list[str]:
    gen_lines = generated.splitlines()
    gen_set = {l.strip() for l in gen_lines}
    issues = []
    for unit in units:
        intent = (intents or {}).get(unit['unit_id']) or {}
        conditional = bool(intent.get('conditional'))
        if intent.get('hunk_role') == 'incidental':
            continue
        hunks = unit.get('hunks') or []
        added_kinds = {_stmt_kind(t[2].strip()) for h in hunks
                       for t in h.get('add_lines', [])}
        added_kinds.discard('')
        kept = {c.strip() for h in hunks
                for c in h.get('context_before', []) + h.get('context_after', [])}
        for h in hunks:
            for t in h.get('delete_lines', []):
                line = t[2].strip()
                if len(line) < _MIN_LINE or line not in gen_set \
                        or line in kept:
                    continue
                if conditional:
                    continue
                if _stmt_kind(line) in added_kinds:
                    continue
                issues.append(f'source patch deleted line still present '
                              f'unchanged: `{line}`{_at(gen_lines, line)}')
    return issues

def _statement_roots(lines: list[str]) -> list[int]:
    roots = [0] * len(lines)
    depth = 0
    root = 0
    for i, l in enumerate(lines):
        if depth <= 0:
            root = i
        roots[i] = root
        depth += l.count('(') - l.count(')')
    return roots

def _attribution_issues(units: list[dict], original: str,
                        generated: str, ctx: dict | None = None) -> list[str]:
    src_lines = {t[2].strip() for u in units for h in (u.get('hunks') or [])
                 for t in h.get('add_lines', []) + h.get('delete_lines', [])}
    orig_lines = {l.strip() for l in original.splitlines()}
    pc_lines: set[str] = set()
    pc_call_names: set[str] = set()
    if ctx is not None:
        meta = ctx.get('meta') or {}
        u0 = units[0] if units else {}
        pc_path = snapshot_path(ctx.get('task_dir', ''), 'pc',
                                (meta.get('commits') or {}).get('pc', ''),
                                u0.get('pc_file') or u0.get('file', ''))
        if os.path.exists(pc_path):
            try:
                with open(pc_path, encoding='utf-8', errors='replace') as f:
                    pc_text = f.read()
                pc_lines = {l.strip() for l in pc_text.splitlines()}
                pc_call_names = {c for c in _CALL_RE.findall(pc_text)
                                 if c not in _KEYWORDS}
            except OSError:
                pc_lines = set()
    gen_lines = generated.splitlines()
    roots = _statement_roots(gen_lines)
    preproc_continues = False
    issues = []
    for ln_no, line in enumerate(gen_lines, 1):
        s = line.strip()
        if s.startswith('#'):
            preproc_continues = line.rstrip().endswith('\\')
            exempt = True
        elif preproc_continues:
            exempt = True
            preproc_continues = line.rstrip().endswith('\\')
        else:
            exempt = False
        if len(s) < _MIN_LINE or _COMMENT_RE.match(line) \
                or s in orig_lines or s in src_lines or s in pc_lines or exempt:
            continue
        stmt_calls = set(_CALL_RE.findall(gen_lines[roots[ln_no - 1]])) \
            | set(_CALL_RE.findall(line))
        if pc_call_names and stmt_calls & pc_call_names:
            lt = set(_WORD_RE.findall(s))
            if any(len(lt & set(_WORD_RE.findall(t))) / max(1, len(lt)) >= 0.4
                   for t in (src_lines | orig_lines)):
                continue
        issues.append(f'line not attributable to the source patch (keep only '
                      f'minimal changes): `{s}` (your output line {ln_no})')
    return issues

def _tok_overlap(a: str, b: str) -> float:
    ta, tb = Counter(_WORD_RE.findall(a)), Counter(_WORD_RE.findall(b))
    if not ta or not tb:
        return 0.0
    return sum((ta & tb).values()) / max(sum(ta.values()), sum(tb.values()))

_KEYLINE_RE = re.compile(
    r'\b(?:smp_[rw]?mb|[rw]?mb|read_barrier_depends|barrier|smp_store_release'
    r'|smp_load_acquire|spin_(?:un)?lock(?:_irq)?|mutex_(?:un)?lock'
    r'|INIT_(?:LIST_HEAD|HLIST_HEAD|WORK|DELAYED_WORK)|memset|memzero'
    r'|atomic[64]_(?:set|inc|dec|add|sub)|refcount_(?:set|inc|dec)'
    r'|list_add(?:_tail)?|hlist_add_head)\s*\(')
_PREPROC_OK = ('#include', '#define')

def account_add_lines(add_lines: list, gen_lines: list[str],
                      pc_words: set[str] = frozenset()) -> tuple[set, int]:
    joined = [' '.join(gen_lines[i:i + 3])
              for i in range(max(1, len(gen_lines) - 2))]
    accounted: set[int] = set()
    total = 0
    for i, t in enumerate(add_lines):
        line = t[2].strip()
        keyline = bool(_KEYLINE_RE.search(line)) \
            or line.startswith(_PREPROC_OK)
        if not keyline and (len(line) < _MIN_LINE
                            or line.startswith(('#', '/*', '*', '//'))):
            continue
        total += 1
        thr = 0.75 if len(_WORD_RE.findall(line)) <= 4 else 0.6
        if any(l.strip() == line or _tok_overlap(line, l.strip()) >= thr
               for l in gen_lines) \
                or any(_tok_overlap(line, j) >= thr for j in joined):
            accounted.add(i)
            continue
        callee = _CALL_RE.search(line + '(')
        callee = callee.group(1) if callee else ''
        if pc_words and callee and callee not in pc_words:
            accounted.add(i)
    return accounted, total

_NUMECHO_RE = re.compile(r'^\s*\d+\s')

def _numbered_echo_issues(units: list[dict], generated: str) -> list[str]:
    src: set[str] = set()
    for u in units:
        for h in (u.get('hunks') or []):
            for t in h.get('add_lines', []):
                src.add(t[2].strip())
    if not src:
        return []
    issues = []
    for i, line in enumerate(generated.splitlines(), 1):
        if not _NUMECHO_RE.match(line):
            continue
        stripped = _NUMECHO_RE.sub('', line, count=1).strip()
        if stripped in src and line.strip() not in src:
            issues.append(f'line-number echo: your output line {i} embeds a '
                          f'line-number prefix (`{line.strip()[:40]}`) — '
                          f'remove the number, output the code line only')
            if len(issues) >= 3:
                break
    return issues

def _unit_dispositions(units: list[dict], generated: str,
                       intents: dict | None = None,
                       ctx: dict | None = None):
    gen_lines = generated.splitlines()
    out_set = {l.strip() for l in gen_lines}
    pc_words: set[str] = set()
    if ctx is not None:
        meta = ctx.get('meta') or {}
        u0 = units[0] if units else {}
        pc_path = snapshot_path(ctx.get('task_dir', ''), 'pc',
                                (meta.get('commits') or {}).get('pc', ''),
                                u0.get('pc_file') or u0.get('file', ''))
        if os.path.exists(pc_path):
            try:
                with open(pc_path, encoding='utf-8', errors='replace') as f:
                    pc_words = set(_WORD_RE.findall(f.read()))
            except OSError:
                pc_words = set()
    dispositions: list[dict] = []
    issues: list[str] = []
    for unit in units:
        uid = unit.get('unit_id', '?')
        intent = (intents or {}).get(uid) or {}
        if intent.get('hunk_role') == 'incidental':
            dispositions.append({'unit_id': uid, 'status': 'exempt'})
            continue
        add_lines = [t for h in (unit.get('hunks') or [])
                     for t in h.get('add_lines', [])]
        del_lines = [t[2].strip() for h in (unit.get('hunks') or [])
                     for t in h.get('delete_lines', [])]
        accounted, total = account_add_lines(add_lines, gen_lines, pc_words)
        adopt_ok = total == 0 or len(accounted) / total >= 0.8
        weak_hunks: list[str] = []
        for h in (unit.get('hunks') or []):
            h_acc, h_total = account_add_lines(h.get('add_lines', []),
                                               gen_lines, pc_words)
            if h_total >= 2 and len(h_acc) / h_total < 0.5:
                weak_hunks.append(f"@{h.get('old_start', '?')}: "
                                  f"{h_total - len(h_acc)}/{h_total} not adopted")
                adopt_ok = False
        nontriv_del = [d for d in del_lines
                       if len(d) >= _MIN_LINE and not _COMMENT_RE.match(d)]
        gone = sum(1 for d in nontriv_del if d not in out_set)
        del_ok = not nontriv_del or gone / len(nontriv_del) >= 0.8
        gen_words = set(_WORD_RE.findall(generated))
        strict_accounted, _strict_total = account_add_lines(
            add_lines, gen_lines, gen_words)
        gaps = [t[2].strip() for i, t in enumerate(add_lines)
                if i not in strict_accounted and len(t[2].strip()) >= _MIN_LINE
                and not t[2].strip().startswith(('#', '/*', '*', '//'))]
        if adopt_ok and del_ok:
            if gaps:
                dispositions.append({'unit_id': uid,
                                     'status': 'addressed_with_gaps',
                                     'missing_lines': gaps})
            else:
                dispositions.append({'unit_id': uid, 'status': 'addressed'})
            continue
        miss = gaps[:2]
        survivors = [d for d in nontriv_del if d in out_set][:2]
        dispositions.append({'unit_id': uid, 'status': 'unaddressed',
                             'unadopted': gaps,
                             'surviving_deletions': survivors})
        why = []
        if weak_hunks:
            why.append('a merged hunk went missing: ' + '; '.join(weak_hunks[:2]))
        if total - len(accounted):
            why.append(f'{total - len(accounted)} added line(s) not adopted, '
                       f'first: `{miss[0]}`' if miss else
                       f'{total - len(accounted)} added line(s) not adopted')
        if len(nontriv_del) - gone:
            why.append(f'{len(nontriv_del) - gone} deleted line(s) still '
                       f'present, first: `{survivors[0]}`' if survivors else
                       f'{len(nontriv_del) - gone} deleted line(s) still present')
        goal = (intent.get('security_goal') or intent.get('summary') or '')[:80]
        issues.append(f"unit not addressed: {unit.get('file', '?')}"
                      f"::{unit.get('func', '?')} (unit {uid}) — {'; '.join(why)}"
                      + (f'; unit intent: {goal}' if goal else '')
                      + ' (incorporate this unit\'s change into the output '
                        'function)')
    return dispositions, issues

def _hunk_bodies(raw_diff: str | None) -> list[list[tuple[str, str]]]:
    bodies: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] | None = None
    for ln in (raw_diff or '').splitlines():
        if ln.startswith('@@'):
            if cur:
                bodies.append(cur)
            cur = []
        elif cur is None or ln.startswith(('diff --git', '--- ', '+++ ', 'index ')):
            continue
        elif ln.startswith('\\'):
            continue
        elif ln.startswith('+'):
            cur.append(('+', ln[1:].rstrip()))
        elif ln.startswith('-'):
            cur.append(('-', ln[1:].rstrip()))
        else:
            cur.append((' ', (ln[1:] if ln.startswith(' ') else ln).rstrip()))
    if cur:
        bodies.append(cur)
    return bodies

def _has_verbatim_hunk(units: list[dict], ctx: dict,
                       original: str | None) -> bool:
    hay_src = original
    if not hay_src:
        meta = ctx.get('meta') or {}
        if not units:
            return False
        pc_path = snapshot_path(ctx['task_dir'], 'pc',
                                meta['commits']['pc'],
                                units[0].get('pc_file') or units[0]['file'])
        if not os.path.exists(pc_path):
            return False
        try:
            with open(pc_path, encoding='utf-8', errors='replace') as f:
                hay_src = f.read()
        except OSError:
            return False
    hay = [l.rstrip() for l in hay_src.splitlines()]
    pos: dict[str, list[int]] = {}
    for i, l in enumerate(hay):
        pos.setdefault(l, []).append(i)
    for u in units:
        for body in _hunk_bodies(u.get('raw_diff')):
            if any(k in '-+' for k, _ in body) \
                    and _match_site(body, hay, pos) is not None:
                return True
    return False

_RESOURCE_FAMILIES = (
    ('lock', 'unlock'), ('get', 'put'), ('alloc', 'free'),
    ('create', 'destroy'), ('create', 'free'), ('register', 'unregister'),
    ('open', 'close'), ('start', 'stop'), ('request', 'release'),
    ('map', 'unmap'), ('begin', 'end'), ('enable', 'disable'),
)
_EXIT_TYPES = ('return_statement', 'break_statement', 'continue_statement')
_CONDITIONAL_TYPES = ('if_statement', 'while_statement', 'for_statement',
                      'switch_statement', 'conditional_expression',
                      'case_clause')

def _resource_pairs(units: list[dict]) -> dict[str, str]:
    added: set[str] = set()
    for u in units:
        for h in (u.get('hunks') or []):
            for _o, _n, t in h.get('add_lines', []):
                added.update(_CALL_RE.findall(t))
    pairs: dict[str, str] = {}
    for name in added:
        for a_sfx, r_sfx in _RESOURCE_FAMILIES:
            if name.endswith(a_sfx):
                partner = name[:-len(a_sfx)] + r_sfx
                if partner in added:
                    pairs[name] = partner
                break
    return pairs

def _walk_nodes(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)

def _nkey(n):
    return (n.type, n.start_byte, n.end_byte)

def _lca(a, b):
    seen = set()
    cur = a
    while cur is not None:
        seen.add(_nkey(cur))
        cur = cur.parent
    cur = b
    while cur is not None:
        if _nkey(cur) in seen:
            return cur
        cur = cur.parent
    return None

def _lca_child(n, lca):
    cur = n
    while cur is not None and cur.parent is not None:
        if _nkey(cur.parent) == _nkey(lca):
            return cur
        cur = cur.parent
    return None

# _ASSIGN_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*(?:\[[^]]*\])?\s*=[^=]')

def _bracket_balance_issues(original: str, generated: str,
                            units: list[dict]) -> tuple[list[str], list[str]]:
    pairs = _resource_pairs(units)
    if not pairs:
        return [], []
    orig_lines = {l.strip() for l in original.splitlines()}
    gen_lines = generated.splitlines()
    acquires: list[tuple[int, str]] = []
    for i, l in enumerate(gen_lines):
        if l.strip() in orig_lines or _COMMENT_RE.match(l):
            continue
        for m in _CALL_RE.finditer(l):
            if m.group(1) in pairs:
                acquires.append((i + 1, m.group(1)))
    if not acquires:
        return [], []
    if re.search(r'\bgoto\b', generated):
        return [], [f'(advisory) resource bracket `{acquires[0][1]}` present '
                    f'but the function uses goto-based cleanup; bracket '
                    f'balance was not verified']
    try:
        root = ASTParser.parse_code(generated).root_node
    except Exception:
        return [], []
    acq_nodes: list[tuple[int, str, object]] = []
    releases: list[tuple[int, str, object]] = []
    exits: list[tuple[int, object]] = []
    for n in _walk_nodes(root):
        if n.type in _EXIT_TYPES:
            exits.append((n.start_point[0] + 1, n))
        elif n.type == 'call_expression':
            line_no = n.start_point[0] + 1
            if gen_lines[line_no - 1].strip() in orig_lines:
                continue
            m = _CALL_RE.match(generated[n.start_byte:n.end_byte])
            if not m:
                continue
            if m.group(1) in pairs:
                acq_nodes.append((line_no, m.group(1), n))
            if m.group(1) in set(pairs.values()):
                releases.append((line_no, m.group(1), n))

    def _covers(acq_node, rel_node, exit_node) -> bool:
        if rel_node.start_byte >= exit_node.start_byte \
                and rel_node.end_byte <= exit_node.end_byte:
            return True
        lca = _lca(acq_node, exit_node)
        if lca is None:
            return False
        r_child = _lca_child(rel_node, lca)
        e_child = _lca_child(exit_node, lca)
        if r_child is None or e_child is None:
            return False
        if _nkey(r_child) == _nkey(e_child):
            return True
        return r_child.start_byte <= e_child.start_byte \
            and r_child.type not in _CONDITIONAL_TYPES

    issues = []
    for a_line, a_name, a_node in acq_nodes:
        rel_name = pairs[a_name]
        reported = False
        for e_line, e_node in exits:
            if e_line < a_line or reported:
                continue
            covered = any(a_line < r_line and rn == rel_name
                          and _covers(a_node, r_node, e_node)
                          for r_line, rn, r_node in releases)
            if not covered:
                issues.append(
                    f'resource bracket `{a_name}` (your output line '
                    f'{a_line}) is not released on every path: the '
                    f'{e_node.type.split("_")[0]} at line {e_line} exits '
                    f'while it is held — release it before that exit')
                reported = True
    return issues, []

# _DECL_TYPE_TMPL = (r'\b(?:struct|union|enum|unsigned|signed|bool|char|short|'
#                    r'int|long|size_t|ssize_t|ptrdiff_t|off_t|u8|u16|u32|u64|'
#                    r's8|s16|s32|s64|void|float|double)\b[^;]*\b{name}\b\s*[;,)]')

def _window_sides(body: list[tuple[str, str]], lo: int, hi: int
                  ) -> tuple[list[str], list[str]]:
    pa = [t for k, t in body[lo:hi] if k in (' ', '-')]
    pb = [t for k, t in body[lo:hi] if k in (' ', '+')]
    return pa, pb

def _match_site(body: list[tuple[str, str]], hay: list[str],
                pos: dict[str, list[int]]) -> tuple[list[str], list[str]] | None:
    def find(pa: list[str]) -> bool:
        if not pa or len(pa) > len(hay):
            return False
        return any(i + len(pa) <= len(hay) and hay[i:i + len(pa)] == pa
                   for i in pos.get(pa[0], ()))

    lo, hi = 0, len(body)
    while lo < hi and any(k in '-+' for k, _ in body[lo:hi]):
        pa, pb = _window_sides(body, lo, hi)
        if find(pa):
            return pa, pb
        if body[lo][0] == ' ':
            lo += 1
        elif body[hi - 1][0] == ' ':
            hi -= 1
        else:
            return None
    return None

def _word_count(text: str, symbol: str) -> int:
    return len(re.findall(r'\b' + re.escape(symbol) + r'\b', text))

_TREE_CACHE: dict[tuple[str, str, str], int | str | None] = {}

def _tree_symbol_files(task_dir: str, meta: dict, symbol: str) -> list[str]:
    commit = (meta.get('commits') or {}).get('pc', '')
    key = (task_dir, commit, 'sym:' + symbol)
    if key not in _TREE_CACHE:
        files: list[str] | int = []
        try:
            _, repo = _get_repos(meta.get('mode') or 'backport')
            r = subprocess.run(
                ['git', '-C', repo, 'grep', '-l', '-w', symbol, commit],
                capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                prefix = f'{commit}:'
                files = [ln[len(prefix):] if ln.startswith(prefix) else ln
                         for ln in r.stdout.splitlines()][:5]
            elif r.returncode == 1:
                files = []
            else:
                files = -1
        except (OSError, subprocess.TimeoutExpired):
            files = -1
        _TREE_CACHE[key] = files
    got = _TREE_CACHE[key]
    return list(got) if isinstance(got, list) else []

def _tree_type_include(task_dir: str, meta: dict, name: str) -> str | None:
    commit = (meta.get('commits') or {}).get('pc', '')
    key = (task_dir, commit, 'type:' + name)
    if key not in _TREE_CACHE:
        path: str | None = None
        try:
            _, repo = _get_repos(meta.get('mode') or 'backport')
            r = subprocess.run(
                ['git', '-C', repo, 'grep', '-l', '-F', f'struct {name} {{',
                 commit, '--', 'include'],
                capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout:
                first = r.stdout.splitlines()[0].split(':', 1)[-1]
                if first.startswith('include/'):
                    path = '<' + first[len('include/'):].strip() + '>'
        except (OSError, subprocess.TimeoutExpired):
            path = None
        _TREE_CACHE[key] = path
    return _TREE_CACHE[key] if isinstance(_TREE_CACHE[key], str) else None

def _pc_usage_hint(pc_text: str, symbol: str) -> str | None:
    scored = sorted({c for c in set(_CALL_RE.findall(pc_text))
                     if c not in _KEYWORDS and c != symbol},
                    key=lambda c: (difflib.SequenceMatcher(None, symbol, c).ratio(),
                                   -len(c)), reverse=True)
    for cand, in [(c,) for c in scored]:
        if difflib.SequenceMatcher(None, symbol, cand).ratio() < 0.6:
            break
        pat = re.compile(r'\b' + re.escape(cand) + r'\s*\(')
        hits = [l.strip() for l in pc_text.splitlines() if pat.search(l)]
        log_line = next((l for l in hits if '"' in l or 'KERN_' in l), None)
        if log_line:
            return log_line
    return None

_TYPE_REF_RE = re.compile(r'\b(?:struct|union|enum)\s+([A-Za-z_]\w*)')
_SR_RE = re.compile(r'<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE',
                    re.DOTALL)

def _nearest_pc_symbols(pc_text: str, symbol: str, k: int = 3,
                        floor: float = 0.45) -> list[tuple[str, int]]:
    first: dict[str, int] = {}
    try:
        for fn in index_functions(pc_text):
            first.setdefault(fn['name'], fn['start_line'])
    except Exception:
        pass
    for i, l in enumerate(pc_text.splitlines(), 1):
        for m in _CALL_RE.finditer(l):
            first.setdefault(m.group(1), i)
    scored = sorted(((name_sim(symbol, c), -line, c)
                     for c, line in first.items() if c != symbol),
                    reverse=True)
    return [(c, -neg) for sim, neg, c in scored[:k] if sim >= floor]

def _file_scope_loss_issues(units: list[dict], generated: str,
                            intents: dict | None = None) -> list[str]:
    m = _SR_RE.search(generated)
    if not m:
        return []
    search = Counter(l.strip() for l in m.group(1).splitlines())
    rep = Counter(l.strip() for l in m.group(2).splitlines())
    gen_lines = generated.splitlines()
    issues = []
    for unit in units:
        intent = (intents or {}).get(unit['unit_id']) or {}
        patch_deleted = {t[2].strip() for h in (unit.get('hunks') or [])
                         for t in h.get('delete_lines', [])}
        for s, n in search.items():
            if n - rep.get(s, 0) <= 0 or s in patch_deleted \
                    or intent.get('conditional'):
                continue
            if _COMMENT_RE.match(s) and len(s) >= 2:
                issues.append(f'comment line of the SEARCH block is missing '
                              f'from the REPLACE and the source patch does '
                              f'not delete it: `{s}`{_at(gen_lines, s)}')
            elif len(s) >= _MIN_LINE:
                issues.append(f'line of the SEARCH block is missing from the '
                              f'REPLACE and the source patch does not '
                              f'delete it: `{s}`{_at(gen_lines, s)}')
    return issues

def _unknown_type_issues(original: str, generated: str, units: list[dict],
                         ctx: dict) -> list[str]:
    meta = ctx.get('meta') or {}
    unit = units[0]
    pc_path = snapshot_path(ctx['task_dir'], 'pc', meta['commits']['pc'],
                            unit.get('pc_file') or unit['file'])
    pa_path = snapshot_path(ctx['task_dir'], 'pa', meta['commits']['pa'],
                            unit['file'])
    if not (os.path.exists(pc_path) and os.path.exists(pa_path)):
        return []
    with open(pc_path, encoding='utf-8', errors='replace') as f:
        pc_text = f.read()
    with open(pa_path, encoding='utf-8', errors='replace') as f:
        pa_text = f.read()
    orig_lines = {l.strip() for l in original.splitlines()}
    new_text = '\n'.join(l for l in generated.splitlines()
                         if l.strip() not in orig_lines)
    pc_words = set(_WORD_RE.findall(pc_text))
    pa_words = set(_WORD_RE.findall(pa_text))
    for u in units:
        for h in (u.get('hunks') or []):
            for t in h.get('add_lines', []) + h.get('delete_lines', []):
                pa_words.update(_WORD_RE.findall(t[2]))
    include_lines = [l.strip() for l in generated.splitlines()
                     if l.strip().startswith('#')]
    issues = []
    seen: set[str] = set()
    for m in _TYPE_REF_RE.finditer(new_text):
        name = m.group(1)
        if name in seen or name in pc_words or name not in pa_words:
            continue
        seen.add(name)
        header = _tree_type_include(ctx['task_dir'], meta, name)
        if header and any(header.strip('<>') in l for l in include_lines):
            continue
        if header:
            issues.append(
                f'type `{m.group(0)}` is used by the transplant but is not '
                f'known in the PC file — it is defined in the PC tree under '
                f'{header}: add the matching #include line immediately '
                f'before the function in your output')
        else:
            issues.append(
                f'type `{m.group(0)}` is not known anywhere in the PC file '
                f'or the PC tree — adapt to the type/structure PC uses '
                f'instead of copying the PA-world type')
    return issues

def _call_argc(s: str, i: int) -> int | None:
    depth, commas = 0, 0
    saw_arg = False
    for j in range(i, len(s)):
        ch = s[j]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return (commas + 1) if saw_arg else 0
        elif ch == ',' and depth == 1:
            commas += 1
        if depth >= 1 and not (depth == 1 and ch == '(') and ch not in ' \t':
            saw_arg = True
    return None

def _call_arities(text: str) -> dict[str, tuple[int, str]]:
    arities: dict[str, tuple[int, str]] = {}
    dropped: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('#') or _COMMENT_RE.match(line):
            continue
        for m in _CALL_RE.finditer(s):
            name = m.group(1)
            if name in _KEYWORDS or len(name) < 4 or name in dropped:
                continue
            n = _call_argc(s, s.index('(', m.start()))
            if n is None:
                dropped.add(name)
                arities.pop(name, None)
                continue
            prev = arities.get(name)
            if prev is None:
                arities[name] = (n, s)
            elif prev[0] != n:
                dropped.add(name)
                arities.pop(name, None)
    return arities

def _arity_issues(original: str, generated: str,
                  ctx: dict | None = None) -> list[str]:
    if ctx is None:
        return []
    meta = ctx.get('meta') or {}
    u0_file = ''
    if ctx.get('units'):
        u0 = ctx['units'][0]
        u0_file = u0.get('pc_file') or u0.get('file', '')
    if not u0_file:
        return []
    pc_path = snapshot_path(ctx.get('task_dir', ''), 'pc',
                            (meta.get('commits') or {}).get('pc', ''), u0_file)
    if not os.path.exists(pc_path):
        return []
    try:
        with open(pc_path, encoding='utf-8', errors='replace') as f:
            pc_text = f.read()
    except OSError:
        return []
    arities = _call_arities(pc_text)
    if not arities:
        return []
    orig_set = {l.strip() for l in original.splitlines()}
    issues = []
    for line in generated.splitlines():
        s = line.strip()
        if len(s) < 4 or s.startswith('#') or _COMMENT_RE.match(line) \
                or s in orig_set:
            continue
        for m in _CALL_RE.finditer(s):
            name = m.group(1)
            if name in _KEYWORDS or len(name) < 4:
                continue
            want = arities.get(name)
            if want is None:
                continue
            depth, commas = 0, 0
            saw_arg = False
            closed = False
            for ch in s[s.index('(', m.start()):]:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        closed = True
                        break
                elif ch == ',' and depth == 1:
                    commas += 1
                if depth >= 1 and not (depth == 1 and ch == '(') and ch not in ' \t':
                    saw_arg = True
            if not closed:
                continue
            got = (commas + 1) if saw_arg else 0
            if got != want[0]:
                issues.append(
                    f'`{name}` takes {want[0]} argument(s) in this PC branch '
                    f'(e.g. `{want[1][:60]}`) but your output line calls it '
                    f'with {got} — use the PC call form')
                if len(issues) >= 3:
                    return issues
                break
    return issues

def _undefined_symbol_issues(original: str, generated: str,
                             units: list[dict], ctx: dict) -> tuple[list[str], list[str]]:
    meta = ctx.get('meta') or {}
    unit = units[0]
    rel = unit['file']
    pc_path = snapshot_path(ctx['task_dir'], 'pc', meta['commits']['pc'],
                            unit.get('pc_file') or rel)
    pa_path = snapshot_path(ctx['task_dir'], 'pa', meta['commits']['pa'], rel)
    if not (os.path.exists(pc_path) and os.path.exists(pa_path)):
        return [], []
    with open(pc_path, encoding='utf-8', errors='replace') as f:
        pc_text = f.read()
    with open(pa_path, encoding='utf-8', errors='replace') as f:
        pa_text = f.read()

    orig_lines = {l.strip() for l in original.splitlines()}
    callees = set()
    for line in generated.splitlines():
        if line.strip() in orig_lines or _COMMENT_RE.match(line):
            continue
        for m in _CALL_RE.finditer(line):
            sym = m.group(1)
            if sym not in _KEYWORDS:
                callees.add(sym)

    issues, notes = [], []
    for sym in sorted(callees):
        if sym not in pa_text or _word_count(pc_text, sym) > 0:
            continue
        tree_files = _tree_symbol_files(ctx['task_dir'], meta, sym)
        if tree_files:
            notes.append(
                f'symbol `{sym}` is not in this PC file but exists in the '
                f'PC tree (e.g. {tree_files[0]}) — advisory: make sure the '
                f'PC file can see its declaration')
            continue
        hint = _pc_usage_hint(pc_text, sym)
        if hint:
            issues.append(
                f'symbol `{sym}` exists in the PA file but nowhere in the '
                f'PC file or the PC tree — PC performs this operation as '
                f'`{hint}`; adapt the call to the PC form, or drop the '
                f'line if PC has no equivalent')
        else:
            near = _nearest_pc_symbols(pc_text, sym)
            near_txt = ''
            if near:
                near_txt = (' Closest names in this PC file: '
                            + ', '.join(f'`{n}` (line {ln})' for n, ln in near)
                            + ' — adapt the call to whichever performs this '
                              'operation.')
            issues.append(
                f'symbol `{sym}` exists in the PA file but nowhere in the '
                f'PC file — PC cannot resolve it; adapt the call to how '
                f'the PC code performs this operation, or drop the line '
                f'if PC has no equivalent{near_txt}')
    return issues, notes

_DIAG_RE = re.compile(
    r'^(.*?):(\d+)(?::(\d+))?:\s*(fatal error|error|warning):\s*(.*)$')

_NOISE_RE = re.compile(
    r'No such file or directory'
    r'|CONFIG_[A-Za-z0-9_]+'
    r'|/#include'
    r'|library was built')

_INCLUDE_CACHE: dict[str, list[str]] = {}

# def _enabled() -> bool:
#     return os.environ.get('SEMPORT_COMPILE_CHECK', '1') != '0'

def _include_dirs(pc_dir: str) -> list[str]:
    if pc_dir in _INCLUDE_CACHE:
        return _INCLUDE_CACHE[pc_dir]
    candidates = [
        os.path.join(pc_dir, 'include'),
        os.path.join(pc_dir, 'include', 'uapi'),
        os.path.join(pc_dir, 'arch', 'x86', 'include'),
        os.path.join(pc_dir, 'arch', 'x86', 'include', 'uapi'),
        os.path.join(pc_dir, 'arch', 'x86', 'include', 'generated'),
        os.path.join(pc_dir, 'arch', 'x86', 'include', 'generated', 'uapi'),
    ]
    odir = os.environ.get('SEMPORT_KBUILD_O', '/tmp/linux-build')
    candidates += [os.path.join(odir, 'include'),
                   os.path.join(odir, 'arch', 'x86', 'include'),
                   os.path.join(odir, 'arch', 'x86', 'include', 'generated'),
                   os.path.join(odir, 'arch', 'x86', 'include', 'uapi'),
                   os.path.join(odir, 'arch', 'x86', 'include',
                                'generated', 'uapi')]
    dirs = [d for d in candidates if os.path.isdir(d)]
    _INCLUDE_CACHE[pc_dir] = dirs
    return dirs

def _run_gcc(src_path: str, pc_dir: str, extra_dirs=()) -> str | None:
    gcc = shutil.which('gcc')
    if not gcc:
        return None
    cmd = [gcc, '-fsyntax-only', '-D__KERNEL__',
           '-Werror=implicit-function-declaration', '-Werror=implicit-int']
    kconfig = os.path.join(pc_dir, 'include', 'linux', 'kconfig.h')
    if os.path.exists(kconfig):
        cmd += ['-include', kconfig]
    for d in _include_dirs(pc_dir):
        cmd += ['-I', d]
    for d in extra_dirs:
        cmd += ['-I', d]
    cmd.append(src_path)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=_GCC_TIMEOUT)
        return r.stderr or ''
    except subprocess.TimeoutExpired:
        logger.info('  compile check: gcc timeout — skipped')
        return None
    except Exception as exc:
        logger.warning(f'  compile check: gcc failed: {exc}')
        return None

def _parse(output: str, src_name: str) -> list[dict]:
    out = []
    for ln in output.split('\n'):
        m = _DIAG_RE.match(ln.strip())
        if not m:
            continue
        path, line, _col, sev, msg = m.groups()
        out.append({'line': int(line), 'message': msg.strip(), 'severity': sev,
                    'in_src': os.path.basename(path) == src_name})
        if len(out) >= _MAX_ERRORS:
            break
    return out

def _symbol_evidence_issues(generated: str, ctx: dict | None) -> list[str]:
    if not ctx:
        return []
    evidence = [e for e in (ctx.get('symbol_evidence') or [])
                if e.get('kind') != 'patch_introduced']
    if not evidence:
        return []
    issues = []
    for ev in evidence:
        st = ev.get('pc_status')
        if st not in ('absent', 'nearby_absent'):
            continue
        sym = ev['sym']
        pat = (r'(?:->|\.\s*)' + re.escape(sym) + r'\b' if ev['kind'] == 'field'
               else r'\b' + re.escape(sym) + r'\b')
        if not re.search(pat, generated):
            continue
        if st == 'nearby_absent':
            if not confirm_absent(sym, ev['kind'], ctx):
                continue
        near = ev.get('nearest_top3') or []
        issues.append(f"symbol `{sym}` ({ev['kind']}) is used but absent "
                      f"from the PC tree"
                      + (f" — nearest: {', '.join(near)}" if near else '')
                      + '; use the PC-side equivalent')
        if len(issues) >= 3:
            break
    return issues

def _key(d: dict) -> tuple[str, str]:
    return (d['severity'], d['message'])

def _params_in_sig(sig: str) -> int | None:
    s = sig.strip()
    if not (s.startswith('(') and s.endswith(')')):
        return None
    s = s[1:-1]
    if s.strip() in ('', 'void'):
        return 0
    depth = commas = 0
    for ch in s:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch == ',' and depth == 0:
            commas += 1
    return commas + 1

def _def_arities(path: str) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    try:
        tags = ctags_parse(path)
    except Exception:
        tags = {}
    for t in tags.values():
        kind = t.get('kind') or b''
        if isinstance(kind, bytes):
            kind = kind.decode('utf-8', 'replace')
        if kind not in ('f', 'function'):
            continue
        name = t.get('name') or b''
        if isinstance(name, bytes):
            name = name.decode('utf-8', 'replace')
        sig = t.get('signature') or ''
        if isinstance(sig, bytes):
            sig = sig.decode('utf-8', 'replace')
        n = _params_in_sig(sig)
        try:
            ln = int(t.get('start_line'))
        except (TypeError, KeyError, ValueError):
            continue
        if n is not None and name:
            out[name] = (n, ln)
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            text = f.read()
        for fn in index_functions(text):
            if fn.get('params') is not None and fn['params'] >= 0 \
                    and fn['name'] not in out:
                out[fn['name']] = (fn['params'], fn.get('start_line') or 0)
    except Exception:
        pass
    return out

def file_arity_mismatch_issues(mod_text: str,
                               baseline_text: str
                               ) -> list[tuple[str, int, int]]:
    if not mod_text:
        return []

    def _mismatches(text: str, defs: dict[str, tuple[int, int]]) -> set:
        out = set()
        for ln, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith('#') or _COMMENT_RE.match(line):
                continue
            for m in _CALL_RE.finditer(s):
                name = m.group(1)
                if name in _KEYWORDS or len(name) < 4 or name not in defs:
                    continue
                if re.match(rf'^[\w\*\s]+?\b{name}\s*\(', s):
                    continue
                argc = _call_argc(s, s.index('(', m.start()))
                if argc is not None and argc != defs[name][0]:
                    out.add((name, argc, defs[name][0], ln, defs[name][1]))
        return out

    with tempfile.TemporaryDirectory(prefix='semport_ar_') as tmp:
        mp, bp = os.path.join(tmp, 'm.c'), os.path.join(tmp, 'b.c')
        with open(mp, 'w', errors='ignore') as f:
            f.write(mod_text)
        with open(bp, 'w', errors='ignore') as f:
            f.write(baseline_text)
        base = {(a, b, c) for a, b, c, _, _ in
                _mismatches(baseline_text, _def_arities(bp))}
        mod = _mismatches(mod_text, _def_arities(mp))
    issues: list[tuple[str, int, int]] = []
    for name, argc, n, ln, dln in sorted(mod, key=lambda x: x[3]):
        if (name, argc, n) in base:
            continue
        issues.append((f"arity mismatch: call of `{name}' passes "
                       f"{argc} args, definition has {n}", ln, dln))
        if len(issues) >= 3:
            break
    return issues

def placement_issues(mod_text: str, pristine_text: str,
                     regions: list[dict], expected_fn: str | None,
                     check_depth: bool = True) -> list[tuple[str, int]]:
    if not mod_text or not regions:
        return []
    lines = mod_text.splitlines()
    pristine_lines = {l.strip() for l in (pristine_text or '').splitlines()}
    try:
        fn_spans = index_functions(mod_text)
    except Exception:
        fn_spans = []
    issues: list[tuple[str, int]] = []
    for r in regions:
        lo, hi = max(1, r['new_start']), min(len(lines), r['new_end'])
        if lo > hi:
            continue
        host = next((f['name'] for f in fn_spans
                     if f['start_line'] <= lo <= f['end_line']), None)
        if expected_fn and host is not None \
                and normalize_name(host) != normalize_name(expected_fn):
            issues.append((f'inserted into wrong function: expected '
                           f'`{expected_fn}`, content landed in `{host}`', lo))
            continue
        if not check_depth:
            continue
        depth = sum(l.count('{') - l.count('}')
                    for l in lines[:lo - 1])
        for i in range(lo, hi + 1):
            s = lines[i - 1].strip()
            is_added = bool(s) and not s.startswith('#') \
                and not _COMMENT_RE.match(s) and s not in pristine_lines
            if is_added and depth == 0 and s.endswith(';') and '(' in s:
                issues.append(('file-scope statement: added line landed '
                               f'outside any function (line {i}): '
                               f'`{s[:60]}`', i))
            depth += lines[i - 1].count('{') - lines[i - 1].count('}')
    return issues

def check_compile(code: str, baseline_code: str, pc_dir: str,
                  tag: str = 'file', rel: str | None = None
                  ) -> list[tuple[str, int]]:
    if not code or not code.strip():
        return []
    extra = []
    if rel and os.path.dirname(rel):
        d = os.path.join(pc_dir, os.path.dirname(rel))
        if os.path.isdir(d):
            extra.append(d)
    with tempfile.TemporaryDirectory(prefix='semport_cc_') as tmp:
        src = os.path.join(tmp, 'gen.c')
        with open(src, 'w', errors='ignore') as f:
            f.write(code)
        out = _run_gcc(src, pc_dir, extra)
        if out is None:
            return []
        diags = [d for d in _parse(out, 'gen.c') if d['in_src']]
        if not diags:
            return []
        base_keys: Counter = Counter()
        if baseline_code and baseline_code.strip():
            base = os.path.join(tmp, 'base.c')
            with open(base, 'w', errors='ignore') as f:
                f.write(baseline_code)
            base_out = _run_gcc(base, pc_dir, extra) or ''
            base_keys = Counter(_key(d) for d in _parse(base_out, 'base.c'))
        issues = []
        seen: Counter = Counter()
        for d in diags:
            k = _key(d)
            seen[k] += 1
            if seen[k] <= base_keys.get(k, 0) or d['severity'] != 'error' \
                    or _NOISE_RE.search(d['message']):
                continue
            issues.append((f"syntax error (line {d['line']}): "
                           f"{d['message']}", d['line']))
            if len(issues) >= 2:
                break
        if issues:
            logger.info(f'  compile check ({tag}): {len(issues)} actionable')
        return issues

def validate_function(group: dict, ctx: dict) -> dict:
    notes: list[str] = []
    generated = (group.get('func_text') or '').strip('\n')
    if not generated:
        return {'ok': False,
                'issues': ['no output: the response carried no usable '
                           'function body / SEARCH-REPLACE block — return '
                           'the transplant'],
                'notes': notes, 'dispositions': [], 'addressed_count': 0}
    ctx_path = os.path.join(ctx['task_dir'], 'anchor',
                            f"{group['safe_id']}.ctx.c")
    original = None
    if os.path.exists(ctx_path):
        with open(ctx_path, encoding='utf-8', errors='replace') as f:
            original = f.read()
    issues = []
    units = group.get('units') or []
    dispositions: list[dict] = []
    issues += _numbered_echo_issues(units, generated)
    if original is None:
        notes.append('anchor ctx.c missing: only the undefined-symbol check applies')
        original = ''
        issues += _file_scope_loss_issues(units, generated, ctx.get('intents'))
    else:
        if _WS_RE.sub('', generated) == _WS_RE.sub('', original):
            issues.append(
                "the output is identical to the PC original — none of the "
                "source patch's changes are present; apply the source patch "
                "to the PC function")
        issues += _deletion_issues(units, generated, ctx.get('intents'))
        bracket_issues, bracket_notes = _bracket_balance_issues(
            original, generated, units)
        issues += bracket_issues
        notes += bracket_notes
        arity = _arity_issues(original, generated, ctx)
        issues += arity
        dispositions, disp_issues = _unit_dispositions(
            units, generated, ctx.get('intents'), ctx)
        issues += disp_issues
        issues += _attribution_issues(units, original, generated, ctx)
        orig_c, gen_c = _branch_counts(original), _branch_counts(generated)
        if orig_c and gen_c:
            lost = [f'branch structure lost: {k} {orig_c[k]} -> {gen_c[k]}'
                    for k in BRANCH_KINDS if gen_c[k] < orig_c[k]]
            if lost:
                if not group.get('incidental_units') and (
                        _has_verbatim_hunk(units, ctx, original)
                        or (group.get('anchor') or {}).get('status') == 'EXACT'):
                    issues += lost
                else:
                    notes += ['(advisory, drifted region — inlining/unwrap '
                              'may be the legitimate adaptation) ' + m
                              for m in lost]
    sym_issues, sym_notes = _undefined_symbol_issues(original, generated,
                                                     units, ctx)
    issues += sym_issues
    notes += sym_notes
    issues += _unknown_type_issues(original, generated, units, ctx)
    issues += _symbol_evidence_issues(generated, ctx)
    unit_issues = list(dict.fromkeys(
        i for i in issues if i.startswith('unit not addressed')))
    other = list(dict.fromkeys(i for i in issues
                               if not i.startswith('unit not addressed')))
    issues = unit_issues + other[:_MAX_ISSUES]
    addressed = sum(d['status'] in ('addressed', 'addressed_with_gaps', 'exempt')
                    for d in dispositions)
    if issues:
        logger.info(f'  validator[{group["safe_id"]}]: {len(issues)} issue(s)')
    return {'ok': not issues, 'issues': issues, 'notes': notes,
            'dispositions': dispositions, 'addressed_count': addressed}
