import os
import subprocess
import tempfile

from dataset.prepare import _get_repos

def _diff_targets(diff_text: str) -> list[str]:
    seen, out = set(), []
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            rel = line[len('+++ b/'):].strip()
            if rel and rel != '/dev/null' and rel not in seen:
                seen.add(rel)
                out.append(rel)
    return out

def check_apply(diff_text: str, task_dir: str, meta: dict) -> tuple[bool, str]:
    if not diff_text.strip():
        return False, 'empty diff'
    _, repo = _get_repos(meta.get('mode') or 'backport')
    with tempfile.TemporaryDirectory(prefix='apply_check_') as tmp:
        subprocess.run(['git', 'init', '-q', tmp], check=True, capture_output=True)
        for rel in _diff_targets(diff_text):
            r = subprocess.run(['git', '-C', repo, 'show',
                                f"{meta['commits']['pc']}:{rel}"],
                               capture_output=True, timeout=30)
            if r.returncode != 0:
                continue
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst) or tmp, exist_ok=True)
            with open(dst, 'wb') as f:
                f.write(r.stdout)
        r = subprocess.run(['git', '-C', tmp, 'apply', '--check', '-'],
                           input=diff_text, capture_output=True, text=True, timeout=60)
        return r.returncode == 0, (r.stderr.strip() or 'ok')[:300]
