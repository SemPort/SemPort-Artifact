import ctags
import hashlib
import os
import subprocess
import tempfile

def ctags_parse(filepath: str, tag_dir: str | None = None) -> dict:
    if tag_dir:
        os.makedirs(tag_dir, exist_ok=True)
        digest = hashlib.sha1(
            f'{os.getpid()}:{os.path.abspath(filepath)}'.encode()).hexdigest()[:8]
        tag_path = os.path.join(tag_dir, f'tag_{os.getpid()}_{digest}')
        return _parse(filepath, tag_path)
    with tempfile.TemporaryDirectory(prefix='ctags_') as tmp:
        return _parse(filepath, os.path.join(tmp, 'tags'))

def _parse(filepath: str, tag_path: str) -> dict:
    try:
        cmd = ['ctags', '-f', tag_path, '--fields=+neS', '-R',
               '--kinds-C++=-m', '--kinds-C++=-e', filepath]
        subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        tags = ctags.CTags(tag_path)
        entry = ctags.TagEntry()
        results = dict()

        while tags.next(entry):
            try:
                try:
                    end_line = int(entry[b'end'])
                except (TypeError, KeyError):
                    end_line = int(entry['lineNumber'])
                try:
                    signature = entry[b'signature']
                except (TypeError, KeyError):
                    signature = b''
                tag_dict = {
                    "name": entry['name'],
                    "file": entry['file'],
                    "pattern": entry['pattern'],
                    "kind": entry['kind'],
                    'start_line': int(entry['lineNumber']),
                    'end_line': end_line,
                    'signature': signature,
                }
            except TypeError:
                continue
            results[int(tag_dict['start_line'])] = tag_dict

        return {k: results[k] for k in sorted(results.keys())}
    finally:
        if os.path.exists(tag_path):
            os.unlink(tag_path)
