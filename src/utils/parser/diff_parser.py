import re
from utils.logger import get_logger

logger = get_logger(__name__)

class Diff:

    def __init__(self, path):
        self._path = path
        with open(self._path, 'r') as _f:
            self._content = _f.readlines()
        logger.info('Parsing diff: %s (%d lines)', path, len(self._content))
        self._diff_chunks = self.__diff_parser()
        for fname, chunks in self._diff_chunks.items():
            logger.debug('  File: %s, %d chunk(s)', fname, len(chunks))

    def __diff_parser(self) -> dict:
        diff_chunks = dict()

        i_chunk = None
        i_file = None
        i_func = None
        a_lineno = -1
        b_lineno = -1

        pattern_1 = re.compile(r'^(diff)\s+(-.*?)\s+(?P<a>.*?)\s+(?P<b>.*?)$')
        pattern_2 = re.compile(r'^(index)\s+([a-z0-9]+)..([a-z0-9]+)\s*(.*)$')
        pattern_3 = re.compile(r'^(---|\+\+\+)\s+(\S+)\s*(.*)$')
        pattern_4 = re.compile(r'^(@@)\s+(?P<a>-[0-9]+,[0-9]+)\s+(?P<b>\+[0-9]+,[0-9]+)\s+(@@)\s*(?P<signature>.*)$')
        pattern_4_1 = re.compile(r'^.*\s(?P<func>.+?)\(.*$')

        for _line in self._content:
            _strip_line = _line.strip()
            if match := pattern_1.search(_strip_line):
                if i_chunk is not None:
                    if i_func and i_func in i_chunk \
                            and '_current_hunk' in i_chunk[i_func]:
                        i_chunk[i_func]['hunks'].append(
                            i_chunk[i_func].pop('_current_hunk'))
                    diff_chunks[i_file] = i_chunk
                i_chunk = dict()
                b_path = match.group('b')
                if b_path.startswith('b/') or b_path.startswith('b:'):
                    i_file = b_path[2:]
                else:
                    i_file = b_path.split('/')[-1]
                i_func = None
            elif match := pattern_2.search(_strip_line):
                continue
            elif match := pattern_3.search(_strip_line):
                if i_chunk is None:
                    i_chunk = dict()
                    if i_file is None:
                        header_path = match.group(2)
                        if header_path.startswith('a/') or header_path.startswith('a:'):
                            i_file = header_path[2:]
                        else:
                            i_file = header_path.split('/')[-1]
                continue
            elif match := pattern_4.search(_strip_line):
                a_lineno = int(match.group('a').removeprefix('-').split(',')[0])
                b_lineno = int(match.group('b').removeprefix('+').split(',')[0])
                i_signature = match.group('signature')

                old_func = i_func
                if m := pattern_4_1.search(i_signature):
                    i_func = m.group('func').removeprefix('*')
                else:
                    i_func = i_signature if i_signature.strip() else f'__anon_{a_lineno}'

                if old_func and old_func in i_chunk and '_current_hunk' in i_chunk[old_func]:
                    i_chunk[old_func]['hunks'].append(i_chunk[old_func].pop('_current_hunk'))

                if i_func not in i_chunk:
                    i_chunk[i_func] = dict()
                    i_chunk[i_func]['signature'] = i_signature
                    i_chunk[i_func]['delete_lines'] = list()
                    i_chunk[i_func]['delete_lineno'] = list()
                    i_chunk[i_func]['add_lines'] = list()
                    i_chunk[i_func]['add_lineno'] = list()
                    i_chunk[i_func]['hunks'] = list()

                i_chunk[i_func]['_current_hunk'] = {
                    'delete_lines': list(),
                    'delete_lineno': list(),
                    'add_lines': list(),
                    'add_lineno': list(),
                    'context_before': list(),
                    'context_after': list(),
                    'signature': i_signature,
                    '_has_changes': False,
                }
            elif _strip_line.startswith('-'):
                entry = (a_lineno, _line.removeprefix('-'), _strip_line.removeprefix('-').strip())
                i_chunk[i_func]['delete_lines'].append(entry)
                i_chunk[i_func]['delete_lineno'].append(a_lineno)
                if '_current_hunk' in i_chunk[i_func]:
                    cur = i_chunk[i_func]['_current_hunk']
                    cur['delete_lines'].append(entry)
                    cur['delete_lineno'].append(a_lineno)
                    cur['_has_changes'] = True
                a_lineno = a_lineno + 1
            elif _strip_line.startswith('+'):
                entry = (b_lineno, _line.removeprefix('+'), _strip_line.removeprefix('+').strip())
                i_chunk[i_func]['add_lines'].append(entry)
                i_chunk[i_func]['add_lineno'].append(b_lineno)
                if '_current_hunk' in i_chunk[i_func]:
                    cur = i_chunk[i_func]['_current_hunk']
                    cur['add_lines'].append(entry)
                    cur['add_lineno'].append(b_lineno)
                    cur['_has_changes'] = True
                b_lineno = b_lineno + 1
            else:
                a_lineno = a_lineno + 1
                b_lineno = b_lineno + 1
                if i_func and i_func in i_chunk and '_current_hunk' in i_chunk[i_func]:
                    cur = i_chunk[i_func]['_current_hunk']
                    if cur['_has_changes']:
                        cur['context_after'].append(_line.rstrip('\n'))
                    else:
                        cur['context_before'].append(_line.rstrip('\n'))

        if i_chunk and i_func and i_func in i_chunk and '_current_hunk' in i_chunk[i_func]:
            i_chunk[i_func]['hunks'].append(i_chunk[i_func].pop('_current_hunk'))

        if i_file:
            diff_chunks[i_file] = i_chunk
        return diff_chunks

    @property
    def diff_chunks(self):
        return self._diff_chunks
