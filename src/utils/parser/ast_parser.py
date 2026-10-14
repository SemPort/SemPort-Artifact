import re
from tree_sitter import Language, Parser, Tree
import tree_sitter_c as tsc
# from enum import Enum

# class ASTNodeType(Enum):
#     TRANSLATION_UNIT = 'translation_unit'

#     PREPROC_INCLUDE = 'preproc_include'
#     PREPROC_DEF = 'preproc_def'
#     PREPROC_FUNCTION_DEF = 'preproc_function_def'
#     DECLARATION = 'declaration'
#     FUNCTION_DEFINITION = 'function_definition'

#     EXPRESSION_STATEMENT = 'expression_statement'

#     STORAGE_CLASS_SPECIFIER = 'storage_class_specifier'
#     PRIMITIVE_TYPE = 'primitive_type'
#     POINTER_DECLARATOR='pointer_declarator'
#     FUNCTION_DECLARATOR = 'function_declarator'
#     COMPOUND_STATEMENT = 'compound_statement'

#     IDENTIFIER = 'identifier'
#     TYPE_IDENTIFIER = 'type_identifier'
#     STRUCT_SPECIFIER='struct_specifier'
#     PARAMETER_LIST = 'parameter_list'

#     COMMENT = 'comment'

#     IF_STATEMENT = 'if_statement'
#     WHILE_STATEMENT = 'while_statement'
#     FOR_STATEMENT = 'for_statement'
#     SWITCH_STATEMENT = 'switch_statement'
#     GOTO_STATEMENT = 'goto_statement'
#     RETURN_STATEMENT = 'return_statement'

#     UNDEFINED = 'undefined'

#     @classmethod
#     def _missing_(cls, value):
#         return cls.UNDEFINED

class ASTParser:
    _language = Language(tsc.language())
    _parser = Parser(_language)

    @staticmethod
    def parse_file(filepath: str) -> Tree:
        with open(filepath, 'r', errors='ignore') as f:
            code = f.read()
        tree = ASTParser._parser.parse(bytes(code, 'utf8'))
        if tree.root_node.has_error:
            best, best_fn = tree, len([c for c in tree.root_node.children
                                       if c.type == 'function_definition'])
            clean1 = re.sub(r'__attribute__\s*\(\([^)]*\)\)', '', code)
            clean2 = re.sub(r'\b__[A-Za-z_]\w*\b', '', clean1)
            for cand in (clean1, clean2):
                if cand == code:
                    continue
                t = ASTParser._parser.parse(bytes(cand, 'utf8'))
                n = len([c for c in t.root_node.children
                         if c.type == 'function_definition'])
                if n > best_fn:
                    best, best_fn = t, n
            return best
        return tree

    @staticmethod
    def parse_code(code: str) -> Tree:
        return ASTParser._parser.parse(bytes(code, 'utf8'))

def declarator_identifier(declarator_node) -> str:
    if declarator_node is None:
        return ''
    if declarator_node.type == 'identifier':
        try:
            return declarator_node.text.decode('utf-8')
        except Exception:
            return ''
    for field in ('declarator', 'name'):
        c = declarator_node.child_by_field_name(field)
        if c is not None:
            r = declarator_identifier(c)
            if r:
                return r
    for c in declarator_node.children:
        if c.type == 'identifier':
            try:
                return c.text.decode('utf-8')
            except Exception:
                pass
    return ''

_PREPROC_WRAPPER_TYPES = {'preproc_ifdef', 'preproc_if', 'preproc_elif',
                          'preproc_else', 'preproc_ifndef'}

def iter_function_definitions(root_node):
    stack = list(root_node.children)
    while stack:
        node = stack.pop()
        if node.type == 'function_definition':
            yield node
        elif node.type in _PREPROC_WRAPPER_TYPES:
            stack.extend(reversed(node.children))
