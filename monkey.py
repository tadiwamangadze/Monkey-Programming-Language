#Tadiwa Mangadze - Project

#you can either run the code line 
#            by line,
#               using terminal: python monkey.py
# or run the file mycode.mnky:
#               using terminal: python monkey.py mycode.mnky
# press enter to exit prompt

import sys
import os

class Token:
    ILLEGAL = 'ILLEGAL'
    EOF = 'EOF'
    IDENT = 'IDENT'
    INT = 'INT'
    ASSIGN = '='
    PLUS = '+'
    MINUS = '-'
    COMMA = ','
    SEMICOLON = ';'
    LPAREN = '('
    RPAREN = ')'
    LBRACE = '{'
    RBRACE = '}'
    FUNCTION = 'FUNCTION'
    LET = 'LET'
    STRING = 'STRING'
    COLON = ':'

    def __init__(self, type_='', literal=''):
        self.type = type_
        self.literal = literal

# 1. LEXER -------------------------------------------------------------------------

class TheLexer:
    KEYWORDS = {
                'fn': Token.FUNCTION,
                'let': Token.LET,
            }

    VALID_IDENTS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    VALID_NUMBERS = '0123456789'
    WHITESPACES = [' ', '\t', '\r', '\n']

    def __init__(self, input_='', position=0, read=0, ch=''):
        self.input = input_
        self.position = position
        self.read = read
        self.ch = ch

    def read_char(self):
        if self.read >= len(self.input):
            self.ch = ''
        else:
            self.ch = self.input[self.read]
        self.position = self.read
        self.read += 1

    def peek_char(self):
        if self.read >= len(self.input):
            return ''
        else:
            return self.input[self.read]

    def new_token(self, token, t, ch):
        token.type = t
        token.literal = ch
        return token

    def next_token(self):
        t = Token()

        self.skip_whitespace()

        if self.ch == '=':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                t = self.new_token(t, Token.EQ, ch + self.ch)
            else:
                t = self.new_token(t, Token.ASSIGN, self.ch)
        elif self.ch == '+':
            t = self.new_token(t, Token.PLUS, self.ch)
        elif self.ch == '-':
            t = self.new_token(t, Token.MINUS, self.ch)
        elif self.ch == ';':
            t = self.new_token(t, Token.SEMICOLON, self.ch)
        elif self.ch == '(':
            t = self.new_token(t, Token.LPAREN, self.ch)
        elif self.ch == ')':
            t = self.new_token(t, Token.RPAREN, self.ch)
        elif self.ch == ',':
            t = self.new_token(t, Token.COMMA, self.ch)
        elif self.ch == '+':
            t = self.new_token(t, Token.PLUS, self.ch)
        elif self.ch == '{':
            t = self.new_token(t, Token.LBRACE, self.ch)
        elif self.ch == '}':
            t = self.new_token(t, Token.RBRACE, self.ch)
        elif self.ch == '':
            t.literal = ''
            t.type = Token.EOF
        else:
            if self.is_letter(self.ch):
                t.literal = self.read_ident()
                t.type = self.lookup_ident(t.literal)
                return t
            elif self.is_digit(self.ch):
                t.literal = self.read_number()
                t.type = Token.INT
                return t
            else:
                t = self.new_token(t, Token.ILLEGAL, self.ch)
        self.read_char()
        return t

    def read_ident(self):
        pos = self.position
        while True:
            if not self.ch:
                break
            test = self.is_letter(self.ch)
            if not test:
                break
            self.read_char()
        ret = self.input[pos:self.position]
        return ret

    def read_number(self):
        pos = self.position
        while True:
            if not self.ch:
                break
            test = self.is_digit(self.ch)
            if not test:
                break
            self.read_char()
        ret = self.input[pos:self.position]
        return ret

    def read_string(self):
        pos = self.position + 1
        while True:
            self.read_char()
            if self.ch == '"' or self.ch == '':
                break
 
        ret = self.input[pos:self.position]
        return ret

    def lookup_ident(self, s):
        ret = TheLexer.KEYWORDS.get(s)
        if ret:
            return ret
        return Token.IDENT

    def is_letter(self, ch):
        return ch in TheLexer.VALID_IDENTS

    def is_digit(self, ch):
        return ch in TheLexer.VALID_NUMBERS
        
    def skip_whitespace(self):
        while (self.ch in TheLexer.WHITESPACES):
            self.read_char()

    def new(s):
        l = TheLexer()
        l.input = s
        l.read_char()
        return l
    new = staticmethod(new)


class Node:
    def __init__(self):
        pass

    def token_literal(self):
        return ''

    def string(self):
        return ''


class Statement(Node):
    def __init__(self):
        pass

    def statement_node(self):
        pass


class Expression(Node):
    def __init__(self):
        pass

    def expression_node(self):
        pass


class Identifier(Expression):
    def __init__(self, value=''):
        self.token = Token()
        self.value = value

    def token_literal(self):
        return self.token.literal
    
    def string(self):
        return self.value


class LetStatement(Statement):
    def __init__(self):
       self.token = Token()
       self.name = Identifier()
       self.value = Expression()

    def token_literal(self):
        return self.token.literal

    def string(self):
        ret = self.token_literal() + ' '
        ret += self.name.string()
        ret += ' = '
        
        if self.value:
           ret += self.value.string()
     
        ret += ';'
        return ret
        

class ReturnStatement(Statement):
    def __init__(self):
       self.token = Token()
       self.return_value = Expression()

    def token_literal(self):
        return self.token.literal

    def string(self):
        ret = self.token_literal() + ' '
        if self.return_value:
            ret += self.return_value.string()
        
        ret += ';'
        return ret


class ExpressionStatement(Statement):
    def __init__(self):
       self.token = Token()
       self.expression = Expression()

    def token_literal(self):
        return self.token.literal

    def string(self):
        if self.expression:
            return self.expression.string()
        
        return ''


class BlockStatement(Statement):
    def __init__(self):
       self.token = Token()
       self.statements = []

    def is_empty(self):
        return len(self.statements) == 0

    def token_literal(self):
        return self.token.literal

    def string(self):
        ret = '%s{%s' %(line_separation, line_separation)
        
        for s in self.statements:
            ret += '%s;%s' %(s.string(), line_separation)
        
        ret += '}%s' %(line_separation)
        return ret


class IntegerLiteral(Expression):
    def __init__(self, value=None):
       self.token = Token()
       self.value = value

    def token_literal(self):
        return self.token.literal

    def string(self):
        return self.token.literal


class StringLiteral(Expression):
    def __init__(self, value=''):
       self.token = Token()
       self.value = value

    def token_literal(self):
        return self.token.literal

    def string(self):
        return self.token.literal


class FunctionLiteral(Expression):
    def __init__(self):
       self.token = Token()
       self.parameters = []
       self.body = BlockStatement()

    def token_literal(self):
        return self.token.literal

    def string(self):
        params = []
        for p in self.parameters:
            params.append(p.string())
        
        ret = self.token_literal()
        ret += '('
        ret += ', '.join(params)
        ret += ')'
        ret += self.body.string()
        
        return ret


class CallExpression(Expression):
    def __init__(self):
       self.token = Token()
       self.function = Expression()
       self.arguments = []

    def token_literal(self):
        return self.token.literal

    def string(self):
        args = []
        for a in self.arguments:
            args.append(a.string())
        
        ret = self.function.string()
        ret += '('
        ret += ', '.join(args)
        ret += ')'
        
        return ret


class PrefixExpression(Expression):
    def __init__(self):
       self.token = Token()
       self.operator = ''
       self.right = Expression()

    def token_literal(self):
        return self.token.literal

    def string(self):
        ret = '('
        ret += self.operator
        ret += self.right.string()
        ret += ')'
        
        return ret


class InfixExpression(Expression):
    def __init__(self):
       self.token = Token()
       self.left = Expression()
       self.operator = ''
       self.right = Expression()

    def token_literal(self):
        return self.token.literal

    def string(self):
        ret = '('
        ret += self.left.string()
        ret += ' ' + self.operator + ' '
        ret += self.right.string()
        ret += ')'

        return ret


class IndexExpression(Expression):
    def __init__(self):
       self.token = Token()
       self.left = Expression()
       self.index = Expression()

    def token_literal(self):
        return self.token.literal

    def string(self):
        ret = '('
        ret += self.left.string()
        ret += '['
        ret += self.index.string()
        ret += '])'
        
        return ret


class HashLiteral(Expression):
    def __init__(self):
       self.token = Token()
       self.pairs = {}

    def token_literal(self):
        return self.token.literal

    def string(self):
        pairs = []
        for k in self.pairs.keys():
            v = self.pairs.get(k)
            pairs.append('%s:%s' %(k.string(), v.string()))
        
        ret = '{'
        ret += ', '.join(pairs)
        ret += '}'
        
        return ret

# 2. PARSER -------------------------------------------------------------------------

class TheParser:
    LOWEST = 1
    EQUALS = 2
    SUM = 4
    PRODUCT = 5
    PREFIX = 6
    CALL = 7
    INDEX = 8

    PRECEDENCES = {
        Token.LPAREN: CALL,
        Token.PLUS: SUM,
        Token.MINUS: SUM,
    }

    def __init__(self):
        self.lexer = None
        self.cur_token = None
        self.peek_token = None
        self.errors = []
        self.prefix_parse_fns = {}
        self.infix_parse_fns = {}
        
        self.register_prefix(Token.IDENT, self.parse_identifier)
        self.register_prefix(Token.INT, self.parse_integer_literal)
        self.register_prefix(Token.MINUS, self.parse_prefix_expression)
        self.register_prefix(Token.LPAREN, self.parse_grouped_expression)
        self.register_prefix(Token.FUNCTION, self.parse_function_literal)
        self.register_prefix(Token.STRING, self.parse_string_literal)
        
        self.register_infix(Token.PLUS, self.parse_infix_expression)
        self.register_infix(Token.MINUS, self.parse_infix_expression)
        self.register_infix(Token.LPAREN, self.parse_call_expression)

    def next_token(self):
        self.cur_token = self.peek_token
        self.peek_token = self.lexer.next_token()

    def parse_program(self):
        program = monkeyProgram()
        
        while self.cur_token.type != Token.EOF:
            s = self.parse_statement()
            if s:
                program.statements.append(s)
            self.next_token()

        return program

    def parse_statement(self):
        if self.cur_token.type == Token.LET:
            return self.parse_let_statement()
        else:
            return self.parse_expression_statement()
        return None

    def parse_let_statement(self):
        s = LetStatement()
        s.token = self.cur_token
        if not self.expect_peek(Token.IDENT):
            return None
        
        s.name = Identifier()
        s.name.token = self.cur_token
        s.name.value = self.cur_token.literal
        if not self.expect_peek(Token.ASSIGN):
            return None
        
        self.next_token()
        s.value = self.parse_expression(self.LOWEST)
        if self.peek_token_is(Token.SEMICOLON):
            self.next_token()
        
        return s

    def parse_return_statement(self):
        s = ReturnStatement()
        s.token = self.cur_token

        self.next_token()

        s.return_value = self.parse_expression(self.LOWEST)
        if self.peek_token_is(Token.SEMICOLON):
            self.next_token()
        
        return s

    def parse_expression_statement(self):
        s = ExpressionStatement()
        s.token = self.cur_token
        s.expression = self.parse_expression(self.LOWEST)
        
        if self.peek_token_is(Token.SEMICOLON):
            self.next_token()
        
        return s

    def parse_block_statement(self):
        block = BlockStatement()
        block.token = self.cur_token
        
        self.next_token()
        while not self.cur_token_is(Token.RBRACE) and \
            not self.cur_token_is(Token.EOF):
            s = self.parse_statement()
            if s is not None:
                block.statements.append(s)
            self.next_token()
        
        return block

    def parse_expression(self, precedence):
        prefix = self.prefix_parse_fns.get(self.cur_token.type)
        if prefix is None:
            self.no_prefix_parse_fn_error(self.cur_token.type)
            return None
        left_exp = prefix()
        
        while not self.peek_token_is(Token.SEMICOLON) and \
            precedence < self.peek_precedence():
            infix = self.infix_parse_fns.get(self.peek_token.type)
            if infix is None:
                return left_exp
            
            self.next_token()
            left_exp = infix(left_exp)
        
        return left_exp

    def parse_identifier(self):
        ret = Identifier()
        ret.token = self.cur_token
        ret.value = self.cur_token.literal
        return ret

    def parse_integer_literal(self):
        lit = IntegerLiteral()
        lit.token = self.cur_token
        try:
            test = int(self.cur_token.literal)
        except:
            msg = 'could not parse %s as integer' %(self.cur_token.literal)
            self.errors.append(msg)
            return None
        
        lit.value = int(self.cur_token.literal)
        return lit

    def parse_string_literal(self):
        lit = StringLiteral()
        lit.token = self.cur_token
        lit.value = self.cur_token.literal
        return lit
    
    
    def parse_hash_literal(self):
        h = HashLiteral()
        h.token = self.cur_token
        
        while not self.peek_token_is(Token.RBRACE):
            self.next_token()
            key = self.parse_expression(self.LOWEST)
            
            if not self.expect_peek(Token.COLON):
                return None
            
            self.next_token()
            value = self.parse_expression(self.LOWEST)
            
            h.pairs[key] = value
            
            if not self.peek_token_is(Token.RBRACE) and \
                not self.expect_peek(Token.COMMA):
                return None
        
        if not self.expect_peek(Token.RBRACE):
            return None
        
        return h
    
    def parse_prefix_expression(self):
        e = PrefixExpression()
        e.token = self.cur_token
        e.operator = self.cur_token.literal
        
        self.next_token()
        e.right = self.parse_expression(self.PREFIX)
        
        return e

    def parse_infix_expression(self, left):
        e = InfixExpression()
        e.token = self.cur_token
        e.operator = self.cur_token.literal
        e.left = left
        
        precedence = self.cur_precedence()
        self.next_token()
        e.right = self.parse_expression(precedence)
        
        return e

    def parse_grouped_expression(self):
        self.next_token()
        e = self.parse_expression(self.LOWEST)
        
        if not self.expect_peek(Token.RPAREN):
            return None
        
        return e

    def parse_function_literal(self):
        lit = FunctionLiteral()
        lit.token = self.cur_token
        
        if not self.expect_peek(Token.LPAREN):
            return None
        
        lit.parameters = self.parse_function_parameters()
        
        if not self.expect_peek(Token.LBRACE):
            return None
        
        lit.body = self.parse_block_statement()
        
        return lit

    def parse_function_parameters(self):
        identifiers = []
        
        if self.peek_token_is(Token.RPAREN):
            self.next_token()
            return identifiers
        
        self.next_token()
        ident = Identifier()
        ident.token = self.cur_token
        ident.value = self.cur_token.literal
        identifiers.append(ident)
        
        while self.peek_token_is(Token.COMMA):
            self.next_token()
            self.next_token()
            ident = Identifier()
            ident.token = self.cur_token
            ident.value = self.cur_token.literal
            identifiers.append(ident)
        
        if not self.expect_peek(Token.RPAREN):
            return None
        
        return identifiers

    def parse_call_expression(self, function):
        exp = CallExpression()
        exp.token = self.cur_token
        exp.function = function
        exp.arguments = self.parse_expression_list(Token.RPAREN)
        return exp

    def parse_expression_list(self, end):
        ret = []
        
        if self.peek_token_is(end):
            self.next_token()
            return ret
        
        self.next_token()
        ret.append(self.parse_expression(self.LOWEST))
        
        while self.peek_token_is(Token.COMMA):
            self.next_token()
            self.next_token()
            ret.append(self.parse_expression(self.LOWEST))
    
        if not self.expect_peek(end):
            return None
        
        return ret
    
    def parse_index_expression(self, left):
        exp = IndexExpression()
        exp.token = self.cur_token
        exp.left = left
        
        self.next_token()
        exp.index = self.parse_expression(self.LOWEST)
        
        if not self.expect_peek(Token.RBRACKET):
            return None
        
        return exp

    def cur_token_is(self, t):
        return self.cur_token.type == t

    def peek_token_is(self, t):
        return self.peek_token.type == t

    def expect_peek(self, t):
        if self.peek_token_is(t):
            self.next_token()
            return True
        else:
            self.peek_error(t)
            return False

    def peek_error(self, t):
        m = 'expected next token to be %s, got %s instead' %(
                t, self.peek_token.type
            )
        self.errors.append(m)

    def register_prefix(self, token_type, fn):
        self.prefix_parse_fns[token_type] = fn

    def register_infix(self, token_type, fn):
        self.infix_parse_fns[token_type] = fn

    def no_prefix_parse_fn_error(self, token_type):
        m = 'no prefix parse function for %s found' %(token_type)
        self.errors.append(m)

    def peek_precedence(self):
        p = self.PRECEDENCES.get(self.peek_token.type)
        if p:
            return p
        
        return self.LOWEST

    def cur_precedence(self):
        p = self.PRECEDENCES.get(self.cur_token.type)
        if p:
            return p
        
        return self.LOWEST

    def new(l):
        p = TheParser()
        p.lexer = l
        p.next_token()
        p.next_token()
        return p
    new = staticmethod(new)


class monkeyProgram(Node):
    def __init__(self):
        self.statements = []

    def token_literal(self):
        if len(self.statements) > 0:
            return self.statements[0].token_literal()
        else:
            return ''
    
    def string(self):
        ret = ''
        for s in self.statements:
            ret += s.string()
        return ret


class Hashable:
    def hash_key(self):
        pass


class Object:
    INTEGER_OBJ = 'INTEGER'
    NULL_OBJ = 'NULL'
    RETURN_VALUE_OBJ = 'RETURN_VALUE'
    ERROR_OBJ = 'ERROR'
    FUNCTION_OBJ = 'FUNCTION'
    STRING_OBJ = 'STRING'

    def __init__(self, value=None, type_=''):
        self.value = value
        self.object_type = type_

    def type(self):
        return self.object_type

    def inspect(self):
        return ''

    def inspect_value(self):
        return self.inspect()


class ObjectInteger(Object, Hashable):
    def type(self):
        return self.INTEGER_OBJ

    def inspect(self):
        return '%s' %(self.value)

    def hash_key(self):
        o = HashKey(type_=self.type(), value=self.value)
        return o


class ObjectString(Object, Hashable):
    def type(self):
        return self.STRING_OBJ

    def inspect(self):
        return '"%s"' %(self.value)

    def hash_key(self):
        o = HashKey(type_=self.type(), value=hash(self.value))
        return o

    def inspect_value(self):
        return self.value

    def inspect(self):
        ret = '%s' %(self.value)
        ret = ret.lower()
        return ret
    
    def hash_key(self):
        o = HashKey(type_=self.type())
        if self.value:
            o.value = 1
        else:
            o.value = 0
        return o


class ObjectNull(Object):
    def type(self):
        return self.NULL_OBJ

    def inspect(self):
        return 'null'


class ObjectReturnValue(Object):
    def type(self):
        return self.RETURN_VALUE_OBJ

    def inspect(self):
        return self.value.inspect()


class ObjectError(Object):
    def __init__(self, value=None, message=''):
        self.message = message
        self.value = value

    def type(self):
        return self.ERROR_OBJ

    def inspect(self):
        return 'ERROR: %s' %(self.message)


class ObjectFunction(Object):
    def __init__(self):
        self.parameters = []
        self.body = BlockStatement()
        self.env = Environment.new()

    def type(self):
        return self.FUNCTION_OBJ

    def inspect(self):
        params = []
        for p in self.parameters:
            params.append(p.string())
        
        ret = 'fn'
        ret += '('
        ret += ', '.join(params)
        ret += ')'
        ret += self.body.string()
        
        return ret


class Environment:
    def __init__(self, outer=None):
        self.store = {}
        self.outer = outer

    def get(self, name):
        obj = self.store.get(name)
        if obj is None and self.outer is not None:
            obj = self.outer.get(name)
        return obj

    def set(self, name, value):
        self.store[name] = value
        return value

    def debug(self):
        for k in self.store.keys():
            v = self.store.get(k)
            if v is not None:
                MyMonkeyClass.output('%s: %s' %(
                        k,
                        v.inspect(),
                    )
                )

    def new():
        e = Environment()
        return e
    new = staticmethod(new)

    def new_enclosed(outer):
        e = Environment()
        e.outer = outer
        return e
    new_enclosed = staticmethod(new_enclosed)

    def from_dictionary(d):
        e = Environment()
        if not isinstance(d, dict):
            return e
        
        for k in d.keys():
            v = d.get(k)
            key = None
            value = None
            if type(k) == type(''):
                key = k
            else:
                key = str(k)
            
            if type(v) == type(''):
                value = ObjectString(value=v)
            elif type(v) == type(1):
                value = ObjectInteger(value=v)
            else:
                value = ObjectString(value=str(v))
            
            if key is not None and value is not None:
                e.set(key, value)
        return e
    from_dictionary = staticmethod(from_dictionary)


# 3. EVALUATOR -------------------------------------------------------------------------

class TheEvaluator:
    NULL = ObjectNull()

    def __init__(self):
        self.output = sys.stdout

    def eval(self, node, env):
        if isinstance(node, monkeyProgram):
            return self.eval_program(node, env)
        elif isinstance(node, ExpressionStatement):
            return self.eval(node.expression, env)
        elif isinstance(node, IntegerLiteral):
            o = ObjectInteger()
            o.value = node.value
            return o
        elif isinstance(node, PrefixExpression):
            right = self.eval(node.right, env)
            if self.is_error(right):
                return right
            
            return self.eval_prefix_expression(node.operator, right)
        elif isinstance(node, InfixExpression):
            left = self.eval(node.left, env) 
            if self.is_error(left):
                return left
            
            right = self.eval(node.right, env)
            if self.is_error(right):
                return right
            
            return self.eval_infix_expression(node.operator, left, right)
        elif isinstance(node, BlockStatement):
            return self.eval_block_statement(node, env)
        elif isinstance(node, ReturnStatement):
            val = self.eval(node.return_value, env)
            if self.is_error(val):
                return val
            
            o = ObjectReturnValue()
            o.value = val
            return o
        elif isinstance(node, LetStatement):
            val = self.eval(node.value, env)
            if self.is_error(val):
                return val
            
            env.set(node.name.value, val)
        elif isinstance(node, Identifier):
            return self.eval_identifier(node, env)
        elif isinstance(node, FunctionLiteral):
            params = node.parameters
            body = node.body
            
            o = ObjectFunction()
            o.parameters = params
            o.body = body
            o.env = env
            return o
        elif isinstance(node, CallExpression):
            function = self.eval(node.function, env)
            if self.is_error(function):
                return function
            
            args = self.eval_expressions(node.arguments, env)
            if len(args) == 1 and self.is_error(args[0]):
                return args[0]
            
            return self.apply_function(function, args)
        elif isinstance(node, StringLiteral):
            o = ObjectString()
            o.value = node.value
            return o
        elif isinstance(node, IndexExpression):
            left = self.eval(node.left, env)
            if self.is_error(left):
                return left
            
            index = self.eval(node.index, env)
            if self.is_error(index):
                return index
            
            return self.eval_index_expression(left, index)
        elif isinstance(node, HashLiteral):
            return self.eval_hash_literal(node, env)
        
        return None

    def eval_program(self, program, env):
        ret = Object()
        for s in program.statements:
            ret = self.eval(s, env)
            
            if isinstance(ret, ObjectReturnValue):
                return ret.value
            elif isinstance(ret, ObjectError):
                return ret
        
        return ret

    def eval_block_statement(self, block, env):
        ret = Object()
        for s in block.statements:
            ret = self.eval(s, env)
            
            if ret:
                rt = ret.type()
                if rt == Object.RETURN_VALUE_OBJ or \
                    rt == Object.ERROR_OBJ:
                    return ret
        
        return ret

    def eval_prefix_expression(self, operator, right):
        if operator == '-':
            return self.eval_minus_prefix_operator_expression(right)
        return self.new_error('unknown operator: %s%s' %(
            operator, right.type()))

    def eval_infix_expression(self, operator, left, right):
        if left.type() == Object.INTEGER_OBJ and \
            right.type() == Object.INTEGER_OBJ:
            return self.eval_integer_infix_expression(operator, left, right)
        elif left.type() == Object.STRING_OBJ and \
            right.type() == Object.STRING_OBJ:
            return self.eval_string_infix_expression(operator, left, right)

        return self.new_error('unknown operator: %s %s %s' %(
            left.type(), operator, right.type()))

    def eval_integer_infix_expression(self, operator, left, right):
        left_val = left.value
        right_val = right.value
        
        o = ObjectInteger()
        if operator == '+':
            o.value = left_val + right_val
            return o
        elif operator == '-':
            o.value = left_val - right_val
            return o
        return self.new_error('unknown operator: %s %s %s' %(
            left.type(), operator, right.type()))

    def eval_string_infix_expression(self, operator, left, right):
        left_val = left.value
        right_val = right.value
        
        o = ObjectString()
        if operator != '+':
            return self.new_error('unknown operator: %s %s %s' %(
                left.type(), operator, right.type()))
        
        o.value = left_val + right_val
        return o

    def eval_minus_prefix_operator_expression(self, right):
        if right.type() != Object.INTEGER_OBJ:
            return self.new_error('unknown operator: -%s' %(right.type()))
        
        val = right.value
        ret = ObjectInteger()
        ret.value = -val
        
        return ret

    def eval_identifier(self, node, env):
        val = env.get(node.value)
        if val:
            return val
        
        return self.new_error('identifier not found: %s' %(node.value))
    
    def eval_expressions(self, exp, env):
        result = []
        
        for e in exp:
            evaluated = self.eval(e, env)
            if self.is_error(evaluated):
                result.append(evaluated)
                return result
            result.append(evaluated)
        
        return result

    def eval_index_expression(self, left, index):
        if left.type() == Object.ARRAY_OBJ and \
            index.type() == Object.INTEGER_OBJ:
            return self.eval_array_index_expression(left, index)
        elif left.type() == Object.HASH_OBJ:
            return self.eval_hash_index_expression(left, index)
        return self.new_error('index operator not supported: %s' %(
            left.type()))

    def apply_function(self, fn, args):
        if isinstance(fn, ObjectFunction):
            extended_env = self.extend_function_env(fn, args)
            evaluated = self.eval(fn.body, extended_env)
            return self.unwrap_return_value(evaluated)
        
        return self.new_error('not a function: %s' %(fn.type()))

    def extend_function_env(self, fn, args):
        env = Environment.new_enclosed(fn.env)
        for p in range(len(fn.parameters)):
            param = fn.parameters[p] 
            env.set(param.value, args[p])
        
        return env

    def unwrap_return_value(self, obj):
        if isinstance(obj, ObjectReturnValue):
            return obj.value
        
        return obj

    def is_truthy(self, obj):
        if obj == self.NULL:
            return False
        elif obj == self.TRUE:
            return True
        elif obj == self.FALSE:
            return False
        else:
            return True

    def new_error(self, message):
        ret = ObjectError()
        ret.message = message
        return ret
   
    def is_error(self, obj):
        if obj:
            return obj.type() == Object.ERROR_OBJ
        
        return False

    def new():
        e = TheEvaluator()
        return e
    new = staticmethod(new)

line_separation = os.linesep
class MyMonkeyClass:
    prompt = 'user-monkeylang % '

    def input(s):
        try:
            return raw_input(s)
        except:
            return input(s)
    input = staticmethod(input)

    def output(s, f=sys.stdout):
        try:
            f.write('Result => %s%s' %(s, line_separation))
        except:
            pass
    output = staticmethod(output)

    def lexer():

        while True:
            inp = MyMonkeyClass.input(MyMonkeyClass.prompt).strip()
            if not inp:
                break
            l = TheLexer.new(inp)
            while True:
                t = l.next_token()
                if t.type == Token.EOF:
                    break
                MyMonkeyClass.output(
                    'Type: %s, Literal: %s' %(t.type, t.literal))
    lexer = staticmethod(lexer)

    def parser():

        while True:
            inp = MyMonkeyClass.input(MyMonkeyClass.prompt).strip()
            if not inp:
                break
            l = TheLexer.new(inp)
            p = TheParser.new(l)
            program = p.parse_program()
            
            if p.errors:
                MyMonkeyClass.print_parse_errors(p.errors)
                continue
            
            MyMonkeyClass.output(program.string())
    parser = staticmethod(parser)

    def print_parse_errors(e, output=sys.stdout):
        for i in e:
            MyMonkeyClass.output('PARSER ERROR: %s' %(i), output)
    print_parse_errors = staticmethod(print_parse_errors)

    def evaluator():

        env = Environment.new()
        while True:
            inp = MyMonkeyClass.input(MyMonkeyClass.prompt).strip()
            if not inp:
                break
            l = TheLexer.new(inp)
            p = TheParser.new(l)
            program = p.parse_program()
            
            if p.errors:
                MyMonkeyClass.print_parse_errors(p.errors)
                continue
            
            evaluator = TheEvaluator.new()
            evaluated = evaluator.eval(program, env)
            if evaluated:
                MyMonkeyClass.output(evaluated.inspect())
    evaluator = staticmethod(evaluator)
    def evaluator_string(s, environ=None, output=sys.stdout):
        if environ is None or not isinstance(environ, Environment):
            env = Environment.new()
        else:
            env = environ
        l = TheLexer.new(s)
        p = TheParser.new(l)
        program = p.parse_program()

        if p.errors:
            MyMonkeyClass.print_parse_errors(p.errors, output)
            return

        evaluator = TheEvaluator.new()
        evaluator.output = output
        evaluated = evaluator.eval(program, env)
        if evaluated:
            MyMonkeyClass.output(evaluated.inspect(), output)
    evaluator_string = staticmethod(evaluator_string)


    def main(argv):
        if len(argv) < 2:
            MyMonkeyClass.evaluator()
        else:
            t = argv[1]
            s = t
            if os.path.exists(t):
                try:
                    s = open(t).read()
                except:
                    pass

            if s:
                MyMonkeyClass.evaluator_string(s)
    main = staticmethod(main)


if __name__ == '__main__':
    MyMonkeyClass.main(sys.argv)


