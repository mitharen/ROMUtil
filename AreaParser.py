#!/usr/bin/env python
import sys

import ply.lex as lex
import ply.yacc as yacc

class Lexer():
    states = (
        ('string', 'exclusive'),
        ('optional', 'inclusive'),
    )
    tokens = (
        'AREA',
        'MOBILES',
        'OBJECTS',
        'ROOMS',
        'RESETS',
        'SHOPS',
        'SPECIALS',
        'END',
        'NULL',
        'EOL',
        'VNUM',
        'COMMENT',
        'NUMBER',
        'WORD',
        'QUOTED',
        'STRING',
        'S',
        'APPLY',
        'DOOR',
        'EXT',
        'FLAG',
        'REGEN',
    )

    t_AREA = r'\#AREA'
    t_MOBILES = r'\#MOBILES'
    t_OBJECTS = r'\#OBJECTS'
    t_ROOMS = r'\#ROOMS'
    t_RESETS = r'\#RESETS'
    t_SHOPS = r'\#SHOPS'
    t_SPECIALS = r'\#SPECIALS'
    t_END = r'\#\$'
    t_ignore = ' \t'

    def t_NULL(self, t):
        r'\#0'
        t.lexer.begin('INITIAL')
        return t
    def t_EOL(self, t):
        r'\n'
        return t
    def t_VNUM(self, t):
        r'\#\d+'
        t.value = int(t.value[1:])
        t.lexer.begin('INITIAL')
        return t
    def t_COMMENT(self, t):
        r'\*[^\n]*'
        return t
    def t_NUMBER(self, t):
        r'(?:\-|\+)?\d+'
        t.value = int(t.value)
        return t
    def t_WORD(self, t):
        r'\w+'
        return t
    def t_QUOTED(self, t):
        r'\'.*?\''
        t.value = t.value.strip('\'')
        return t
    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    t_string_ignore = ''
    def t_string_STRING(self, t):
        r'(?:[^~]|\.)*~'
        t.value = t.value[:-1].strip()
        t.lexer.begin('INITIAL')
        return t
    def t_string_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    t_optional_ignore = ''
    def t_optional_S(self, t):
        r'S'
        t.lexer.begin('INITIAL')
        return t
    def t_optional_APPLY(self, t):
        r'A'
        t.lexer.begin('INITIAL')
        return t
    def t_optional_DOOR(self, t):
        r'D'
        t.lexer.begin('INITIAL')
        return t
    def t_optional_EXT(self, t):
        r'E'
        t.lexer.begin('INITIAL')
        return t
    def t_optional_FLAG(self, t):
        r'F'
        t.lexer.begin('INITIAL')
        return t
    def t_optional_REGEN(self, t):
        r'H'
        t.lexer.begin('INITIAL')
        return t
    def t_optional_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def lex_file(self, file):
        lexer = lex.lex(module=self)
        with open(file, 'r') as f:
            lexer.input(f.read())
        while True:
            tok = lexer.token()
            if not tok: break
            print(tok)

class Parser():
    def p_file(self, p):
        'file : sections END EOL'
        p[0] = p[1]

    def p_sections(self, p):
        '''sections : section sections
                    | EOL sections
                    |
            objects : object objects
                    | NULL EOL
   object_optionals : object_optional object_optionals
                    |
            mobiles : mobile mobiles
                    | NULL EOL
      mob_optionals : mob_optional mob_optionals
                    |
              rooms : room rooms
                    | NULL EOL
     room_optionals : room_optional room_optionals
                    |
             resets : reset resets
                    | WORD EOL
              shops : shop shops
                    | NUMBER EOL
           specials : special specials
                    | WORD EOL'''
        if len(p) != 3 or p[2] == '\n':
            p[0] = None
            return
        p[0] = [p[1]]
        if p[2]: p[0].extend(p[2])
        if p[1] == '\n': p[0] = p[2]

    def p_section(self, p):
        '''section : AREA EOL area
                   | MOBILES EOL mobiles
                   | OBJECTS EOL objects
                   | ROOMS EOL rooms
                   | RESETS EOL comments resets
                   | SHOPS EOL shops
                   | SPECIALS EOL comments specials'''
        p[0] = (p[1], p[3] if len(p) != 5 else p[4])

    def p_area(self, p):
        'area : str STRING EOL str STRING EOL str STRING EOL NUMBER NUMBER EOL'
        p[0] = (p[2], p[5], p[8], (p[10], p[11]))

    def p_mobile(self, p):
        '''mobile : VNUM EOL str STRING EOL str STRING EOL str STRING EOL \
                    str STRING EOL \
                    str STRING EOL \
                    flags flags NUMBER NUMBER EOL \
                    NUMBER NUMBER hitndam hitndam hitndam WORD EOL \
                    NUMBER NUMBER NUMBER NUMBER EOL \
                    flags flags flags flags EOL \
                    WORD WORD WORD NUMBER EOL \
                    flags flags WORD flags optional EOL \
                    mob_optionals'''
#        print('Mob: %s %s %s'%(p[4], p[7], p[10]))
        p[0] = ()
    def p_mob_optional(self, p):
        '''mob_optional : flag_remove'''
        p[0] = p[3] if len(p) > 3 else None

    def p_object(self, p):
        '''object : VNUM EOL str STRING EOL str STRING EOL str STRING EOL str STRING EOL \
                    WORD flags flags EOL \
                    param param param param param EOL \
                    NUMBER NUMBER NUMBER flags optional EOL \
                    object_optionals'''
        p[0] = ()
#        print('Obj: %s %s %s'%(p[4], p[7], p[10]))
    def p_object_optional(self, p):
        '''object_optional : apply
                           | ext
                           | flag_add'''
        p[0] = p[1]

    def p_room(self, p):
        '''room : VNUM EOL str STRING EOL str STRING EOL \
                  NUMBER flags NUMBER optional EOL \
                  room_optionals S EOL'''
        p[0] = (p[1], p[4], p[14])
#        print('Room: %s'%(p[4]))
    def p_room_optional(self, p):
        '''room_optional : door
                         | ext
                         | regen'''
        p[0] = p[1]
    def p_door(self, p):
        '''door : DOOR NUMBER EOL str STRING EOL str STRING EOL \
                  NUMBER NUMBER NUMBER optional EOL'''
        p[0] = (p[2], p[12])
    def p_regen(self, p):
        'regen : REGEN NUMBER WORD NUMBER optional EOL'
        pass

    def p_reset(self, p):
        '''reset : WORD NUMBER NUMBER NUMBER NUMBER NUMBER comment
                 | WORD NUMBER NUMBER NUMBER NUMBER comment
                 | WORD NUMBER NUMBER NUMBER comment'''
        p[0] = list(v for v in p[1:])

    def p_shop(self, p):
        '''shop : NUMBER NUMBER NUMBER NUMBER NUMBER NUMBER \
                  NUMBER NUMBER NUMBER NUMBER comment'''
        p[0] = list(v for v in p[1:])

    def p_special(self, p):
        '''special : WORD NUMBER WORD comment'''
        p[0] = ()

    def p_param(self, p):
        '''param : WORD
                 | NUMBER
                 | QUOTED'''
        p[0] = p[1]
    def p_flags(self, p):
        '''flags : WORD
                 | NUMBER '''
        p[0] = p[1] if p[1] else None
    def p_hitndam(self, p):
        'hitndam : NUMBER WORD NUMBER'
        p[0] = (p[1], p[2], p[3])
    def p_flag_add(self, p):
        'flag_add : FLAG EOL WORD NUMBER NUMBER flags optional EOL'
        p[0] = (p[1], p[2])
    def p_flag_remove(self, p):
        'flag_remove : FLAG WORD flags optional EOL'
        p[0] = (p[1], p[2])
    def p_apply(self, p):
        'apply : APPLY EOL NUMBER NUMBER optional EOL'
        p[0] = (p[1], p[2])
    def p_ext(self, p):
        'ext : EXT EOL str STRING EOL str STRING optional EOL'
        #p[0] = (p[4], p[7])
        pass

    def p_str(self, p):
        'str :'
        p.lexer.begin('string')
    def p_optional(self, p):
        'optional :'
        p.lexer.begin('optional')
    def p_comment(self, p):
        '''comment : COMMENT EOL
                   | EOL'''
        p[0] = p[1] if len(p) == 3 else None
    def p_comments(self, p):
        '''comments : COMMENT comments
                    | '''
        p[0] = [p[1]]+p[2] if len(p) == 3 else []

    def p_error(self, p):
        if p:
            # get formatted representation of stack
            stack_state_str = ' '.join([symbol.type for symbol in self.parser.symstack][1:])

            print('Syntax error in input! Parser State:{} {} . {}'
                  .format(self.parser.state,
                          stack_state_str, p))
            sys.exit(1)
        else:
            print("Syntax error at EOF")

    def __init__(self):
        self.tokens = Lexer.tokens
        self.lexer = Lexer().build()
        self.parser = yacc.yacc(module=self)

    def parse(self, buffer):
        return self.parser.parse(buffer, lexer=self.lexer, debug=False)

def main():
    with open(file, 'r') as f:
        area = Parser().parse(sys.argv[1])

if __name__=='__main__':
    main()
