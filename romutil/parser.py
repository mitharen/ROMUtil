import logging
import re
import sys
from typing import Any
import ply.lex as lex
import ply.yacc as yacc
from romutil.models import (
    AreaData,
    AreaHeader,
    ExitDef,
    ExtraDescr,
    HelpDef,
    MobileDef,
    ObjectDef,
    ResetDef,
    RoomDef,
    ShopDef,
    SocialDef,
    SpecialDef,
)

log = logging.getLogger('AreaParser')


def eval_flags(val: Any) -> Any:
    """Evaluate raw flag values into integer bitmasks or normalized strings.

    Supports piped bitmasks (e.g. '4|8|1024' -> 1036), integer literals,
    and letter/alphanumeric flag strings.
    """
    if isinstance(val, int):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return 0
    if '|' in s:
        parts = s.split('|')
        all_numeric = True
        acc = 0
        for part in parts:
            part_s = part.strip()
            if part_s.isdigit() or (part_s.startswith(('-', '+')) and part_s[1:].isdigit()):
                acc |= int(part_s)
            else:
                all_numeric = False
                break
        if all_numeric:
            return acc
        return s
    if s.isdigit() or (s.startswith(('-', '+')) and s[1:].isdigit()):
        return int(s)
    return s


def normalize_dialect_buffer(buffer: str) -> str:
    """Normalize dialect variations across Merc, Envy, Diku, and CircleMUD.

    Handles:
    - Envy #AREADATA ... End header converted into standard #AREA format.
    - Merc inline or single-line #AREA headers converted to 4-line standard format.
    - Missing #AREA header synthesized with default area metadata.
    - Piped bitmask flags evaluated (4|8|1024 -> 1036, A|B -> AB).
    - Strips unsupported dialect-specific sections (#GAMES, #CLANS, #ECONOMY, #OLC).
    - Ensures CircleMUD / headerless room definitions have #ROOMS section marker.
    - Normalizes CircleMUD / Diku EOF delimiters ($~, $) to standard #$.
    - Ensures room sections terminate with #0 before EOF.
    """
    if not buffer or not buffer.strip():
        return buffer

    # 1. Normalize piped bitmask flags
    def replace_pipe_flags(match):
        parts = match.group(0).split('|')
        val = 0
        for p in parts:
            val |= int(p)
        return str(val)

    buffer = re.sub(r'\b\d+(?:\|\d+)+\b', replace_pipe_flags, buffer)
    buffer = re.sub(r'(?<=[A-Za-z])\|(?=[A-Za-z])', '', buffer)

    # 2. Normalize Envy #AREADATA ... End
    def replace_areadata(match):
        body = match.group(1)
        data = {}
        for line in body.splitlines():
            line = line.strip()
            if not line or line == 'End':
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                k = parts[0]
                v = parts[1].rstrip('~').strip()
                data[k] = v
                if k == 'VNUMs':
                    nums = [int(n) for n in v.split() if n.lstrip('-+').isdigit()]
                    if len(nums) >= 2:
                        data['vnum_min'] = nums[0]
                        data['vnum_max'] = nums[1]
        name = data.get('Name', 'Area')
        author = data.get('Author', data.get('Builders', 'Unknown'))
        v_min = data.get('vnum_min', 0)
        v_max = data.get('vnum_max', 0)
        file_name = data.get('FileName', name.lower().replace(' ', '_') + ".are")
        return f"#AREA\n{file_name}~\n{name}~\n{author}~\n{v_min} {v_max}\n"

    buffer = re.sub(r'#AREADATA\s*\n(.*?)\nEnd\b', replace_areadata, buffer, flags=re.DOTALL)

    # 3. Normalize Merc single-line or inline #AREA headers
    def replace_merc_area_inline(match):
        raw = match.group(1).strip().rstrip('~').strip()
        m = re.match(r'\{[^}]*\}\s*(?:(\w+)\s+)?(.*)', raw)
        if m:
            builder = m.group(1) or "Unknown"
            name = m.group(2).strip() or raw
        else:
            builder = "Unknown"
            name = raw
        file_name = name.lower().replace(' ', '_') + ".are"
        return f"#AREA\n{file_name}~\n{name}~\n{builder}~\n0 0\n"

    buffer = re.sub(r'#AREA[ \t]+([^\n~]*~)', replace_merc_area_inline, buffer)

    def replace_merc_area_single_line(match):
        raw = match.group(1).strip().rstrip('~').strip()
        m = re.match(r'\{[^}]*\}\s*(?:(\w+)\s+)?(.*)', raw)
        if m:
            builder = m.group(1) or "Unknown"
            name = m.group(2).strip() or raw
        else:
            builder = "Unknown"
            name = raw
        file_name = name.lower().replace(' ', '_') + ".are"
        return f"#AREA\n{file_name}~\n{name}~\n{builder}~\n0 0\n"

    buffer = re.sub(r'#AREA\s*\n([^\n~]+~)\s*\n(?=#)', replace_merc_area_single_line, buffer)

    # 4. Strip dialect-specific non-standard sections
    buffer = re.sub(r'#(?:GAMES|CLANS|ECONOMY|OLC|PRACTICERS|RESETCONT)\b.*?(?=\n#|\Z)', '', buffer, flags=re.DOTALL)

    # 5. If no #ROOMS or #ROOMDATA header but room VNUMs exist in standalone room files, prepend #ROOMS
    if not re.search(r'#(?:ROOMS|ROOMDATA)\b', buffer) and not re.search(r'#(?:HELPS|SOCIALS|MOBILES|OBJECTS)\b', buffer):
        if re.search(r'^\#[1-9]\d*', buffer, re.MULTILINE):
            buffer = re.sub(r'(^\#[1-9]\d*)', r'#ROOMS\n\1', buffer, count=1, flags=re.MULTILINE)

    # 7. Normalize CircleMUD / Diku EOF delimiters ($~ or trailing $)
    buffer = re.sub(r'\n\$\~[ \t]*\n?$', '\n#$\n', buffer)
    buffer = re.sub(r'\n\$[ \t]*\n?$', '\n#$\n', buffer)

    # 8. Ensure #ROOMS has a terminating #0
    def fix_rooms_terminator(match):
        content = match.group(0)
        if not re.search(r'\n#0\b', content):
            content = content.rstrip() + '\n#0\n'
        return content

    buffer = re.sub(r'#(?:ROOMS|ROOMDATA)\s*\n.*?(?=\n#[A-Z$]|\Z)', fix_rooms_terminator, buffer, flags=re.DOTALL)

    return buffer

class Lexer:
    states = (
        ('line', 'exclusive'),
        ('string', 'exclusive'),
        ('optional', 'inclusive'),
    )
    tokens = (
        'AREA',
        'HELPS',
        'SOCIALS',
        'MOBILES',
        'OBJECTS',
        'ROOMS',
        'RESETS',
        'SHOPS',
        'SPECIALS',
        'END',
        'NULL',
        'NULLSTR',
        'EOL',
        'VNUM',
        'COMMENT',
        'NUMBER',
        'WORD',
        'SYMBOL',
        'QUOTED',
        'STRING',
        'EMPTY',
        'LEND',
        'TOEOL',
        'O',
        'S',
        'TILDE',
        'APPLY',
        'DOOR',
        'EXT',
        'FLAG',
        'REGEN',
    )

    t_ignore = ' \t'

    def t_AREA(self, t):
        r'\#(?:AREA|AREADATA)'
        t.value = '#AREA'
        return t

    def t_HELPS(self, t):
        r'\#HELPS'
        return t

    def t_SOCIALS(self, t):
        r'\#SOCIALS'
        return t

    def t_MOBILES(self, t):
        r'\#(?:MOBILES|MOBDATA)'
        t.value = '#MOBILES'
        return t

    def t_OBJECTS(self, t):
        r'\#(?:OBJECTS|NEWOBJECTS|OBJECTDATA)'
        t.value = '#OBJECTS'
        return t

    def t_ROOMS(self, t):
        r'\#(?:ROOMS|ROOMDATA)'
        t.value = '#ROOMS'
        return t

    def t_RESETS(self, t):
        r'\#RESETS'
        return t

    def t_SHOPS(self, t):
        r'\#SHOPS'
        return t

    def t_SPECIALS(self, t):
        r'\#SPECIALS'
        return t

    def t_END(self, t):
        r'\#\$'
        return t

    def t_NULL(self, t):
        r'\#0'
        t.lexer.begin('INITIAL')
        return t

    def t_NULLSTR(self, t):
        r'\$~'
        return t

    def t_INITIAL_line_EOL(self, t):
        r'\n'
        return t

    def t_TILDE(self, t):
        r'~'
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

    def t_SYMBOL(self, t):
        r'[^~\s]+'
        return t

    def t_error(self, t):
        log.error(f'Illegal character {repr(t.value[0])}')
        t.lexer.skip(1)

    t_string_ignore = ''

    def t_string_STRING(self, t):
        r'(?:[^~]|\.)*~'
        t.value = t.value[:-1].strip()
        t.lexer.begin('INITIAL')
        return t

    def t_string_error(self, t):
        log.error(f'Illegal string character {repr(t.value[0])}')
        t.lexer.skip(1)

    t_line_ignore = ' '

    def t_line_EMPTY(self, t):
        r'\$(?=\n)'
        t.lexer.begin('INITIAL')
        return t

    def t_line_LEND(self, t):
        r'\#\n'
        t.lexer.begin('INITIAL')
        return t

    def t_line_TOEOL(self, t):
        r'\S[^\n]*'
        t.value = t.value.strip()
        t.lexer.begin('INITIAL')
        return t

    def t_line_error(self, t):
        log.error(f'Illegal line character {repr(t.value[0])}')
        t.lexer.skip(1)

    t_optional_ignore = ''

    def t_optional_O(self, t):
        r'O'
        t.lexer.begin('INITIAL')
        return t

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
        log.error(f'Illegal optional character {repr(t.value[0])}')
        t.lexer.skip(1)

    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)
        return self.lexer

    def lex_file(self, file):
        lexer = lex.lex(module=self)
        with open(file, 'r', encoding='latin-1', errors='replace') as f:
            lexer.input(f.read())
        while True:
            tok = lexer.token()
            if not tok:
                break
            print(tok)


class Parser:
    def p_file(self, p):
        'file : sections END EOL'
        header = None
        rooms = []
        mobiles = []
        objects = []
        resets = []
        shops = []
        specials = []
        helps = []
        socials = []

        if p[1]:
            for sec in p[1]:
                if not sec or not isinstance(sec, tuple) or len(sec) < 2:
                    continue
                tag, data = sec[0], sec[1]
                if tag in ('#AREA', '#AREADATA'):
                    header = data
                elif tag in ('#ROOMS', '#ROOMDATA'):
                    if data:
                        rooms.extend(data)
                elif tag in ('#MOBILES', '#MOBDATA'):
                    if data:
                        mobiles.extend(data)
                elif tag in ('#OBJECTS', '#NEWOBJECTS', '#OBJECTDATA'):
                    if data:
                        objects.extend(data)
                elif tag == '#RESETS':
                    if data:
                        resets.extend([r for r in data if isinstance(r, (ResetDef, tuple))])
                elif tag == '#SHOPS':
                    if data:
                        shops.extend(data)
                elif tag == '#SPECIALS':
                    if data:
                        specials.extend([s for s in data if isinstance(s, (SpecialDef, tuple))])
                elif tag == '#HELPS':
                    if data:
                        helps.extend([h for h in data if isinstance(h, (HelpDef, tuple))])
                elif tag == '#SOCIALS':
                    if data:
                        socials.extend([s for s in data if isinstance(s, (SocialDef, tuple))])

        p[0] = AreaData(
            header=header,
            rooms=tuple(rooms),
            mobiles=tuple(mobiles),
            objects=tuple(objects),
            resets=tuple(resets),
            shops=tuple(shops),
            specials=tuple(specials),
            helps=tuple(helps),
            socials=tuple(socials),
        )

    def p_sections(self, p):
        '''sections : section sections
                    | EOL sections
                    |
              helps : help helps
                    | EOL helps
                    | NUMBER NULLSTR
            socials : social socials
                    | EOL socials
                    | NULL EOL
            mobiles : mobile mobiles
                    | NULL EOL
      mob_optionals : mob_optional mob_optionals
                    |
            objects : object objects
                    | NULL EOL
   object_optionals : object_optional object_optionals
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
        if p[2]:
            p[0].extend(p[2])
        if p[1] == '\n':
            p[0] = p[2]

    def p_section(self, p):
        '''section : AREA EOL area
                   | HELPS EOL helps
                   | SOCIALS EOL socials
                   | MOBILES EOL mobiles
                   | OBJECTS EOL objects
                   | ROOMS EOL rooms
                   | RESETS EOL comments resets
                   | SHOPS EOL shops
                   | SPECIALS EOL comments specials'''
        p[0] = (p[1], p[3] if len(p) != 5 else p[4])

    def p_area(self, p):
        '''area : str STRING EOL str STRING EOL str STRING EOL NUMBER NUMBER EOL'''
        p[0] = AreaHeader(
            filename=p[2],
            name=p[5],
            builder=p[8],
            vnum_min=p[10],
            vnum_max=p[11],
        )

    def p_help(self, p):
        '''help : NUMBER help_keywords TILDE EOL str STRING EOL'''
        p[0] = HelpDef(level=p[1], keywords=tuple(p[2]), text=p[6])
        log.debug(f'Help: {p[0]}')

    def p_help_keywords(self, p):
        '''help_keywords : help_keyword help_keywords
                         | EOL help_keywords
                         | '''
        p[0] = [p[1]] + p[2] if len(p) > 1 else []

    def p_help_keyword(self, p):
        '''help_keyword : WORD
                        | SYMBOL
                        | QUOTED'''
        p[0] = p[1]

    def p_social(self, p):
        '''social : WORD line EOL social_descs endl EOL
                  | WORD NUMBER NUMBER line EOL social_descs endl EOL'''
        descs = p[4] if len(p) == 7 else p[6]
        p[0] = SocialDef(name=p[1], stages=tuple(descs) if descs else ())
        log.debug(f'Social: {p[0]}')

    def p_social_descs(self, p):
        '''social_descs : TOEOL line EOL social_descs
                        | EMPTY line EOL social_descs
                        | LEND
                        |'''
        p[0] = [p[1]] + p[len(p)-1] if len(p) > 3 else []

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
        p[0] = MobileDef(
            vnum=p[1],
            player_name=p[4],
            short_desc=p[7],
            long_desc=p[10],
            desc=p[13],
            race=p[16],
            act_flags=p[18],
            affected_by=p[19],
            alignment=p[20],
            group=p[21],
            level=p[23],
            hitroll=p[24],
            hit=p[25],
            mana=p[26],
            damage=p[27],
            dam_type=p[28],
            ac_pierce=p[30],
            ac_bash=p[31],
            ac_slash=p[32],
            ac_exotic=p[33],
            off_flags=p[35],
            imm_flags=p[36],
            res_flags=p[37],
            vuln_flags=p[38],
            start_pos=p[40],
            default_pos=p[41],
            sex=p[42],
            wealth=p[43],
            form=p[45],
            parts=p[46],
            size=p[47],
            material=p[48],
            optionals=tuple(p[51]) if p[51] else (),
        )
        log.debug(f'Mob: "{p[4]}" "{p[7]}" "{p[10]}"')

    def p_mob_optional(self, p):
        '''mob_optional : flag_remove'''
        p[0] = p[3] if len(p) > 3 else None

    def p_object(self, p):
        '''object : VNUM EOL str STRING EOL str STRING EOL str STRING EOL str STRING EOL \
                    WORD flags flags EOL \
                    param param param param param EOL \
                    NUMBER NUMBER NUMBER flags optional EOL \
                    object_optionals'''
        p[0] = ObjectDef(
            vnum=p[1],
            name=p[4],
            short_desc=p[7],
            desc=p[10],
            material=p[13],
            item_type=p[15],
            extra_flags=p[16],
            wear_flags=p[17],
            values=(p[19], p[20], p[21], p[22], p[23]),
            level=p[25],
            weight=p[26],
            cost=p[27],
            condition=p[28],
            optionals=tuple(p[31]) if p[31] else (),
        )
        log.debug(f'Obj: "{p[4]}" "{p[7]}" "{p[10]}"')

    def p_object_optional(self, p):
        '''object_optional : apply
                           | ext
                           | flag_add'''
        p[0] = p[1]

    def p_room(self, p):
        '''room : VNUM EOL str STRING EOL str STRING EOL \
                  NUMBER flags NUMBER room_extra_params optional EOL \
                  room_optionals S EOL'''
        exits = []
        extras = []
        if p[15]:
            for opt in p[15]:
                if isinstance(opt, ExitDef):
                    exits.append(opt)
                elif isinstance(opt, (tuple, list)) and len(opt) == 2 and isinstance(opt[0], int) and isinstance(opt[1], int):
                    exits.append(ExitDef(direction=opt[0], dst_vnum=opt[1]))
                elif isinstance(opt, ExtraDescr):
                    extras.append(opt)
                elif opt is not None:
                    extras.append(opt)

        p[0] = RoomDef(
            vnum=p[1],
            name=p[4],
            description=p[7],
            room_flags=p[10],
            sector=p[11],
            exits=tuple(exits),
            extras=tuple(extras),
        )
        log.debug(f'Room: {p[4]}')

    def p_room_extra_params(self, p):
        '''room_extra_params : room_extra_param room_extra_params
                             | '''
        pass

    def p_room_extra_param(self, p):
        '''room_extra_param : NUMBER
                            | WORD
                            | SYMBOL'''
        p[0] = p[1]

    def p_room_optional(self, p):
        '''room_optional : door
                         | ext
                         | regen
                         | owner'''
        p[0] = p[1]

    def p_door(self, p):
        '''door : DOOR NUMBER EOL str STRING EOL str STRING EOL door_params optional EOL'''
        flags, key, dst = p[10]
        p[0] = ExitDef(
            direction=p[2],
            dst_vnum=dst,
            description=p[5] if p[5] else "",
            keyword=p[8] if p[8] else "",
            key_vnum=key,
            flags=flags,
        )

    def p_door_params(self, p):
        '''door_params : NUMBER NUMBER NUMBER NUMBER NUMBER
                       | NUMBER NUMBER NUMBER NUMBER
                       | NUMBER NUMBER NUMBER
                       | NUMBER NUMBER
                       | NUMBER'''
        nums = p[1:]
        if len(nums) == 1:
            p[0] = (0, 0, nums[0])
        elif len(nums) == 2:
            p[0] = (nums[0], 0, nums[1])
        else:
            p[0] = (nums[0], nums[1], nums[2])

    def p_regen(self, p):
        '''regen : REGEN NUMBER WORD NUMBER optional EOL'''
        pass

    def p_owner(self, p):
        '''owner : O str STRING optional EOL'''
        pass

    def p_reset(self, p):
        '''reset : WORD NUMBER NUMBER NUMBER NUMBER NUMBER comment
                 | WORD NUMBER NUMBER NUMBER NUMBER comment
                 | WORD NUMBER NUMBER NUMBER comment
                 | COMMENT EOL'''
        if p[1] and isinstance(p[1], str) and p[1].startswith('*'):
            p[0] = None
            return
        p[0] = ResetDef(
            command=p[1],
            args=tuple(v for v in p[2:-1] if v is not None),
            comment=p[len(p)-1] if isinstance(p[len(p)-1], str) else None,
        )

    def p_shop(self, p):
        '''shop : NUMBER NUMBER NUMBER NUMBER NUMBER NUMBER \
                  NUMBER NUMBER NUMBER NUMBER comment'''
        p[0] = ShopDef(
            keeper=p[1],
            buy_types=(p[2], p[3], p[4], p[5], p[6]),
            profit_buy=p[7],
            profit_sell=p[8],
            open_hour=p[9],
            close_hour=p[10],
            comment=p[11] if isinstance(p[11], str) else None,
        )

    def p_special(self, p):
        '''special : WORD NUMBER WORD comment
                   | COMMENT EOL'''
        if p[1] and isinstance(p[1], str) and p[1].startswith('*'):
            p[0] = None
            return
        p[0] = SpecialDef(
            command=p[1],
            vnum=p[2],
            spec_fun=p[3],
            comment=p[4] if isinstance(p[4], str) else None,
        )

    def p_param(self, p):
        '''param : WORD
                 | NUMBER
                 | QUOTED'''
        p[0] = p[1]

    def p_flags(self, p):
        '''flags : WORD
                 | NUMBER '''
        p[0] = eval_flags(p[1]) if p[1] is not None else None

    def p_hitndam(self, p):
        '''hitndam : NUMBER WORD NUMBER'''
        p[0] = (p[1], p[2], p[3])

    def p_flag_add(self, p):
        '''flag_add : FLAG EOL WORD NUMBER NUMBER flags optional EOL'''
        p[0] = (p[1], p[2])

    def p_flag_remove(self, p):
        '''flag_remove : FLAG WORD flags optional EOL'''
        p[0] = (p[1], p[2])

    def p_apply(self, p):
        '''apply : APPLY EOL NUMBER NUMBER optional EOL'''
        p[0] = (p[1], p[2])

    def p_ext(self, p):
        '''ext : EXT EOL str STRING EOL str STRING optional EOL'''
        p[0] = ExtraDescr(keyword=p[4], description=p[7])

    def p_str(self, p):
        '''str :'''
        p.lexer.begin('string')

    def p_line(self, p):
        '''line :'''
        p.lexer.begin('line')

    def p_endl(self, p):
        '''endl :'''
        p.lexer.begin('INITIAL')

    def p_optional(self, p):
        '''optional :'''
        p.lexer.begin('optional')

    def p_comment(self, p):
        '''comment : COMMENT EOL
                   | EOL'''
        p[0] = p[1] if len(p) == 3 else None

    def p_comments(self, p):
        '''comments : comment comments
                    | '''
        p[0] = [p[1]] + p[2] if len(p) == 3 else []

    def p_error(self, p):
        if p:
            stack_state_str = ' '.join([symbol.type for symbol in self.parser.symstack][1:])
            raise Exception(f'Syntax error in input! Parser State:{self.parser.state} {stack_state_str} . {p}')
        else:
            raise Exception('Syntax error at EOF')

    def __init__(self, write_tables=False):
        self.tokens = Lexer.tokens
        self.lexer = Lexer().build()
        self.parser = yacc.yacc(module=self, write_tables=write_tables, debug=False)

    def parse(self, buffer):
        buffer = normalize_dialect_buffer(buffer)
        return self.parser.parse(buffer, lexer=self.lexer, debug=False)

def parse_file(filepath):
    with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
        return Parser().parse(f.read())

def main():
    if len(sys.argv) > 1:
        parse_file(sys.argv[1])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
