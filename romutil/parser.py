import logging
from pathlib import Path
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


def sanitize_ackmud_colour(text: str) -> str:
    """Sanitize ACK!MUD colour markup tokens (@@<char>, @@@) into clean plain text.

    Strips @@<char> sequences (such as @@y, @@b, @@R, @@N, @@W, @@d) and expands
    escaped @@@ sequences into literal @ characters, while preserving whitespace
    and ASCII room layout formatting.
    """
    if not text or '@@' not in text:
        return text

    # Handle escaped @@: @@@ -> literal @ (using placeholder to avoid accidental stripping)
    escaped_marker = '\x00_ACK_AT_\x00'
    cleaned = text.replace('@@@', escaped_marker)

    # Cleanly strip @@<char> sequences where char is not newline or tilde
    cleaned = re.sub(r'@@[^\n~]', '', cleaned)

    # Strip any dangling @@ before newline, tilde, or EOF
    cleaned = re.sub(r'@@(?=[\n~]|\Z)', '', cleaned)

    # Restore literal @
    return cleaned.replace(escaped_marker, '@')


def normalize_dialect_buffer(buffer: str) -> str:
    """Normalize dialect variations across Merc, Envy, Diku, CircleMUD, and ACK!MUD.

    Handles:
    - ACK!MUD colour markup tokens (@@<char>, @@@) sanitized into clean plain text.
    - ACK!MUD 4.3 tagged #AREA headers (K, V, O, L, Q, etc.) converted into canonical 4-line format.
    - Envy #AREADATA ... End header converted into standard #AREA format.
    - Merc inline or single-line #AREA headers converted to 4-line standard format.
    - Missing #AREA header synthesized with default area metadata.
    - Piped bitmask flags evaluated (4|8|1024 -> 1036, A|B -> AB).
    - ANATOLIA 3.0 #RESETMESSAGE and #FLAG top-level sections normalized for PLY grammar ingestion.
    - Strips unsupported dialect-specific sections (#GAMES, #CLANS, #ECONOMY, #OLC).
    - Ensures CircleMUD / headerless room definitions have #ROOMS section marker.
    - Normalizes CircleMUD / Diku EOF delimiters ($~, $) to standard #$.
    - Ensures room sections terminate with #0 before EOF.
    """
    if not buffer or not buffer.strip():
        return buffer

    # 1. Sanitize ACK!MUD colour codes across the entire buffer
    buffer = sanitize_ackmud_colour(buffer)

    # 2. Normalize piped bitmask flags
    def replace_pipe_flags(match):
        parts = match.group(0).split('|')
        val = 0
        for p in parts:
            val |= int(p)
        return str(val)

    buffer = re.sub(r'\b\d+(?:\|\d+)+\b', replace_pipe_flags, buffer)
    buffer = re.sub(r'(?<=[A-Za-z])\|(?=[A-Za-z])', '', buffer)

    # 3. Normalize Envy #AREADATA ... End
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
                if k in ('VNUMs', 'V'):
                    nums = [int(n) for n in v.split() if n.lstrip('-+').isdigit()]
                    if len(nums) >= 2:
                        data['vnum_min'] = nums[0]
                        data['vnum_max'] = nums[1]
        name = data.get('Name', data.get('K', 'Area'))
        author = data.get('Author', data.get('Builders', data.get('O', data.get('L', 'Unknown'))))
        v_min = data.get('vnum_min', 0)
        v_max = data.get('vnum_max', 0)
        file_name = data.get('FileName', name.lower().replace(' ', '_') + ".are")
        return f"#AREA\n{file_name}~\n{name}~\n{author}~\n{v_min} {v_max}\n"

    buffer = re.sub(r'#AREADATA\s*\n(.*?)\nEnd\b', replace_areadata, buffer, flags=re.DOTALL)

    # 4. Normalize ACK!MUD 4.3 tagged single-character #AREA headers
    def replace_ackmud_area(match: re.Match[str]) -> str:
        inline_fn = match.group(1)
        body = match.group(2)
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        tags: dict[str, str] = {}
        other_lines: list[str] = []
        for line in lines:
            if line.startswith('*') or line in ('End', 'E'):
                continue
            m = re.match(r'^([A-Za-z])\s+(.*)', line)
            if m:
                tag_char = m.group(1).upper()
                val = m.group(2).strip()
                if tag_char not in tags:
                    tags[tag_char] = val
            else:
                other_lines.append(line)

        # Require presence of characteristic ACK!MUD tags
        ack_chars = {'K', 'V', 'Q', 'O', 'L', 'N', 'I', 'X', 'F', 'U', 'R', 'W', 'M', 'S'}
        matched_ack = ack_chars.intersection(tags.keys())
        if not matched_ack:
            return match.group(0)
        if not ('K' in tags or 'V' in tags or 'Q' in tags or len(matched_ack) >= 2):
            return match.group(0)

        name = tags.get('K', '').rstrip('~').strip()
        v_min = 0
        v_max = 0
        if 'V' in tags:
            nums = [int(n) for n in tags['V'].split() if n.lstrip('-+').isdigit()]
            if len(nums) >= 2:
                v_min, v_max = nums[0], nums[1]
            elif len(nums) == 1:
                v_min = nums[0]

        builder = tags.get('O', '').rstrip('~').strip()
        if not builder:
            builder = tags.get('L', '').rstrip('~').strip()
        if not builder:
            builder = 'Unknown'

        file_name = ''
        if inline_fn:
            file_name = inline_fn.rstrip('~').strip()
        elif 'A' in tags:
            file_name = tags['A'].rstrip('~').strip()
        else:
            for other in other_lines:
                cleaned = other.rstrip('~').strip()
                if cleaned.lower().endswith('.are') or '.' in cleaned:
                    file_name = cleaned
                    break
        if not file_name:
            base_name = name or 'area'
            file_name = base_name.lower().replace(' ', '_') + '.are'
        if not name:
            file_name_stem = Path(file_name).stem
            name = file_name_stem.replace('_', ' ').title() if file_name_stem else 'Area'

        return f"#AREA\n{file_name}~\n{name}~\n{builder}~\n{v_min} {v_max}\n"

    buffer = re.sub(
        r'#AREA(?:[ \t]+([^\n~]*~))?[ \t]*\n(.*?)(?=\n#[A-Z$]|\Z)',
        replace_ackmud_area,
        buffer,
        flags=re.DOTALL,
    )

    # 5. Normalize Merc single-line or inline #AREA headers
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

    # ANATOLIA 3.0: Normalize #RESETMESSAGE
    def replace_resetmessage(match: re.Match[str]) -> str:
        msg = match.group(1).rstrip("~").strip()
        return f"#RESETMESSAGE\n{msg}~\n"

    buffer = re.sub(
        r'#RESETMESSAGE\s+([^~]*~)',
        replace_resetmessage,
        buffer,
    )

    # ANATOLIA 3.0: Normalize #FLAG
    def replace_area_flag(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if body:
            normalized_flags = ' '.join(body.split())
            return f"#FLAG\n{normalized_flags}\n"
        return "#FLAG\n"

    buffer = re.sub(
        r'#FLAG[ \t]*(.*?)(?=\n#[A-Z$]|\Z)',
        replace_area_flag,
        buffer,
        flags=re.DOTALL,
    )

    # 4. Strip dialect-specific non-standard sections
    buffer = re.sub(r'#(?:GAMES|CLANS|ECONOMY|OLC|PRACTICERS|RESETCONT)\b.*?(?=\n#|\Z)', '', buffer, flags=re.DOTALL)

    # 5. If no #ROOMS or #ROOMDATA header but room VNUMs exist in standalone room files, prepend #ROOMS
    if not re.search(r'#(?:ROOMS|ROOMDATA)\b', buffer) and not re.search(r'#(?:HELPS|SOCIALS|MOBILES|OBJECTS|RESETMESSAGE|FLAG)\b', buffer):
        if re.search(r'^\#\d+\b', buffer, re.MULTILINE):
            buffer = re.sub(r'\A(?:[ \t\r\n]|(?:\*[^\n]*\n))+', '', buffer)
            buffer = re.sub(r'(?m)^\*[^\n]*\n(?=\s*#\d+)', '', buffer)
            buffer = re.sub(r'(^\#\d+\b)', r'#ROOMS\n\1', buffer, count=1, flags=re.MULTILINE)

    # 7. Normalize CircleMUD / Diku EOF delimiters ($~, trailing $) and strip sentinels (#99999\n$~, #0\n$~)
    buffer = re.sub(r'\n\#\d+\s*\n?\$~?\s*$', '\n#$\n', buffer)
    buffer = re.sub(r'\n\#(?:99999)\s*$', '\n#$\n', buffer)
    buffer = re.sub(r'\n\$\~[ \t]*\n?$', '\n#$\n', buffer)
    buffer = re.sub(r'\n\$[ \t]*\n?$', '\n#$\n', buffer)

    # 8. Ensure #ROOMS has a terminating #0
    def fix_rooms_terminator(match: re.Match[str]) -> str:
        content = match.group(0)
        if not re.search(r'\n#0\s*$', content.rstrip()):
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
        'RESETMESSAGE',
        'AREA_FLAG',
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

    def t_RESETMESSAGE(self, t):
        r'\#RESETMESSAGE'
        return t

    def t_AREA_FLAG(self, t):
        r'\#FLAG'
        t.value = '#FLAG'
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
        r'\#0(?!\d)'
        # If #0 is followed by a room title (a string ending with ~), it is Room VNUM 0!
        rest = t.lexer.lexdata[t.lexer.lexpos:]
        stripped = rest.lstrip(' \t\r\n')
        while stripped.startswith('*'):
            stripped = stripped.split('\n', 1)[1].lstrip(' \t\r\n') if '\n' in stripped else ''
        first_line = stripped.split('\n', 1)[0].strip() if stripped else ''
        if first_line.rstrip().endswith('~') and not first_line.lstrip().startswith(('#', '$')):
            t.type = 'VNUM'
            t.value = 0
            t.lexer.begin('INITIAL')
            return t
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
        t.value = sanitize_ackmud_colour(t.value[:-1].strip())
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
        t.value = sanitize_ackmud_colour(t.value.strip())
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
        reset_message = None
        flag = None

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
                elif tag == '#RESETMESSAGE':
                    reset_message = data
                elif tag == '#FLAG':
                    flag = data

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
            reset_message=reset_message,
            flag=flag,
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
                   | RESETMESSAGE EOL reset_message
                   | RESETMESSAGE reset_message
                   | AREA_FLAG EOL area_flag
                   | AREA_FLAG area_flag
                   | AREA_FLAG EOL
                   | HELPS EOL helps
                   | SOCIALS EOL socials
                   | MOBILES EOL mobiles
                   | OBJECTS EOL objects
                   | ROOMS EOL rooms
                   | RESETS EOL comments resets
                   | SHOPS EOL shops
                   | SPECIALS EOL comments specials'''
        if len(p) == 5:
            data = p[4]
        elif len(p) == 4:
            data = p[3]
        elif len(p) == 3:
            data = p[2] if p[2] != "\n" else None
        else:
            data = None
        p[0] = (p[1], data)

    def p_reset_message(self, p):
        '''reset_message : str STRING EOL
                         | str STRING'''
        p[0] = p[2]

    def p_area_flag(self, p):
        '''area_flag : area_flag_tokens EOL
                     | area_flag_tokens'''
        p[0] = p[1]

    def p_area_flag_tokens(self, p):
        '''area_flag_tokens : area_flag_token area_flag_tokens
                            | area_flag_token'''
        if len(p) == 3:
            p[0] = f"{p[1]} {p[2]}"
        else:
            p[0] = str(p[1])

    def p_area_flag_token(self, p):
        '''area_flag_token : WORD
                           | NUMBER
                           | SYMBOL'''
        p[0] = str(p[1])

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


def parse_circlemud_zone_file(file_path_or_content: str | Path) -> tuple[AreaHeader, tuple[ResetDef, ...]]:
    """Parse a CircleMUD 3.x / tbaMUD zone (.zon) configuration file.

    Extracts zone identification, builder/author, room VNUM boundaries,
    lifespan/reset configurations, and reset commands into domain models.
    """
    if isinstance(file_path_or_content, Path) or (
        isinstance(file_path_or_content, str)
        and "\n" not in file_path_or_content
        and Path(file_path_or_content).is_file()
    ):
        p = Path(file_path_or_content)
        filename = p.name
        with open(p, "r", encoding="latin-1", errors="replace") as f:
            content = f.read()
    else:
        filename = "zone.zon"
        content = str(file_path_or_content)

    lines = content.splitlines()
    idx = 0
    zone_num = None
    header_line = ""
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith(("*", ";")):
            continue
        if line.startswith("#"):
            m = re.match(r"^#(\d+)\s*(.*)", line)
            if m:
                zone_num = int(m.group(1))
                header_line = m.group(2).strip()
                break
        raise ValueError(f"Invalid CircleMUD zone header: expected '#<zone_num>', got {line!r}")

    if zone_num is None:
        raise ValueError("Invalid CircleMUD zone file: no '#<zone_num>' found")

    current_str_parts: list[str] = []
    if header_line:
        current_str_parts.append(header_line)

    numeric_tokens: list[str] = []
    while idx < len(lines):
        raw_line = lines[idx]
        idx += 1
        line = raw_line.strip()
        if not line or line.startswith(("*", ";")):
            continue

        parts = line.split()
        if parts and parts[0].isdigit() and "~" not in line:
            if not current_str_parts or all("~" in s for s in current_str_parts):
                numeric_tokens = parts
                break

        current_str_parts.append(line)

    combined_str_text = " ".join(current_str_parts)
    raw_strings = re.findall(r"([^~]+)~", combined_str_text)
    tilde_strings = [s.strip() for s in raw_strings if s.strip()]

    if not numeric_tokens:
        raise ValueError("Invalid CircleMUD zone file: missing numeric zone parameters line")

    if len(tilde_strings) >= 2:
        builder = tilde_strings[0]
        name = tilde_strings[1]
    elif len(tilde_strings) == 1:
        builder = "Unknown"
        name = tilde_strings[0]
    else:
        builder = "Unknown"
        name = f"Zone {zone_num}"

    nums = [int(t) for t in numeric_tokens if t.lstrip("-+").isdigit()]
    if len(nums) >= 4 and nums[0] <= nums[1]:
        vnum_min = nums[0]
        vnum_max = nums[1]
    elif len(nums) >= 3:
        vnum_max = nums[0]
        vnum_min = zone_num * 100 if vnum_max >= zone_num * 100 else max(0, vnum_max - 99)
    elif len(nums) >= 1:
        vnum_max = nums[0]
        vnum_min = zone_num * 100 if vnum_max >= zone_num * 100 else 0
    else:
        vnum_min = zone_num * 100
        vnum_max = zone_num * 100 + 99

    resets: list[ResetDef] = []
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith(("*", ";")):
            continue
        if line.startswith(("S", "$")):
            break

        comment = None
        cmd_part = line
        for marker in ("*", ";"):
            if marker in line:
                cmd_part, comment_part = line.split(marker, 1)
                comment = comment_part.strip()
                break
        if comment is None and "(" in line and line.rstrip().endswith(")"):
            c_start = line.index("(")
            cmd_part = line[:c_start].strip()
            comment = line[c_start + 1:].rstrip(")").strip()

        tokens = cmd_part.strip().split()
        if not tokens:
            continue

        cmd = tokens[0].upper()
        args: list[Any] = []
        for t in tokens[1:]:
            if t.lstrip("-+").isdigit():
                args.append(int(t))
            else:
                args.append(t)

        resets.append(ResetDef(command=cmd, args=tuple(args), comment=comment))

    header = AreaHeader(
        filename=filename,
        name=name,
        builder=builder,
        vnum_min=vnum_min,
        vnum_max=vnum_max,
    )
    return header, tuple(resets)


def parse_circlemud_directory(directory_path: str | Path) -> AreaData:
    """Parse a split-file CircleMUD 3.x or tbaMUD world directory.

    Discovers room definitions (*.wld) and zone metadata (*.zon) across
    flat directories or hierarchical lib/world/ trees, matching zones by
    filename prefix or VNUM range, and merges them into a cohesive AreaData model.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")

    # Determine wld and zon directories
    wld_dir = dir_path / "wld" if (dir_path / "wld").is_dir() else dir_path
    if (dir_path / "zon").is_dir():
        zon_dir = dir_path / "zon"
    elif dir_path.name == "wld" and (dir_path.parent / "zon").is_dir():
        zon_dir = dir_path.parent / "zon"
    else:
        zon_dir = dir_path

    # Discover .wld files via index or file glob
    wld_files: list[Path] = []
    for index_candidate in (wld_dir / "index", dir_path / "index"):
        if index_candidate.is_file():
            with open(index_candidate, "r", encoding="latin-1", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("*"):
                        continue
                    if line == "$":
                        break
                    fname = line if line.endswith(".wld") else f"{line}.wld"
                    candidate = wld_dir / fname
                    if not candidate.is_file():
                        candidate = dir_path / fname
                    if candidate.is_file():
                        wld_files.append(candidate)
            if wld_files:
                break

    if not wld_files:
        found = list(wld_dir.glob("*.wld"))
        if not found and wld_dir != dir_path:
            found = list(dir_path.glob("*.wld"))
        if not found:
            found = list(dir_path.rglob("*.wld"))

        def sort_key(p: Path):
            stem = p.stem
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        wld_files = sorted(found, key=sort_key)

    if not wld_files:
        raise FileNotFoundError(f"No CircleMUD .wld room files found in directory: {directory_path}")

    # Discover and parse .zon files
    zon_files = list(zon_dir.glob("*.zon"))
    if not zon_files and zon_dir != dir_path:
        zon_files = list(dir_path.glob("*.zon"))
    if not zon_files:
        zon_files = list(dir_path.rglob("*.zon"))

    zones_by_stem: dict[str, tuple[AreaHeader, tuple[ResetDef, ...]]] = {}
    zones_by_range: list[tuple[AreaHeader, tuple[ResetDef, ...]]] = []
    for zf in zon_files:
        h, r = parse_circlemud_zone_file(zf)
        zones_by_stem[zf.stem] = (h, r)
        zones_by_range.append((h, r))

    # Parse each .wld file and match with zone metadata
    all_rooms: list[RoomDef] = []
    all_resets: list[ResetDef] = []
    matched_headers: list[AreaHeader] = []

    parser = Parser()
    for wf in wld_files:
        with open(wf, "r", encoding="latin-1", errors="replace") as f:
            wld_content = f.read()

        parsed_area = parser.parse(wld_content)
        rooms = list(parsed_area.rooms)
        if not rooms:
            continue

        all_rooms.extend(rooms)
        room_vnums = [r.vnum for r in rooms]
        min_vnum = min(room_vnums)
        max_vnum = max(room_vnums)

        header: AreaHeader | None = None
        resets: tuple[ResetDef, ...] = ()
        if wf.stem in zones_by_stem:
            header, resets = zones_by_stem[wf.stem]
        else:
            for zh, zr in zones_by_range:
                if not (max_vnum < zh.vnum_min or min_vnum > zh.vnum_max):
                    header, resets = zh, zr
                    break

        if header is None:
            header = AreaHeader(
                filename=f"{wf.stem}.zon",
                name=f"Zone {wf.stem}",
                builder="Unknown",
                vnum_min=min_vnum,
                vnum_max=max_vnum,
            )
            resets = ()

        matched_headers.append(header)
        all_resets.extend(resets)

    if not all_rooms:
        raise ValueError(f"No valid room definitions found in CircleMUD .wld files in {directory_path}")

    all_rooms.sort(key=lambda r: r.vnum)

    if len(matched_headers) == 1:
        single = matched_headers[0]
        combined_header = AreaHeader(
            filename=single.filename,
            name=single.name,
            builder=single.builder,
            vnum_min=min(single.vnum_min, all_rooms[0].vnum),
            vnum_max=max(single.vnum_max, all_rooms[-1].vnum),
        )
    elif matched_headers:
        min_v = min(h.vnum_min for h in matched_headers)
        max_v = max(h.vnum_max for h in matched_headers)
        names = [h.name for h in matched_headers if h.name]
        combined_name = " / ".join(names) if len(names) <= 3 else f"{names[0]} (+{len(names)-1} zones)"
        combined_builder = matched_headers[0].builder
        combined_header = AreaHeader(
            filename=f"{dir_path.name}.are",
            name=combined_name,
            builder=combined_builder,
            vnum_min=min(min_v, all_rooms[0].vnum),
            vnum_max=max(max_v, all_rooms[-1].vnum),
        )
    else:
        combined_header = AreaHeader(
            filename=f"{dir_path.name}.are",
            name=dir_path.name.capitalize(),
            builder="Unknown",
            vnum_min=all_rooms[0].vnum,
            vnum_max=all_rooms[-1].vnum,
        )

    return AreaData(
        header=combined_header,
        rooms=tuple(all_rooms),
        resets=tuple(all_resets),
    )

def parse_file(filepath):
    with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
        return Parser().parse(f.read())

def main():
    if len(sys.argv) > 1:
        parse_file(sys.argv[1])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
