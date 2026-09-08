import os
import pytest
import AreaParser

SAMPLE_AREAS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../QuickMUD/area'))

MINIMAL_VALID_AREA = """#AREA
test.are~
Test Area~
{ 1 10 } Builder Test Area~
100 199

#ROOMS
#100
Room One~
First room description.~
0 0 0
D0
North door~
~
0 0 101
S
#101
Room Two~
Second room description.~
0 0 0
D2
South door~
~
0 0 100
S
#0

#$
"""

HELPS_VALID_AREA = """#AREA
test_help.are~
Help Area~
{ 1 10 } Builder Help~
200 299

#HELPS
1 'TEST KEYWORD'~
Help text for test keyword.
~
0 $~

#$
"""

SOCIALS_VALID_AREA = """#AREA
test_social.are~
Social Area~
{ 1 10 } Builder Social~
300 399

#SOCIALS
wave
You wave goodbye.
$n waves goodbye.
You wave goodbye to $N.
$n waves goodbye to $N.
$n waves goodbye to you.
#

#0

#$
"""


class TestLexer:
    """Unit tests for AreaParser Lexer tokenization and state transitions."""

    def test_lexer_build(self):
        lexer_obj = AreaParser.Lexer()
        lexer_obj.build()
        assert lexer_obj.lexer is not None

    def test_lexer_section_tokens(self):
        lexer_obj = AreaParser.Lexer()
        lexer_obj.build()
        data = "#AREA #HELPS #SOCIALS #MOBILES #OBJECTS #ROOMS #RESETS #SHOPS #SPECIALS #$ #0 $~"
        lexer_obj.lexer.input(data)
        tokens = []
        while True:
            tok = lexer_obj.lexer.token()
            if not tok:
                break
            tokens.append(tok.type)
        expected = [
            'AREA', 'HELPS', 'SOCIALS', 'MOBILES', 'OBJECTS',
            'ROOMS', 'RESETS', 'SHOPS', 'SPECIALS', 'END', 'NULL', 'NULLSTR'
        ]
        assert tokens == expected

    def test_lexer_numbers_and_vnums(self):
        lexer_obj = AreaParser.Lexer()
        lexer_obj.build()
        data = "#1234 56 -78 +90"
        lexer_obj.lexer.input(data)
        toks = []
        while True:
            tok = lexer_obj.lexer.token()
            if not tok:
                break
            toks.append((tok.type, tok.value))
        assert toks == [
            ('VNUM', 1234),
            ('NUMBER', 56),
            ('NUMBER', -78),
            ('NUMBER', 90)
        ]

    def test_lexer_quoted_and_word(self):
        lexer_obj = AreaParser.Lexer()
        lexer_obj.build()
        data = "'sample quoted' keyword"
        lexer_obj.lexer.input(data)
        toks = []
        while True:
            tok = lexer_obj.lexer.token()
            if not tok:
                break
            toks.append((tok.type, tok.value))
        assert toks == [
            ('QUOTED', 'sample quoted'),
            ('WORD', 'keyword')
        ]

    def test_lexer_lex_file(self, tmp_path, capsys):
        test_file = tmp_path / "test.are"
        test_file.write_text("#AREA\ntest.are~\nTest~\nBuilder~\n1 10\n#$\n")
        lexer_obj = AreaParser.Lexer()
        lexer_obj.lex_file(str(test_file))
        captured = capsys.readouterr().out
        assert "LexToken(AREA" in captured

    def test_parser_main(self, tmp_path, monkeypatch):
        test_file = tmp_path / "test_main.are"
        test_file.write_text(MINIMAL_VALID_AREA)
        monkeypatch.setattr("sys.argv", ["AreaParser.py", str(test_file)])
        AreaParser.main()

    def test_parser_room_with_regen_and_owner(self):
        area_text = """#AREA
test_opt.are~
Test Opt~
{ 1 10 } Builder Opt~
100 199

#ROOMS
#100
Room With Regen~
Room desc.~
0 0 0
H 100 mana 50
O TheKing~
S
#0

#$
"""
        parser = AreaParser.Parser()
        result = parser.parse(area_text)
        assert result is not None
        sections = {s[0]: s[1] for s in result if s}
        assert '#ROOMS' in sections
        assert sections['#ROOMS'][0][0] == 100

    def test_lexer_illegal_character(self, caplog):
        lexer_obj = AreaParser.Lexer()
        lexer_obj.build()
        lexer_obj.lexer.input("\r")
        tok = lexer_obj.lexer.token()
        assert "Illegal character" in caplog.text

    def test_lexer_line_illegal_character(self, caplog):
        lexer_obj = AreaParser.Lexer()
        lexer_obj.build()
        lexer_obj.lexer.input("\t\n")
        lexer_obj.lexer.begin('line')
        tok = lexer_obj.lexer.token()
        assert "Illegal line character" in caplog.text


class TestParserPositive:
    """Positive test cases for parsing area files."""

    def test_parse_minimal_valid_area(self):
        parser = AreaParser.Parser()
        result = parser.parse(MINIMAL_VALID_AREA)
        assert result is not None

        sections = {s[0]: s[1] for s in result if s}
        assert '#AREA' in sections
        assert '#ROOMS' in sections

        area_meta = sections['#AREA']
        assert area_meta[0] == 'test.are'
        assert area_meta[1] == 'Test Area'
        assert area_meta[3] == (100, 199)

        rooms = sections['#ROOMS']
        assert len(rooms) == 2
        # Room 100
        assert rooms[0][0] == 100
        assert rooms[0][1] == 'Room One'
        # Check exits: door 0 (north) pointing to 101
        assert (0, 101) in rooms[0][3]
        # Room 101
        assert rooms[1][0] == 101
        assert rooms[1][1] == 'Room Two'
        # Door 2 (south) pointing to 100
        assert (2, 100) in rooms[1][3]

    def test_parse_helps_section(self):
        parser = AreaParser.Parser()
        result = parser.parse(HELPS_VALID_AREA)
        assert result is not None
        sections = {s[0]: s[1] for s in result if s}
        assert '#HELPS' in sections
        helps = sections['#HELPS']
        assert len(helps) >= 1
        assert 'TEST KEYWORD' in helps[0][0]

    def test_parse_socials_section(self):
        parser = AreaParser.Parser()
        result = parser.parse(SOCIALS_VALID_AREA)
        assert result is not None
        sections = {s[0]: s[1] for s in result if s}
        assert '#SOCIALS' in sections
        socials = sections['#SOCIALS']
        assert len(socials) >= 1
        assert socials[0][0] == 'wave'

    @pytest.mark.skipif(not os.path.isdir(SAMPLE_AREAS_DIR), reason="QuickMUD area files not available")
    @pytest.mark.parametrize("filename", ["school.are", "smurf.are", "social.are", "help.are"])
    def test_parse_real_area_files(self, filename):
        filepath = os.path.join(SAMPLE_AREAS_DIR, filename)
        parser = AreaParser.Parser()
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()
        result = parser.parse(content)
        assert result is not None
        assert len(result) > 0


class TestParserNegative:
    """Negative test cases verifying robust error reporting on malformed inputs."""

    def test_parse_empty_input(self):
        parser = AreaParser.Parser()
        with pytest.raises(Exception, match=r"Syntax error at EOF"):
            parser.parse("")

    def test_parse_truncated_input(self):
        parser = AreaParser.Parser()
        truncated = "#AREA\ntest.are~\n"
        with pytest.raises(Exception):
            parser.parse(truncated)

    def test_parse_missing_end_marker(self):
        parser = AreaParser.Parser()
        missing_end = MINIMAL_VALID_AREA.replace("#$", "")
        with pytest.raises(Exception):
            parser.parse(missing_end)

    def test_parse_corrupt_room_syntax(self):
        parser = AreaParser.Parser()
        corrupt_room = """#AREA
test.are~
Test~
Builder Test~
100 199

#ROOMS
#100
Missing Tilde Name
Description~
0 0 0
S
#0

#$
"""
        with pytest.raises(Exception):
            parser.parse(corrupt_room)

    def test_parse_invalid_section_header(self):
        parser = AreaParser.Parser()
        bad_section = """#INVALID_HEADER
content
#$
"""
        with pytest.raises(Exception):
            parser.parse(bad_section)
