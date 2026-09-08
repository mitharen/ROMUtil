#!/usr/bin/env python
"""
Backward-compatible wrapper for romutil.parser.
"""
import logging
from romutil.parser import Lexer, Parser, parse_file, main

log = logging.getLogger('AreaParser')

if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    main()
