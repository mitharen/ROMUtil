#!/usr/bin/env python
"""
Backward-compatible wrapper for romutil.parser.
"""
import logging
from romutil.parser import Lexer, Parser, parse_file, main
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

if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    main()
