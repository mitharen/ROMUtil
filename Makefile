AREAS ?= $(or $(QUICKMUD_AREA_DIR),$(wildcard ../QuickMUD/area),areas)
AREA_FILES := $(wildcard $(AREAS)/*.are)
SVG_FILES := $(patsubst $(AREAS)/%.are, %.svg, $(AREA_FILES))

all: $(SVG_FILES)

%.svg: $(AREAS)/%.are
	uv run romutil $<

test:
	uv run pytest

update-deps:
	uv lock --upgrade
	uv sync
	uv run pytest
