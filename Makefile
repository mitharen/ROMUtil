AREA_FILES := $(wildcard $(AREAS)/*.are)
SVG_FILES := $(patsubst $(AREAS)/%.are, %.svg, $(AREA_FILES))

all: $(SVG_FILES)

%.svg: $(AREAS)/%.are
	./Mapper.py $< $@
