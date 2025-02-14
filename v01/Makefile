all: build/v01.pdf

build/v01.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v01.tex
	cp build/v01.pdf ../PDFs/v01.pdf
	cp build/v01.pdf v01.pdf
	
build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
