all: build/v46.pdf

# hier Python-Skripte:
build/plot-Fluss.pdf build/plot-linReg.pdf build/plot-Wert.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/vXXX.pdf deklarieren:
build/v46.pdf: build/plot-Fluss.pdf build/plot-linReg.pdf build/plot-Wert.pdf

build/v46.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v46.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
