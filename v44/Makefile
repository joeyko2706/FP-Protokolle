all: build/v44.pdf

# hier Python-Skripte:
build/DetectorScan.pdf build/Parratt.pdf build/ReflectDiffuserScan.pdf build/Reflectivity.pdf build/RockingScan.pdf build/Zscan.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/v44.pdf deklarieren:
build/v44.pdf: build/DetectorScan.pdf build/Parratt.pdf build/ReflectDiffuserScan.pdf build/Reflectivity.pdf build/RockingScan.pdf build/Zscan.pdf

build/v44.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v44.tex
	cp build/v44.pdf v44.pdf
	cp build/v44.pdf ../PDFs/v44.pdf

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
