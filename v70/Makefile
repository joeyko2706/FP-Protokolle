all: build/v70.pdf

# hier Python-Skripte:
build/plotdrehleck1.pdf build/plotdrehleck2.pdf /plotdrehleck3.pdf build/plotdrehleck4.pdf build/plotdrehevak.pdf build/plotturboevak.pdf build/plotturboleck1.pdf build/plotturboleck2.pdf build/plotturboleck3.pdf build/plotturboleck4.pdf build/saug1.pdf build/saug2.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/vXXX.pdf deklarieren:
build/v70.pdf: build/plotdrehleck1.pdf build/plotdrehleck2.pdf /plotdrehleck3.pdf build/plotdrehleck4.pdf build/plotdrehevak.pdf build/plotturboevak.pdf build/plotturboleck1.pdf build/plotturboleck2.pdf build/plotturboleck3.pdf build/plotturboleck4.pdf build/saug1.pdf build/saug2.pdf

build/v70.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	v70.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
