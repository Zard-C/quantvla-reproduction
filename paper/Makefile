TECTONIC ?= tectonic
PDFLATEX ?= $(or $(shell command -v pdflatex 2>/dev/null),/Library/TeX/texbin/pdflatex)
BIBTEX ?= $(or $(shell command -v bibtex 2>/dev/null),/Library/TeX/texbin/bibtex)
RSVG_CONVERT ?= rsvg-convert

.PHONY: all figures figpdf main-tectonic main-pdflatex clean

all: main.pdf

figures:
	python3 generate_figures.py

figpdf: figures
	mkdir -p figures_pdf
	for f in figures/*.svg; do $(RSVG_CONVERT) -f pdf -o "figures_pdf/$$(basename "$$f" .svg).pdf" "$$f"; done

main.pdf: main-pdflatex

main-tectonic: main.tex references.bib figpdf
	$(TECTONIC) main.tex

main-pdflatex: main.tex references.bib figpdf
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIBTEX) main
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
