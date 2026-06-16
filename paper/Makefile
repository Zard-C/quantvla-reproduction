TECTONIC ?= tectonic
RSVG_CONVERT ?= rsvg-convert

.PHONY: all figures figpdf clean

all: main.pdf

figures:
	python3 generate_figures.py

figpdf: figures
	mkdir -p figures_pdf
	for f in figures/*.svg; do $(RSVG_CONVERT) -f pdf -o "figures_pdf/$$(basename "$$f" .svg).pdf" "$$f"; done

main.pdf: main.tex references.bib figpdf
	$(TECTONIC) main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
