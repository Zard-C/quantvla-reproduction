PDF = main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): main.tex ../paper/references.bib ../paper/figures_pdf/fig5_tactic_pareto.pdf
	tectonic main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
