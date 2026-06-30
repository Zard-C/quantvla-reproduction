PDF = main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): main.tex ../paper/references.bib ../paper/figures_pdf/fig2_offline_drift_regressions.pdf ../paper/figures_pdf/fig3_paired_rollout_flips.pdf ../paper/figures_pdf/fig4_task_redistribution.pdf
	tectonic main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
