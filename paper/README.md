# When Small Action Errors Matter

This directory contains the paper draft and arXiv-ready source material for:

> When Small Action Errors Matter: Closed-Loop Analysis of Post-Training Quantization for VLA Policies

## Build

The PDF can be rebuilt locally with:

```bash
make
```

The build expects:

- `tectonic` for LaTeX compilation
- `rsvg-convert` from `librsvg` for SVG-to-PDF figure conversion
- Python 3 for regenerating the SVG figures

The generated PDF is `main.pdf`.

## arXiv Source

The arXiv source bundle should contain only the files needed to compile the paper:

- `main.tex`
- `main.bbl`
- `references.bib`
- `figures_pdf/*.pdf`

The PDF, logs, auxiliary files, Markdown draft, and figure-generation scripts are useful for local development, but are not required in the arXiv upload bundle.

Suggested arXiv metadata:

- Title: `When Small Action Errors Matter: Closed-Loop Analysis of Post-Training Quantization for VLA Policies`
- Authors: `patrick.zhang`
- Primary category: `cs.RO`
- Cross-list candidates: `cs.LG`, `cs.CV`
