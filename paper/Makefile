# Build recipe for the short paper.
#
# Requires a TeX Live distribution with hyperref, natbib, microtype,
# and the BibTeX engine. On Ubuntu:
#
#   sudo apt install texlive-latex-base texlive-latex-extra \
#                    texlive-fonts-recommended texlive-bibtex-extra
#
# Then `make` from inside this directory.

TEX = pdflatex
BIB = bibtex
TARGET = main

.PHONY: all clean view

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex references.bib
	$(TEX) -interaction=nonstopmode $(TARGET).tex || true
	$(BIB) $(TARGET) || true
	$(TEX) -interaction=nonstopmode $(TARGET).tex || true
	$(TEX) -interaction=nonstopmode $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).log $(TARGET).out \
	      $(TARGET).bbl $(TARGET).blg $(TARGET).toc \
	      $(TARGET).pdf

view: $(TARGET).pdf
	xdg-open $(TARGET).pdf >/dev/null 2>&1 || open $(TARGET).pdf
