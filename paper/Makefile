# Build recipe for the short paper.
#
# Requires a TeX Live distribution with hyperref, natbib, microtype,
# lmodern, and the BibTeX engine. On Ubuntu:
#
#   sudo apt install texlive-latex-base texlive-latex-extra \
#                    texlive-fonts-recommended texlive-bibtex-extra \
#                    lmodern
#
# Then `make` from inside this directory.
#
# Targets:
#   make          -> main.pdf
#   make arxiv    -> arxiv-submission.tar.gz (bundle ready to upload)
#   make clean    -> remove build artefacts and bundle
#   make view     -> open main.pdf (xdg-open / open)

TEX = pdflatex
BIB = bibtex
TARGET = main
ARXIV_BUNDLE = arxiv-submission.tar.gz

.PHONY: all clean view arxiv

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex references.bib
	$(TEX) -interaction=nonstopmode $(TARGET).tex || true
	$(BIB) $(TARGET) || true
	$(TEX) -interaction=nonstopmode $(TARGET).tex || true
	$(TEX) -interaction=nonstopmode $(TARGET).tex

# arXiv prefers source submissions with the .bbl file pre-generated rather
# than the raw .bib file, so its compile step stays deterministic (arXiv
# does not re-run bibtex). The build chain above produces main.bbl as a
# side-effect; this target bundles main.tex + main.bbl into a tarball.
arxiv: $(ARXIV_BUNDLE)

# The bundle requires main.tex (always in git) + main.bbl (committed by the
# paper-pdf CI workflow alongside main.pdf, so a `git pull` is enough on a
# machine without LaTeX installed). If main.bbl is missing locally, run
# `make` to regenerate it (needs LaTeX), or `git pull` from a fresh CI run.
$(ARXIV_BUNDLE): $(TARGET).tex $(TARGET).bbl
	tar czf $(ARXIV_BUNDLE) \
	    $(TARGET).tex \
	    $(TARGET).bbl
	@echo
	@echo "arXiv bundle created: $(ARXIV_BUNDLE)"
	@echo "Contents:"
	@tar tzf $(ARXIV_BUNDLE) | sed 's/^/  /'
	@echo
	@echo "Next: see paper/ARXIV_SUBMISSION.md for the upload procedure."

clean:
	rm -f $(TARGET).aux $(TARGET).log $(TARGET).out \
	      $(TARGET).bbl $(TARGET).blg $(TARGET).toc \
	      $(TARGET).pdf $(ARXIV_BUNDLE)

view: $(TARGET).pdf
	xdg-open $(TARGET).pdf >/dev/null 2>&1 || open $(TARGET).pdf
