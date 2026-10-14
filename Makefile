DOC := doc
TEX := $(DOC)/specification.tex
PDF := $(DOC)/specification.pdf
LATEX := pdflatex -interaction=nonstopmode -halt-on-error -output-directory $(DOC)

.PHONY: pdf clean

pdf: $(PDF)

# Two passes: the first writes the .aux with labels, the second resolves the equation/table references.
$(PDF): $(TEX)
	$(LATEX) $(TEX) > /dev/null || (tail -20 $(DOC)/specification.log; exit 1)
	$(LATEX) $(TEX) > /dev/null || (tail -20 $(DOC)/specification.log; exit 1)
	@echo "built $(PDF)"

clean:
	rm -f $(addprefix $(DOC)/specification.,pdf aux log out toc lof lot fls fdb_latexmk synctex.gz)
