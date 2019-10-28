all:
	$(requirements)
	$(problem_statement)

requirements:
	pdflatex requirements_document.tex
	bibtex requirements_document.aux

problem_statement:
	pdflatex problem_statement.tex
	bibtex problem_statement.aux

clean:
	rm -f *.pdf *.ps *.dvi *.out *.log *.aux *.bbl *.blg *.pyg *.toc