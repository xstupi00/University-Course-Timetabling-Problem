# makefile for compilation
CO=2019-PPFIT-ShortName

all: $(CO).pdf

$(CO).pdf: clean
	pdflatex $(CO)
	bibtex $(CO)
	pdflatex $(CO)
	pdflatex $(CO)

clean:
	rm -f *.dvi *.log $(CO).blg $(CO).bbl $(CO).toc *.aux $(CO).out $(CO).lof $(CO).ptc
	rm -f $(CO).pdf
	rm -f *~

pack:
	tar czvf $(CO).tar.gz *.tex *.bib *.bst ./images/* $(CO).pdf Makefile

vlna:
	vlna -l *.tex

# Spocita normostrany / Count of standard pages
normostrany:
	echo "scale=2; `detex -n *.tex | wc -c`/1800;" | bc

