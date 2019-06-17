#!/bin/bash
# Author: Jacob Griffiths jacob.griffiths18@imperial.ac.uk
# Script: CompileLaTeX.sh
# Desc: Compiles a LaTeX document with references
# Arguments: 1 .tex file and 1 .bib file
# Date: Oct 2018

file="$1"
filename="${file%%.*}" #strip name

# Compile
{
  pdflatex "$filename"
  bibtex "$filename"
  pdflatex "$filename"
} &> /dev/null
pdflatex "$filename"
printf "\n"

# Move output file to ../Results 
mv "$filename".pdf ../Results

{
        gnome-open "$filename.pdf"
} &> /dev/null

## Cleanup
rm -f *~
rm -f *.aux
rm -f *.dvi
rm -f *.log
rm -f *.nav
rm -f *.out
rm -f *.snm
rm -f *.toc
rm -f *.bbl
rm -f *.blg

exit