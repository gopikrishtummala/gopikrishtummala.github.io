# Resume Source Files

This directory contains the LaTeX source files for my resume.

## Building the PDF

To compile the resume PDF, use one of the following methods:

### Using pdflatex (standard)
```bash
cd resume
pdflatex resume.tex
```

### Using latexmk (recommended - handles multiple passes automatically)
```bash
cd resume
latexmk -pdf resume.tex
```

### Clean build artifacts
```bash
cd resume
latexmk -c  # Clean auxiliary files
# or manually:
rm -f *.aux *.log *.out *.fdb_latexmk *.fls *.synctex.gz
```

## Output

The compiled `resume.pdf` should be copied to the `public/` directory to be accessible on the website:

```bash
cp resume.pdf ../public/resume.pdf
```

## Dependencies

The resume uses the following LaTeX packages:
- `article` document class
- `geometry` for margins
- `enumitem` for list formatting
- `hyperref` for links
- `titlesec` for section formatting
- `fontawesome5` for icons
- `xcolor` for colors

Make sure these packages are installed in your LaTeX distribution.
