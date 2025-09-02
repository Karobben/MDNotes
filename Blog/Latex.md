---
title: "Latex grammar highlight Plugin for Atom"
description: "Latex"
date: 2020/08/25
url: latex
toc: true
excerpt: "Latex grammar highlight Plugin for Atom"
tags: [IDE, Latex]
category: [others, else]
cover: 'https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMDA1MDkxNjM4ODYy'
thumbnail: 'https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMDA1MDkxNjM4ODYy'
priority: 10000
---

## Install Tex Live

```bash
sudo apt install texlive-latex-base # not recommended. You may install a very old version which can't install or update packages.
```

Make sure to delete the old version of tex live to avoid the conflict and confusion. If you forget where are they or the name of the package you installed, you can use the following command to find them:


```bash
dpkg -S /usr/bin/pdflatex
dpkg -S /usr/bin/tlmgr
```
<pre>
texlive-latex-base: /usr/bin/pdflatex
texlive-base: /usr/bin/tlmgr
</pre>

Then, you can remove them by using the following command:

```bash
sudo apt remove texlive-latex-base
sudo apt remove texlive-base
# or you can run sudo apt remove texlive*
sudo apt remove biber
```


I went to the [document](https://tug.org/texlive/acquire-netinstall.html) and download the [`install-tl-unx.tar.gz`](https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz) file. Then, I extract the file and run the `install-tl` file. With the default setting, it would end up at: `/usr/local/texlive/2025`


<pre>
# Set TeX Live 2025 path
export PATH="/usr/local/texlive/2025/bin/x86_64-linux:$PATH"
export MANPATH="/usr/local/texlive/2025/texmf-dist/doc/man/man5:$MANPATH"
export INFOPATH="/usr/local/texlive/2025/texmf-dist/doc/info:$INFOPATH"
</pre>


After that, **test** the installation by checking the version:

```bash
pdflatex --version
```
<pre>
pdfTeX 3.141592653-2.6-1.40.23 (TeX Live 2025)
kpathsea version 6.4.1
</pre>

### Error

<pre>
INFO - This is Biber 2.20
INFO - Logfile is 'main.blg'
INFO - Reading 'main.bcf'
ERROR - Error: Found biblatex control file version 3.8, expected version 3.11.
This means that your biber (2.20) and biblatex (3.17) versions are incompatible.
See compat matrix in biblatex or biber PDF documentation.
INFO - ERRORS: 1
</pre>

The easiest way is delete the files and compile again.
```bash
rm *.aux *.bbl *.bcf
```

## Citation (Reference)

For the citation, there are few main types you can use in the latex: `biber` and `bibtex`. The `biber` is the newer one and it is recommended to use. The `bibtex` is the older one and it is not recommended to use. The `biber` is more powerful and flexible than the `bibtex`. The `biber` can handle more complex citation styles and it can handle more complex bibliography data. The `bibtex` is more simple and it is easier to use. The `bibtex` is more suitable for the beginners.

bibtex example:

```latex
\documentclass{article}
\begin{document}
    \cite{Ctrax}
    \bibliographystyle{plain}
    \bibliography{main}
\end{document}
```

how to compile the `bibtex` example:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```


biber example:

```latex
\documentclass{article}
\usepackage[backend=biber]{biblatex}
\addbibresource{main.bib}

\begin{document}
    \cite{Ctrax}
    \printbibliography
\end{document}
``` 

how to compile the `biber` example:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```


## Editor

### Prerequisation

```bash
sudo apt install -y texlive texlive-font-utils texlive-pstricks-doc texlive-base texlive-formats-extra texlive-lang-german texlive-metapost texlive-publishers texlive-bibtex-extra texlive-latex-base texlive-metapost-doc texlive-publishers-doc texlive-binaries texlive-latex-base-doc texlive-science texlive-extra-utils texlive-latex-extra texlive-science-doc texlive-fonts-extra texlive-latex-extra-doc texlive-pictures texlive-xetex texlive-fonts-extra-doc texlive-latex-recommended texlive-pictures-doc texlive-fonts-recommended texlive-humanities texlive-lang-english texlive-latex-recommended-doc texlive-fonts-recommended-doc texlive-humanities-doc texlive-luatex texlive-pstricks perl-tk
sudo apt install latexmk
```

### atom
3 latex packages for atom:
- language-latex（latex高亮）
- latex
- pdf-view（可视化显示）

reference: [Violet-Guo](https://blog.csdn.net/violet_echo_0908/article/details/78160273)

![img](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMDA1MDkxNjM4ODYy)
© [Violet-Guo](https://blog.csdn.net/violet_echo_0908/article/details/78160273) 2017


按`Ctrl+Alt+B`預覽
![img](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMDA1MDkyMjA5Mjk2)
© [Violet-Guo](https://blog.csdn.net/violet_echo_0908/article/details/78160273) 2017

### more editors
https://www.zhihu.com/question/19954023
- Lyx
- TexStudio
- Sublime Text
- Emacs

## Online Platform: Overleaf

I found most of my friends are using Overleaf to edit latex online. So, I tired and use it, too. It is good. All codes and methods below are all test under the Overleaf platform.

## Main structure of the Latex

here is the structure of the latex projects:
<pre>
.
├── figures
│   └── example_figure.pdf
├── ldr-article.cls
├── main.bib
├── main.pdf
├── main.tex
├── README.md
└── Rmarkdown.Rmd
</pre>

This directory is an example from the overleaf (ldr-template).

- The figures is a directory for storing the figures.
- The `ldr-article.cls` is for storing all configures which like `cs` for html.
- the `main.bib` is for storing the citations
- the `main.tex` is the place for your main contented.

```latex
\documentclass{ldr-article}
\addbibresource{main.bib}
\title{Titiel is here }
\author{
Karobben 1\\\\
Karobben 2
}

\begin{document}
    \maketitle
    \begin{abstract}
    There are no abstracts. Lol
    \end{abstract}
    \keywords{latex; write a papre}
\section{Introduction}
This is an exmaple for showing the basic structure of a latex
\section{Result}
\subsection{Result 1}
\subsection{Result 2}
\end{document}
```

## Citation

cite: [LondonRob](https://tex.stackexchange.com/questions/102817/setting-up-texmaker-on-ubuntu-biblatex-sty-not-found)
```bash
sudo apt-get install texlive-bibtex-extra biber
```

In latex, there is at least two ways to cite: `cite` and `parencite`. The different between two of them are `parencite` could automatically add parent symbol, "()", for you.

Contents in `main.bib` file:

<pre>
@article{Ctrax,
  title={High-throughput ethomics in large groups of Drosophila},
  author={Branson, Kristin and Robie, Alice A and Bender, John and Perona, Pietro and Dickinson, Michael H},
  journal={Nature methods},
  volume={6},
  number={6},
  pages={451--457},
  year={2009},
  publisher={Nature Publishing Group US New York}
}
@article{CADABRA,
  title={Automated monitoring and analysis of social behavior in Drosophila},
  author={Dankert, Heiko and Wang, Liming and Hoopfer, Eric D and Anderson, David J and Perona, Pietro},
  journal={Nature methods},
  volume={6},
  number={4},
  pages={297--303},
  year={2009},
  publisher={Nature Publishing Group US New York}
}
</pre>


|Latex| Rendered in PDF|
|:-|:-|
|<pre>Citation test:\\\\<br>- Parencite: \parencite{Ctrax} \\\\<br>- Parencite multiple: \parencite{Ctrax, CADABRA} \\\\<br>- Cite: \cite{Ctrax}\\\\<br>- Cite multiple: \cite{Ctrax, CADABRA}\\\\</pre>|<pre>Citation test: <br>- Parencite: (Branson et al.)<br>- Parencite multiple: (Branson et al.; Dankert et al.)<br>- Cite: Branson et al.<br>- Cite multiple: Branson et al.; Dankert et al.<br></pre>|

Different styles for citation:

Find the line `\RequirePackage[style=` in the file of `ldr-article.cls`

| Style             | Approximate Citation Format |
|-------------------|-----------------------------|
| `alphabetic`      | [Bra+09]                    |
| `authortitle`     | Branson et al., "High-throughput ethomics in large groups of Drosophila" |
| `authoryear`      | Branson et al. 2009         |
| `authoryear-icomp`| Branson et al. 2009         |
| `authoryear-comp` | Branson et al. 2009         |
| `numeric`         | [1]                         |
| `numeric-comp`    | [1]                         |
| `reading`         | Branson, Kristin, et al. 2009 |
| `verbose`         | Branson, Kristin, et al. "High-throughput ethomics in large groups of Drosophila." Nature Methods 6.6 (2009): 451-457 |
| `chem-acs`        | (1)                         |
| `phys`            | [1]                         |
| `nejm`            | 1.                          |
| `nature`          | 1.                          |
| `science`         | 1.                          |
| `ieee`            | [1]                         |


### Other tricks

1. **Commenting**: `% this is a invisible comment`
2. **Dealing with special characters**: In LaTeX, some characters are reserved for special commands. If you need to use these characters as they are, you need to escape them using a backslash (`\`). The special characters are: `# $ % ^ & _ { } ~ \`. For example, if you want to write `5%`, you need to write it as `5\%` in LaTeX.
3. **Inserting images**: The graphicx package provides commands to work with images. You can use the `\includegraphics` command to insert an image.
    ```latex
    \usepackage{graphicx}

    \begin{document}
    \includegraphics{filename}
    \end{document}
    ```
4. **Creating tables**: The `tabular` environment can be used to create tables
    ```latex
    \begin{tabular}{|c|c|}
    \hline
    Header 1 & Header 2 \\
    \hline
    Row 1, Col 1 & Row 1, Col 2 \\
    Row 2, Col 1 & Row 2, Col 2 \\
    \hline
    \end{tabular}
    ```
5. **Dealing with large documents**: For large documents like a thesis or a book, you can use `\input{filename}` or `\include{filename}` to add contents from another file. This can help keep your project organized.
6. **Math mode**: LaTeX is widely used for its superior handling of mathematical equations. You can insert an inline mathematical equation like this: `$E=mc^2$`, or a standalone one like this:
    ```latex
    \begin{equation}
    E=mc^2
    \end{equation}
    ```
7. **Hyperlinks**: With the `hyperref` package, you can add hyperlinks to your document.
    ```latex
    \usepackage{hyperref}
    ...
    \href{https://www.example.com}{Link text}
    ```
8. **Referencing**: With LaTeX, you can easily cross-reference figures, tables, sections, etc. For example, when you label a figure using `\label{fig:my_label}` you can reference it with `\ref{fig:my_label}` and it will automatically update the figure number.
Remember, the power of LaTeX comes from the various packages available. When you want to do something specific, there is probably a package that can help you achieve it. Check the documentation of the packages to make full use of their features.




<style>
pre {
  background-color:#38393d;                               color: #5fd381;
}
</style>


