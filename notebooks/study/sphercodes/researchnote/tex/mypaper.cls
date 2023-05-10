\LoadClass[10pt, a4paper]{article}
\makeatletter
\def\input@path{{./}{./include/}{./sections}{./figures}}
\makeatother
%% TEXT ENCODING %%
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
%% COLOR %%
%\usepackage[table]{xcolor}
%\usepackage[breakable]{tcolorbox}
%% MARGINS %%
\usepackage[margin=3.00cm]{geometry}
%% BIBLIOGRAPHY %%
%% FIGURES AND GRAPHICS %%
\usepackage{graphicx}
\graphicspath{{./figures/}{./}{../figures}}
% Ensure that by default, figures have no caption (until we provide a
% proper Figure object with a Caption API and a way to capture that
% in the conversion process - todo).
\usepackage{booktabs}
\usepackage{pdflscape} %make landascape orientation available
\usepackage{afterpage}
\usepackage{capt-of}
\usepackage{makecell}
\usepackage[Export]{adjustbox} % Used to constrain images to a maximum size
\adjustboxset{max size={\linewidth}{\paperheight}}
%% ALGORITHMS %%
\usepackage{algorithm, algorithmic}
%% AMS %%
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{amsfonts}
\usepackage{xfrac}
\usepackage{amsthm}
\usepackage{amssymb}
\usepackage{esint}

\setlength{\parskip}{7pt}
\setlength{\parindent}{0pt}
\setlength{\headheight}{15pt}

\newtheorem{thm}{Theorem}
\newtheorem{corol}[thm]{Corollary}
\newtheorem{prop}[thm]{Proposition}
\newtheorem{conjecture}[thm]{Conjecture}
\theoremstyle{definition}
\newtheorem{defi}[thm]{Definition}
\newtheorem{remark}[thm]{Remark}
\newtheorem{example}[thm]{Example}
\newtheorem{model}{Model}
\newtheorem{assumption}{Assumption}
\newtheorem{approximation}{Approximation}

%% MISCELLANEOUS %%
\usepackage{fancyhdr, float, enumerate, longtable, pdfpages, booktabs, upquote, mathtools, setspace, tikz}
\sloppy %Prevents overflowing on lines 


