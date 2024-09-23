SHELL=/bin/bash
summary/all.tex: gen
	echo '\summaryDGHyper{\texttt{dg\_0}}{out/011_dg_0_11100}{\texttt{out/011\_dg\_0\_11100}}' >> summary/all.tex
	echo -n '\summaryGraph' >> summary/all.tex
	echo '{\texttt{g\_\{0\}}}{\summaryGraphFile{out/012_g_0_11300110}{\texttt{out/012\_g\_0\_11300110}}}{}' >> summary/all.tex
	echo -n '\summaryGraph' >> summary/all.tex
	echo '{\texttt{g\_\{1\}}}{\summaryGraphFile{out/013_g_1_11300110}{\texttt{out/013\_g\_1\_11300110}}}{}' >> summary/all.tex
summary.pdf: summary/all.tex
	/home/talax/xtof/local/Mod/bin/../bin/mod_post --mode compileSummary
include summary/MakefileImpl
include summary/MakefileDep
include summary/MakefileAllTarget
include summary/MakefileClean
