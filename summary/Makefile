SHELL=/bin/bash
summary/all.tex: gen
	echo '\summaryRule{R1: (S)-lactate + NAD+ = pyruvate + NADH + H+}{\summaryRuleSpan{out/001_r_0_11300110}{out/001\_r\_0\_11300110}}{}{}{out/002_r_0_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R2: Ac-aldehyde + CoA + NAD+ = Ac-CoA + NADH + H+}{\summaryRuleSpan{out/004_r_1_11300110}{out/004\_r\_1\_11300110}}{}{}{out/005_r_1_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R3: L-Aspartate + 2-Oxoglutarate \textless{}=\textgreater{} Oxaloacetate + L-Glutamate}{\summaryRuleSpan{out/007_r_2_11300110}{out/007\_r\_2\_11300110}}{}{}{out/008_r_2_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R4: Acyl-CoA + H2O \textless{}=\textgreater{} CoA + Carboxylate}{\summaryRuleSpan{out/010_r_3_11300110}{out/010\_r\_3\_11300110}}{}{}{out/011_r_3_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R5: SAM + CO2 = SAM-CO2H}{\summaryRuleSpan{out/013_r_4_11300110}{out/013\_r\_4\_11300110}}{}{}{out/014_r_4_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R6: L-malate = fumarate + H2O}{\summaryRuleSpan{out/016_r_5_11300110}{out/016\_r\_5\_11300110}}{}{}{out/017_r_5_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R7: Lactoyl-CoA \textless{}=\textgreater{} Propenoyl-CoA + H2O}{\summaryRuleSpan{out/019_r_6_11300110}{out/019\_r\_6\_11300110}}{}{}{out/020_r_6_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R8: L-Aspartate \textless{}=\textgreater{} Fumarate + Ammonia}{\summaryRuleSpan{out/022_r_7_11300110}{out/022\_r\_7\_11300110}}{}{}{out/023_r_7_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R9: L-lysine = (3S)-3,6-diaminohexanoate}{\summaryRuleSpan{out/025_r_8_11300110}{out/025\_r\_8\_11300110}}{}{}{out/026_r_8_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R10: ATP + Acetate + CoA \textless{}=\textgreater{} AMP + PPi + Ac-CoA}{\summaryRuleSpan{out/028_r_9_11300110}{out/028\_r\_9\_11300110}}{}{}{out/029_r_9_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{R11: ATP + Ac-CoA + HCO3- = ADP + malonyl-CoA + Pi}{\summaryRuleSpan{out/031_r_10_11300110}{out/031\_r\_10\_11300110}}{}{}{out/032_r_10_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{L-glutamate + NAD+ + H2O = 2-oxoglutarate + NADH + NH3 \{ir\}|| ec num:1.4.1.2}{\summaryRuleSpan{out/034_r_11_11300110}{out/034\_r\_11\_11300110}}{}{}{out/035_r_11_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{pyruvate = acetaldehyde + CO2 \{r\}|| ec num: 4.1.1.1}{\summaryRuleSpan{out/037_r_12_11300110}{out/037\_r\_12\_11300110}}{}{}{out/038_r_12_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{cellobiose + phosphate = alpha-D-glucose 1-phosphate + D-glucose}{\summaryRuleSpan{out/040_r_13_11300110}{out/040\_r\_13\_11300110}}{}{}{out/041_r_13_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{ATP+glucose= ADP+Glucose 6-phosphate}{\summaryRuleSpan{out/043_r_14_11300110}{out/043\_r\_14\_11300110}}{}{}{out/044_r_14_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{alpha-D-glucose 1-phosphate = D-glucose 6-phosphate}{\summaryRuleSpan{out/046_r_15_11300110}{out/046\_r\_15\_11300110}}{}{}{out/047_r_15_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{D-Glucose 6-phosphate = L-myo-Inositol 1-phosphate}{\summaryRuleSpan{out/049_r_16_11300110}{out/049\_r\_16\_11300110}}{}{}{out/050_r_16_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{inositol 1-phosphate + H2O = myo-inositol + phosphate}{\summaryRuleSpan{out/052_r_17_11300110}{out/052\_r\_17\_11300110}}{}{}{out/053_r_17_constraints.tex}' >> summary/all.tex
	echo '\summaryRule{glucose= D-Glucose}{\summaryRuleSpan{out/055_r_18_11300110}{out/055\_r\_18\_11300110}}{}{}{out/056_r_18_constraints.tex}' >> summary/all.tex
	echo '\summaryDGHyper{\texttt{dg\_0}}{out/676_dg_0_11100}{\texttt{out/676\_dg\_0\_11100}}' >> summary/all.tex
summary.pdf: summary/all.tex
	/home/talax/xtof/local/Mod/bin/../bin/mod_post --mode compileSummary
include summary/MakefileImpl
include summary/MakefileDep
include summary/MakefileAllTarget
include summary/MakefileClean
