rule[
 ruleID "R2: Ac-aldehyde + CoA + NAD+ = Ac-CoA + NADH + H+" 
 left [
   node [ id 5 label "NAD+"]
   node [ id 8 label "H"]
   edge [ source 2 target 6 label "-"]
   edge [ source 4 target 8 label "-"]
] 
 context [
   node [ id 1 label "C"]
   node [ id 2 label "C"]
   node [ id 3 label "O"]
   node [ id 4 label "S"]
   node [ id 6 label "H"]
   node [ id 7 label "CoA"]
   edge [ source 1 target 2 label "-"]
   edge [ source 2 target 3 label "="]
   edge [ source 4 target 7 label "-"]
 ] 
 right [
   node [ id 5 label "NAD"]
   node [ id 8 label "H+"]
   edge [ source 2 target 4 label "-"]
   edge [ source 5 target 6 label "-"]
 ]
]

