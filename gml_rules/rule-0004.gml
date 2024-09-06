rule[
 ruleID "R4: Acyl-CoA + H2O <=> CoA + Carboxylate" 
 left [
   edge [ source 1 target 3 label "-"]
   edge [ source 4 target 6 label "-"]
 ] 
 context [
   node [ id 1 label "C"]
   node [ id 2 label "O"]
   node [ id 3 label "S"]
   node [ id 4 label "O"]
   node [ id 5 label "CoA"]
   node [ id 6 label "H"]
   node [ id 7 label "H"]
   edge [ source 1 target 2 label "="]
   edge [ source 3 target 5 label "-"]
   edge [ source 4 target 7 label "-"]
 ] 
 right [
   edge [ source 1 target 4 label "-"]
   edge [ source 3 target 6 label "-"]
 ]
]

