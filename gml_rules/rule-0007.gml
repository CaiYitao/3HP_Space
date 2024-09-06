rule[
 ruleID "R7: Lactoyl-CoA <=> Propenoyl-CoA + H2O" 
 left [
   edge [ source 1 target 2 label "-"]
   edge [ source 1 target 8 label "-"]
   edge [ source 2 target 3 label "-"]
 ] 
 context [
   node [ id 1 label "C"]
   node [ id 2 label "C"]
   node [ id 3 label "O"]
   node [ id 4 label "C"]
   node [ id 5 label "O"]
   node [ id 6 label "S"]
   node [ id 7 label "CoA"]
   node [ id 8 label "H"]
   node [ id 9 label "H"]
   edge [ source 2 target 4 label "-"]
   edge [ source 3 target 9 label "-"]
   edge [ source 4 target 5 label "="]
   edge [ source 4 target 6 label "-"]
   edge [ source 6 target 7 label "-"]
 ] 
 right [
   edge [ source 1 target 2 label "="]
   edge [ source 3 target 8 label "-"]
 ]
]

