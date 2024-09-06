rule[
 ruleID "R6: L-malate = fumarate + H2O" 
 left [
   edge [ source 4 target 5 label "-"]
   edge [ source 4 target 11 label "-"]
   edge [ source 5 target 6 label "-"]
 ] 
 context [
   node [ id 1 label "O"]
   node [ id 2 label "C"]
   node [ id 3 label "O"]
   node [ id 4 label "C"]
   node [ id 5 label "C"]
   node [ id 6 label "O"]
   node [ id 7 label "C"]
   node [ id 8 label "O"]
   node [ id 9 label "O"]
   node [ id 10 label "H"]
   node [ id 11 label "H"]
   edge [ source 1 target 2 label "="]
   edge [ source 2 target 3 label "-"]
   edge [ source 2 target 4 label "-"]
   edge [ source 5 target 7 label "-"]
   edge [ source 6 target 10 label "-"]
   edge [ source 7 target 8 label "="]
   edge [ source 7 target 9 label "-"]
 ] 
 right [
   edge [ source 4 target 5 label "="]
   edge [ source 6 target 11 label "-"]
 ]
]

