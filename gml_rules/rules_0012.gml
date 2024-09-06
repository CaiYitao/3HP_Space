rule[
 ruleID "L-glutamate + NAD+ + H2O = 2-oxoglutarate + NADH + NH3  {ir} || ec num: 1.4.1.2"
 
 left [
   node [ id 1 label "NAD+"]
   node [ id 14 label "H"]
   edge [ source 2 target 3 label "-"]
   edge [ source 3 target 15 label "-"]
   edge [ source 12 target 13 label "-"]
   edge [ source 12 target 14 label "-"]
 ]
 
 context [
   node [ id 2 label "N"]
   node [ id 3 label "C"]
   node [ id 4 label "C"]
   node [ id 5 label "C"]
   node [ id 6 label "C"]
   node [ id 7 label "O"]
   node [ id 8 label "O"]
   node [ id 9 label "C"]
   node [ id 10 label "O"]
   node [ id 11 label "O"]
   node [ id 12 label "O"]
   node [ id 13 label "H"]
   node [ id 15 label "H"]
   edge [ source 3 target 4 label "-"]
   edge [ source 3 target 9 label "-"]
   edge [ source 4 target 5 label "-"]
   edge [ source 5 target 6 label "-"]
   edge [ source 6 target 7 label "="]
   edge [ source 6 target 8 label "-"]
   edge [ source 9 target 10 label "="]
   edge [ source 9 target 11 label "-"]
 ]
 
 right [
   node [ id 1 label "NAD"]
   node [ id 14 label "H+"]
   edge [ source 1 target 15 label "-"]
   edge [ source 2 target 13 label "-"]
   edge [ source 3 target 12 label "="]
 ]
]