rule[
 ruleID "R5: SAM + CO2 = SAM-CO2H" 
 left [
   edge [ source 5 target 11 label "-"] 
   edge [ source 8 target 9 label "="]
 ] 
 context [
   node [ id 1 label "C"]
   node [ id 2 label "S+"]
   node [ id 3 label "C"]
   node [ id 4 label "C"]
   node [ id 5 label "C"]
   node [ id 6 label "N"]
   node [ id 7 label "Ad"]
   node [ id 8 label "O"]
   node [ id 9 label "C"]
   node [ id 10 label "O"]
   node [ id 11 label "H"]
   edge [ source 1 target 2 label "-"]
   edge [ source 2 target 3 label "-"]
   edge [ source 2 target 7 label "-"]
   edge [ source 3 target 4 label "-"]
   edge [ source 4 target 5 label "-"]
   edge [ source 5 target 6 label "-"]
   edge [ source 9 target 10 label "="]
 ] 
 right [
   edge [ source 5 target 9 label "-"]
   edge [ source 8 target 9 label "-"]
   edge [ source 8 target 11 label "-"]
 ]
]

