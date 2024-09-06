rule[
 ruleID "pyruvate = acetaldehyde + CO2   {r}  || ec num:  4.1.1.1"
 
 left [
   edge [ source 2 target 4 label "-"]
   edge [ source 4 target 6 label "-"]
   edge [ source 6 target 7 label "-"]
 ]
 
 context [
   node [ id 1 label "C"]
   node [ id 2 label "C"]
   node [ id 3 label "O"]
   node [ id 4 label "C"]
   node [ id 5 label "O"]
   node [ id 6 label "O"]
   node [ id 7 label "H"]
   edge [ source 1 target 2 label "-"]
   edge [ source 2 target 3 label "="]
   edge [ source 4 target 5 label "="]
 ]
 
 right [
   edge [ source 2 target 7 label "-"]
   edge [ source 4 target 6 label "="]
 ]
]