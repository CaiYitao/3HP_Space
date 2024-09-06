rule[
 ruleID "R1: (S)-lactate + NAD+ = pyruvate + NADH + H+" 
 left [
   node [ id 9 label "NAD+"]
   node [ id 7 label "H"]
   edge [ source 2 target 3 label "-"]
   edge [ source 3 target 7 label "-"]
   edge [ source 2 target 8 label "-"]
] 
 context [
   node [ id 1 label "C"]
   node [ id 2 label "C"]
   node [ id 3 label "O"]
   node [ id 4 label "C"]
   node [ id 5 label "O"]
   node [ id 6 label "O"]
   node [ id 8 label "H"]
   edge [ source 1 target 2 label "-"]
   edge [ source 2 target 4 label "-"]
   edge [ source 4 target 5 label "="]
   edge [ source 4 target 6 label "-"]

 ] 
 right [
   node [ id 9 label "NAD"]
   node [ id 7 label "H+"]
   edge [ source 2 target 3 label "="]
   edge [ source 8 target 9 label "-"]
 ]
]

