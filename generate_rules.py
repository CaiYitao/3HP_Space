import numpy as np  

import sys
import subprocess
# sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *
# from utils import smarts_to_gml_with_mapping
import random
from rdkit.Chem import AllChem,Draw
import numpy as np
from rdkit import Chem
# from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
import pandas as pd 
import torch



""" 

    Xe : indicate CoA group
    U : indicate NAD group
    Rn : indicate R group
"""

RULES  = [["[CH3:1][C:2]([H:8])([O:3][H:7])[C:4](=[O:5])[OH:6].[*+:9]>>[CH3:1][C:2](=[O:3])[C:4](=[O:5])[OH:6].[H+:7].[H:8][*:9]","(S)-lactate + NAD+ = pyruvate + NADH + H+","1.1.1.27"],
      ["[CH3:1][C:2]([H:6])=[O:3].[*+:5].[S:4]([H:8])[*:7]>>[CH3:1][C:2](=[O:3])[S:4][*:7].[H+:8].[*:5][H:6]","acetaldehyde + CoA + NAD+ = acetyl-CoA + NADH + H+","1.2.1.10"],
      ["[H:20][N:1]([H:21])[C:2]([CH2:3][C:4](=[O:5])[OH:6])([H:22])([C:7](=[O:8])[OH:9]).[O:10]=[C:11]([OH:12])[CH2:13][CH2:14][C:15](=[O:16])[C:17](=[O:18])[OH:19]>>[C:2]([CH2:3][C:4](=[O:5])[OH:6])([C:7](=[O:8])[OH:9])=[O:16].[H:20][N:1]([H:21])[C:15]([CH2:14][CH2:13][C:11](=[O:10])[OH:12])([H:22])[C:17](=[O:18])[OH:19]","L-Aspartate + 2-Oxoglutarate <=> Oxaloacetate + L-Glutamate","2.6.1.1"],
      ["[CH3:1](=[O:2])[S:3][*:5].[H:7][O:4][H:6]>>[CH3:1](=[O:2])[O:4][H:7].[H:6][S:3][*:5]","Acyl-CoA + H2O <=> CoA + Carboxylate","3.1.2.20"],
      ["[CH3:1][S+:2]([CH2:3][CH2:4][CH:5]([H:11])[NH2:6])[*:7].[O:8]=[C:9]=[O:10]>>[CH3:1][S+:2]([CH2:3][CH2:4][C@H:5]([NH2:6])[C:9](=[O:10])[O:8][H:11])[*:7]","S-adenosyl-L-methionine = (5-deoxy-5-adenosyl)(3-aminopropyl)methylsulfonium salt + CO2","4.1.1.50"],
      ["[O:1]=[C:2]([OH:3])[CH:4]([H:11])[C@H:5]([O:6][H:10])[C:7](=[O:8])[OH:9]>>[O:1]=[C:2]([OH:3])/[CH:4]=[CH:5]/[C:7](=[O:8])[OH:9].[H:10][O:6][H:11]","fumarate + H2O = L-malate","4.2.1.2"],
      ["[H:8][CH2:1][CH:2]([O:3][H:9])[C:4](=[O:5])[S:6][*:7]>>[CH2:1]=[CH:2][C:4](=[O:5])[S:6][*:7].[H:8][O:3][H:9]","Lactoyl-CoA <=> Propenoyl-CoA + H2O","4.2.1.54"],
      ["[H:13][NH2:1][C@@H:2]([CH:3]([H:10])[C:4](=[O:5])[OH:6])[C:7](=[O:8])[OH:9]>>[CH:2](=[CH:3]/[C:4](=[O:5])[OH:6])\\[C:7](=[O:8])[OH:9].[NH2:1][H:10]","L-Aspartate <=> Fumarate + Ammonia","4.3.1.1"],
      ["[NH2:1][CH2:2][CH2:3][CH2:4][CH:5]([H:11])[C@H:6]([NH2:7])[C:8](=[O:9])[OH:10]>>[NH2:1][CH2:2][CH2:3][CH2:4][C@H:5]([NH2:7])[CH:6]([H:11])[C:8](=[O:9])[OH:10]","L-lysine = (3S)-3,6-diaminohexanoate","5.4.3.2"],
      ["[CH3:1][C:2](=[O:3])[OH:4].[Xe:18][S:5][H:19].[CH3:6][O:7][P:8](=[O:9])([OH:10])[O:11][P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:20])[OH:21]>>[CH3:1][C:2](=[O:3])[S:5][Xe:18].[OH:4][P:8]([O:7][CH3:6])(=[O:9])[OH:10].[O:11]([H:19])[P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:20])[OH:21]","ATP + Acetate + CoA <=> AMP + Diphosphate + Acetyl-CoA","6.2.1.1"],
      ["[H:23][CH2:1][C:2](=[O:3])[S:4][Xe:24].[CH3:5][O:6][P:7](=[O:8])([OH:9])[O:10][P:11](=[O:12])([OH:13])[O:14][P:15](=[O:16])([OH:17])[OH:18].[O:19]=[C:20]([OH:21])[OH:22]>>[CH2:1]([C:2](=[O:3])[S:4][Xe:24])[C:20](=[O:19])[OH:22].[CH3:5][O:6][P:7](=[O:8])([OH:9])[O:10][P:11](=[O:12])([OH:13])[OH:21].[H:23][O:14][P:15](=[O:16])([OH:17])[OH:18]","ATP + acetyl-CoA + HCO3- = ADP + malonyl-CoA + phosphate ","6.4.1.2"]
      ]

# RULES =[["[CH3:1][C:2](=[O:3])[S:4][Xe:1000].[NH2:5][c:6]1[cH:7][cH:8][c:9]([C:10](=[O:11])[OH:12])[cH:13][cH:14]1>>[S:4]([H:1001])[Xe:1000].[CH3:1][C:2](=[O:3])[NH:5][c:6]1[cH:7][cH:8][c:9]([C:10](=[O:11])[OH:12])[cH:13][cH:14]1","acetyl-CoA + 4-aminobenzoic acid = CoA + N-acetyl-4-aminobenzoate","2.3.1.50"],
#       ["[CH3:1][C:2](=[O:3])[S:4][Xe:1000].[NH2:5][c:6]1[cH:7][cH:8][c:9]([OH:10])[cH:11][cH:12]1>>[S:4]([H:1001])[Xe:1000].[CH3:1][C:2](=[O:3])[NH2:5][c:6]1[cH:7][cH:8][c:9]([OH:10])[cH:11][cH:12]1","acetyl-CoA + 4-amimophenol = CoA + paracetamol","2.3.1.118"],
# ["[H+].[H:1000][U:1001].[NH2:1][c:2]1[cH:3][cH:4][c:5]([C:6](=[O:7])[OH:8])[cH:9][cH:10]1.[O:11]=[O:12]>>[C:6](=[O:7])=[O:8].[U+:1001].[OH2:12].[NH2:1][c:2]1[cH:3][cH:4][c:5]([OH:11])[cH:9][cH:10]1","4-aminobenzoate + NAD(P)H + H+ + O2 = 4-aminophenol + NAD(P)+ + H2O + CO2","1.14.13.27"],]


def generate_rules(RULES):
    RULES_ARR = np.array(RULES)
    rules_set = RULES_ARR[:,:2]
    RULES_DF = pd.DataFrame(RULES,columns=["mapped","orig_rxn_text","EC"])
    RULES_DF 


    rules = []

    for r in rules_set:
        rule = smarts_to_gml(r)
        rules.append(rule)
        # print(rule)
    
    return rules


def main():
    rules = generate_rules(RULES)
    for i, r in enumerate(rules):

        with open(f"/home/mescalin/yitao/Documents/Code/3HP/rules/rules_{i}.gml","w") as f:

            f.write(r)


if __name__ == "__main__":
    main()
