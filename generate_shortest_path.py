import numpy as np  

import sys
import subprocess
sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *
# from utils import smarts_to_gml_with_mapping
import random
from rdkit.Chem import AllChem,Draw
import numpy as np
from rdkit import Chem
# from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
import pandas as pd 
from utils import smarts_to_gml
from shortest_path_bipartite_ import *


def main():
    asp  = smiles('OC(=O)CC(N)C(=O)O', 'ASP')
    lys  = smiles('NCCCCC(N)C(=O)O', 'LYS')
    akg  = smiles('OC(=O)C(=O)CCC(=O)O', 'AKG')  
    pyr  = smiles('CC(=O)C(=O)O', 'PYR')
    lac  = smiles('CC(O)C(=O)O', 'LAC')
    aca  = smiles('CC=O', 'AcA')
    ac   = smiles('CC(=O)O', 'Ac')
    mal  = smiles('OC(=O)C(O)CC(=O)O','MAL')
    co2  = smiles('O=C=O', 'CO2')
    pi   = smiles('OP(=O)(O)O', 'Pi')
    ppi  = smiles('OP(=O)(O)OP(=O)(O)O', 'PPi')
    nh3  = smiles('[NH3]', 'NH3')
    h2o  = smiles('O', 'H2O')
    hco3 = smiles('OC(=O)O', 'HCO3') 
    hp = smiles('[H+]', 'H+')

    coa     = graphDFS('[CoA]S[H]', 'CoA')
    sam     = graphDFS('[Ad][S+](C)CCCN', 'SAM')
    nadh    = graphDFS('[NAD][H]', 'NADH')
    nadplus = graphDFS('[NAD+]', 'NAD+')
    atp     = graphDFS('[Ad]OP(=O)(O)OP(=O)(O)OP(=O)(O)O', 'ATP')
    accoa   = graphDFS('[CoA]SC(=O)C', 'Ac-CoA')
    laccoa  = graphDFS('[CoA]SC(=O)C(O)C', 'LAC-CoA')

    # for m in inputGraphs:
    #     m.print()

    # rules(s)
    r1  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0001.gml')
    r2  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0002.gml')
    # r3  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0003.gml')
    # r4  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0004.gml')
    # r5  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0005.gml')
    # r6  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0006.gml')
    # r7  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0007.gml')
    # r8  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0008.gml')
    # r9  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0009.gml')
    # r10 = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0010.gml')
    # r11 = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0011.gml')
    strat = (addSubset(inputGraphs) >> repeat[1](inputRules))
    dg = DG()
    dg.build().execute(strat)
    # dg.build().apply(inputGraphs, r1,onlyProper=False)
    dg.print()
    mol_dict = get_mol_dict(dg)
    print(mol_dict)
    print(f"Number of vertices: {len(mol_dict)}")
    hypergraph = dg_to_hypergraph(dg)

    Q1,Q2,V,H = hypergraph_to_biadjacency(hypergraph)
    print(f"Q1 shape: {Q1.shape}")
    print(f"Q2 shape: {Q2.shape}")
    # print(V)
    print(H)

 
    P1_2m1, P2_2m1, Q1_2m1, Q2_2m1, D = torgansin_zimmerman(Q1,Q2)
    print(f"P1_2m1 : {P1_2m1.shape}  Q1_2m1 : {Q1_2m1.shape}  Q2_2m1 : {Q2_2m1.shape}  P2_2m1 : {P2_2m1.shape}")
    print(f"P1_2m1 : {P1_2m1}")
    print(f"Q1_2m1 : {Q1_2m1}")
    print(f"Q2_2m1 : {Q2_2m1}")
    print(f"P2_2m1 : {P2_2m1}")
    D = D.numpy()
    P2_2m1 = P2_2m1.numpy()/2
    df_P2_2m1 = pd.DataFrame(P2_2m1, index = mol_dict.keys(), columns = mol_dict.keys() )
    df_P2_2m1.to_csv("shortest_path_mol.csv")
    print(f"H : {H}")
    print(f"mol_dict : {mol_dict}")
    rule_mol_dict = H | mol_dict
    print(f"rule_mol_dict : {rule_mol_dict}")
    df_D = pd.DataFrame(D, index = rule_mol_dict.keys(), columns = rule_mol_dict.keys())
    # df_D.to_csv("shortest_path_ap.csv")



    # np.savetxt("shortest_path.csv", D, delimiter=",")

    

if __name__ == "__main__":
    main()