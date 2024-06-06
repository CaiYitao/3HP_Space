from env import ChemicalReactionEnv,init_dg,Env
import sys
import subprocess
sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *
import numpy as np
import random
import pandas as pd
from utils import *


REPLACEMENT_DICT = {"[CoA]": "CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCC",
                    "[Ad]": "O(O(N(C32=NC=N)C=C3N=CN2(O1)C)C)CC1C",
                    "[NAD+]":"C1=CC(=C[N+](=C1)C2C(C(C(O2)COP(=O)([O-])OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)O)O)O)O)C(=O)N",
                    "[NAD][H]": "C1C=CN(C=C1C(=O)N)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)O)O)O)O"}

def main():
    # asp  = smiles('OC(=O)CC(N)C(=O)O', 'ASP')
    # lys  = smiles('NCCCCC(N)C(=O)O', 'LYS')
    # akg  = smiles('OC(=O)C(=O)CCC(=O)O', 'AKG')  
    # pyr  = smiles('CC(=O)C(=O)O', 'PYR')
    # lac  = smiles('CC(O)C(=O)O', 'LAC')
    # aca  = smiles('CC=O', 'AcA')
    # ac   = smiles('CC(=O)O', 'Ac')
    # mal  = smiles('OC(=O)C(O)CC(=O)O','MAL')
    # co2  = smiles('O=C=O', 'CO2')
    # pi   = smiles('OP(=O)(O)O', 'Pi')
    # ppi  = smiles('OP(=O)(O)OP(=O)(O)O', 'PPi')
    # nh3  = smiles('[NH3]', 'NH3')
    h2o  = smiles('O', 'H2O')
    # hco3 = smiles('OC(=O)O', 'HCO3') 
    # hp = smiles('[H+]', 'H+')


    coa     = graphDFS('[CoA]S[H]', 'CoA')
    # # sam     = graphDFS('[Ad][S+](C)CCCN', 'SAM')
    nadh    = graphDFS('[NAD][H]', 'NADH')
    nadplus = graphDFS('[NAD+]', 'NAD+')
    # # atp     = graphDFS('[Ad]OP(=O)(O)OP(=O)(O)OP(=O)(O)O', 'ATP')
    accoa   = graphDFS('[CoA]SC(=O)C', 'Ac-CoA')
    laccoa  = graphDFS('[CoA]SC(=O)C(O)C', 'LAC-CoA')

    r1  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0001.gml')
    r2  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0002.gml')
    r3  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0003.gml')
    r4  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0004.gml')
    r5  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0005.gml')
    r6  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0006.gml')
    r7  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0007.gml')
    r8  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0008.gml')
    r9  = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0009.gml')
    r10 = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0010.gml')
    r11 = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0011.gml')

    rules = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11]
    # mols_dict = {"smiles":['OC(=O)CC(N)C(=O)O','NCCCCC(N)C(=O)O','OC(=O)C(=O)CCC(=O)O','CC(=O)C(=O)O','CC(O)C(=O)O','CC=O','CC(=O)O','OC(=O)C(O)CC(=O)O','O=C=O','OP(=O)(O)O','OP(=O)(O)OP(=O)(O)O','[NH3]','OC(=O)O','[H+]'],
    #              "name":['ASP','LYS','AKG','PYR','LAC','AcA','Ac','MAL','CO2','Pi','PPi','NH3','HCO3','H+']}

    # known_mol = pd.DataFrame(mols_dict)
    # known_mol.to_csv("known_mol.csv",index=False)
    # mol_path = "/home/mescalin/yitao/Documents/Code/CRN_IMA/known_mol.csv"
    # # known_mol = [asp,lys,akg,pyr,lac,aca,ac,mal,co2,pi,ppi,nh3,hco3,hp,coa,nadh,nadplus,accoa,laccoa]
    # known_mol = collect_inputGraphs(mol_path)
    # print(f"known_mol: {known_mol}")
    mol_path0 = "/home/mescalin/yitao/Documents/Code/CRN_IMA/known_mol.csv"
    mol_path = "/home/mescalin/yitao/Documents/Code/CRN_IMA/namesmilesinchi_pretty.csv"
    # known_mol = [asp,lys,akg,pyr,lac,aca,ac,mal,co2,pi,ppi,nh3,hco3,hp,coa,nadh,nadplus,accoa,laccoa]
    known_mol0 = collect_inputGraphs(mol_path0)
    # print(f"known_mol0: {known_mol0}")
    known_mol = collect_inputGraphs(mol_path)
    # print(f"known_mol: {known_mol}")
    mol_pool = known_mol0 + known_mol
    # print(f"known_mol within know_mol0: {mol_pool}")
    print(f"len known_mol: {len(mol_pool)}")
    rules_path = "/home/mescalin/yitao/Documents/Code/3HPspace"
    rules = collect_rules(rules_path)
    target_molecule="OCC1OC(O)C(C(C1O)O)O"
    init_mol = [h2o,coa,accoa,nadh,nadplus]
    dg_init = init_dg(init_mol)
    # print("inputGraphs",inputGraphs)
    # print("dg init graphDatabase", dg_init.graphDatabase)
    reward_type = "RXN_Happen_Or_Not"
    # reward_type = "RXN_Distance_to_Target" 
    env = Env(dg_init,rules,target_molecule,mol_pool, reward=reward_type)
    target_molecule="OCC1OC(O)C(C(C1O)O)O"
    init_mol = [h2o,coa,accoa,nadh,nadplus]
    dg_init = init_dg(init_mol)
    # env = ChemicalReactionEnv(dg_init,rules,target_molecule,REPLACEMENT_DICT,mol_path)
    # env = Env(dg_init,rules,target_molecule,REPLACEMENT_DICT,mol_path)
    # env = Env(dg_init,rules,target_molecule,known_mol)

    episodes = 2000
    random.seed(133)
    action = []
    reward = []
    Done = []
    for i in range(episodes):
        rule_action = random.randint(0,len(rules)-1)
        print(f"rule action: R{rule_action+1} at the {i+1}th episode")
        # ma = random.randint(0,known_mol.shape[0]-1)
        mol_action = random.randint(0,len(mol_pool)-1)
        # print(f"molecule action: {known_mol.iloc[ma]['name']} at the {i+1}th episode")
        print(f"molecule action: {mol_pool[mol_action].name} at the {i+1}th episode")
        dg_new,r,done= env.step(rule_action,mol_action)
        # if i % 20 == 0:
        #     env.render()
        action.append([mol_action,rule_action])
        reward.append(r)
        Done.append(done)

    print(f"action: {action}")
    print(f"reward: {reward}")
    print(f"Done: {Done}")
    # dg_reset = env.reset()
    env.render()



if __name__ == "__main__":
    main()





