from env import Env,init_dg
from utils import *
import pandas as pd
import sys
sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *
import random
import subprocess

target_molecule="CC(=O)C(=O)O"

def main():
    # h2o  = smiles('O', 'H2O')
    coa     = graphDFS('[CoA]S[H]', 'CoA')
    # # sam     = graphDFS('[Ad][S+](C)CCCN', 'SAM')
    nadh    = graphDFS('[NAD][H]', 'NADH')
    nadplus = graphDFS('[NAD+]', 'NAD+')
    # # atp     = graphDFS('[Ad]OP(=O)(O)OP(=O)(O)OP(=O)(O)O', 'ATP')
    accoa   = graphDFS('[CoA]SC(=O)C', 'Ac-CoA')
    laccoa  = graphDFS('[CoA]SC(=O)C(O)C', 'LAC-CoA')
    lac = smiles('CC(O)C(=O)O', 'LAC')
    # pyr = smiles('CC(=O)C(=O)O', 'PYR')

    mol_path0 = "/home/mescalin/yitao/Documents/Code/CRN_IMA/known_mol.csv"

    reactants: list = collect_inputGraphs(mol_path0)
    cofactors: list = [coa,nadh,nadplus,accoa,laccoa]
    # mol_pool = {"mols":reactants,"cofactors":cofactors}

    # rules_path = "/home/mescalin/yitao/Documents/Code/3HPspace"
    # rules = collect_rules(rules_path)
    rule = ruleGML('/home/mescalin/yitao/Documents/Code/3HPspace/rule-0001.gml')

    init_mol = [lac]
    dg_init = init_dg(init_mol)
    apply_reaction(dg_init,rule,cofactors)


def apply_reaction(init_dg,rule,cofactors) -> list:

    cofactor = [cofactors[2]]

    mstrat = (rightPredicate[lambda dg: all(g.vLabelCount('C') <= 88 for g in dg.right)](rule) )

    dg = DG()

    with dg.build() as b:
        # b.execute(addUniverse(self.current_dg.graphDatabase))
        if init_dg.edges is not None:
            for e in init_dg.edges:
                print(f"e: {e}")
                print(f"e.targets: {e.targets}")
                b.addHyperEdge(e)
  
        res = b.execute(addSubset(init_dg.graphDatabase + cofactor) >> mstrat)

    products =[]

    if dg.edges is not None:
            for e in dg.edges:    
                 product = set(t.graph for t in e.targets)
      
                 products.append(product)

    print(f"products: {products}")
    dg.print()
    # flush summary file handle
    post.flushCommands()
    # generate summary/summery.pdf
    subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])

    for product in products:
        for smi in product:
            print(f"smi: {smi}")
            if is_isomorphic(smi, linear_encoding(target_molecule,'pyruvate')):
                output = smi.linearEncoding 
                print(f"output: {output}")
                return output

    return None


def is_isomorphic(mol1, mol2):
    if not (is_smiles(mol1.linearEncoding) and is_smiles(mol2.linearEncoding)):
        return mol1.isomorphism(mol2)==1
    else:
        return rdkit_isomorphism_check(mol1, mol2)

def rdkit_isomorphism_check(mol1, mol2):
    # Use RDKit for SMILES isomorphism checks (example implementation)
    smiles1 = mol1.linearEncoding
    smiles2 = mol2.linearEncoding

    rdkit_mol1 = Chem.MolFromSmiles(smiles1) 
    rdkit_mol2 = Chem.MolFromSmiles(smiles2) 

    return rdkit_mol1.HasSubstructMatch(rdkit_mol2) and rdkit_mol2.HasSubstructMatch(rdkit_mol1)

def is_smiles(encoding):
    try:
        mol = Chem.MolFromSmiles(encoding, sanitize=True)
        return mol is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()

    



    # for i in range(episodes):
    #     rule_action = random.randint(0,len(rules)-1)
    #     print(f"rule action: R{rule_action+1} at the {i+1}th episode")
    #     # ma = random.randint(0,known_mol.shape[0]-1)
    #     mol_action = random.randint(0,len(mol_pool)-1)
    #     cof_action = random.randint(0,len(cofactors)-1)
    #     # print(f"molecule action: {known_mol.iloc[ma]['name']} at the {i+1}th episode")
    #     print(f"molecule action: {source_pool[mol_action].name} at the {i+1}th episode")
    #     print(f"cofactor action: {cofactors[cof_action].name} at the {i+1}th episode")
    #     print(f"rule action: {rules[rule_action].name} at the {i+1}th episode")
    #     # dg_new,_,_= env.step(rule_action,(mol_action,cof_action))
    #     dg_new,_,_= env.step(rule_action,mol_action)
    #     # if i % 20 == 0:
    #     #     env.render()
    #     action.append([mol_action,cof_action,rule_action])
