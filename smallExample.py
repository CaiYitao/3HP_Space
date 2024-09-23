from env import Env,init_dg
from utils import *
import pandas as pd
import sys
# sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *
import random
import subprocess
from collections import namedtuple
# Define the named tuple
from rdkit import Chem
from collections import namedtuple

# Define the Molecule named tuple
Molecule = namedtuple('Molecule', ['name', 'smiles'])

# Define the target molecules as a list of named tuples
target_molecules_pathway_A = [
    Molecule('D-glucose', 'C(C1C(C(C(C(O1)O)O)O)O)O'),
    Molecule('glucose-6-phosphate', 'C(C1C(C(C(C(O1)O)O)O)O)OP(=O)(O)O'),
    Molecule('D-glucose-6-phosphate', 'C(C(C(C(C(C=O)O)O)O)O)OP(=O)(O)O'),
    Molecule('L-myo-Inositol 1-phosphate', 'C1(C(C(C(C(C1O)O)OP(=O)(O)O)O)O)O'),
    Molecule('inositol', 'C1(C(C(C(C(C1O)O)O)O)O)O')
]

target_molecules_pathway_B = [
    Molecule('glucose 1-phosphate', 'C(C1C(C(C(C(O1)OP(=O)(O)O)O)O)O)O'),
    Molecule('D-glucose-6-phosphate', 'C(C(C(C(C(C=O)O)O)O)O)OP(=O)(O)O'),
    Molecule('L-myo-Inositol 1-phosphate', 'C1(C(C(C(C(C1O)O)OP(=O)(O)O)O)O)O'),
    Molecule('inositol', 'C1(C(C(C(C(C1O)O)O)O)O)O')
]

def main(pathway='A'):
    h2o = smiles('O', 'H2O')
    ATP = graphDFS('[ADP]P(=O)(O)O', 'ATP')
    phosphate = smiles('OP(=O)(O)O', 'Phosphate')
    cellobiose = smiles('C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O', 'Cellobiose')

    # mol_path0 = "/home/mescalin/yitao/Documents/Code/CRN_IMA/known_mol.csv"
    # reactants = collect_inputGraphs(mol_path0)
    cofactors = [h2o, ATP, phosphate]
    rules = collect_rules('/home/mescalin/yitao/Documents/Code/CRN_IMA/gml_rules')

    init_mol = cofactors
    dg_init = init_dg(init_mol)
    current_dg = dg_init
    print('')
    new_mol = cellobiose

    if pathway == 'A':
        target_molecules = target_molecules_pathway_A
        selected_rules = [0, 1,5, 3, 4]
    elif pathway == 'B':
        target_molecules = target_molecules_pathway_B
        selected_rules = [0, 2, 3, 4]
    else:
        raise ValueError("Invalid pathway selected. Choose 'A' or 'B'.")

    for rule_index, target in zip(selected_rules, target_molecules):
        rule = rules[rule_index]
        mol, dg = apply_reaction(current_dg, rule, new_mol, target)

        current_dg = dg
        new_mol = mol
        print(f"new_mol: {new_mol}")
       
        if mol is None:
          print(f"Reaction did not produce a valid molecule for rule {rule_index} and target {target.name}")
          break
    
    p = GraphPrinter()
    p.setReactionDefault()
    p.withIndex = True
    for r in inputRules:
        r.print(p)
    post.flushCommands()
    subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])  

def apply_reaction(old_dg, rule, new_mol, target_mol):
    print("rule:",rule.name)
    mstrat = (rightPredicate[lambda dg: all(g.vLabelCount('C') <= 66 for g in dg.right)](rule))
    # dg = DG()
    dg = DG(labelSettings=LabelSettings(LabelType.Term, LabelRelation.Specialisation))

    with dg.build() as b:
        if old_dg.edges is not None:
            for e in old_dg.edges:
                print(f"e: {e}")
                print(f"e.targets: {e.targets}")
                b.addHyperEdge(e)
        print("old_dg.graphDatabase in addUniverse before reaction:",old_dg.graphDatabase)
        res = b.execute(addUniverse(old_dg.graphDatabase) >> addSubset(new_mol) >> mstrat)
        print("molecule:",new_mol)
        # res = b.execute(addSubset(smiles("CC(=O)")) >> mstrat)
        print('universe:', res.universe)
        print('subsets:', res.subset)

    print(f'graphDatabase after reaction excution: {dg.graphDatabase}')
    
    products = []

    if dg.edges is not None:
        for e in dg.edges:    
            product = set(t.graph for t in e.targets)
            products.append(product)

    print(f"products: {products}")

    dg.print()

 

    for product in products:
        for mol_graph in product:
            print(f"mol_graph: {mol_graph.linearEncoding}")
            print(f"target_mol: {target_mol.smiles}")
            print(f"iso check: {is_isomorphic(mol_graph, linear_encoding(target_mol.smiles, target_mol.name))}")
            if is_isomorphic(mol_graph, linear_encoding(target_mol.smiles, target_mol.name)):
                output = mol_graph
                print(f"output: {output}")
                return output, dg


    return None, dg

def is_isomorphic(mol1, mol2):
    if not (is_smiles(mol1) and is_smiles(mol2)):
        return mol1.isomorphism(mol2) == 1
    else:
        return rdkit_isomorphism_check(mol1, mol2)

def rdkit_isomorphism_check(mol1, mol2):
    rdkit_mol1 = Chem.MolFromSmiles(mol1)
    rdkit_mol2 = Chem.MolFromSmiles(mol2)
    return rdkit_mol1.HasSubstructMatch(rdkit_mol2) and rdkit_mol2.HasSubstructMatch(rdkit_mol1)

def is_smiles(encoding):
    try:
        mol = Chem.MolFromSmiles(encoding, sanitize=True)
        return mol is not None
    except Exception:
        return False

if __name__ == "__main__":
    main(pathway='A')  # or main(pathway='B') to choose the pathway
