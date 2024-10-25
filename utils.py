import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
import pandas as pd

import sys
sys.path.append("/home/talax/xtof/local/Mod/lib64")
import mod
from mod import *
import glob
import os


# BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

# BOND_NAMES = {t: i for i, t in enumerate(BondType.names.keys())}
# print(f"BOND_NAMES: {BOND_NAMES}")
# print(f"BOND_TYPES: {BOND_TYPES}")

def smarts_to_gml(reaction_smarts_array):
    """
    Convert a SMARTS representation of a chemical reaction 
    to a GML (Graph Modeling Language) representation.

    Parameters:
    - reaction_smarts_array (list): Array or List containing the SMARTS representation 
    of the reaction and the reaction name (string).

    Returns:
    - str: GML format string representing the chemical reaction.
    """

    # Extract SMARTS representation and reaction name from the input array
    reaction_smarts, reaction_name = reaction_smarts_array[0], reaction_smarts_array[1]

    # Generate a Reaction object from the SMARTS using RDKit
    reaction = AllChem.ReactionFromSmarts(reaction_smarts)

    # Initialize the GML representation for the reaction
    gml_rule = {
        "ruleID": f'"{reaction_name}"',
        "left": {"nodes": [], "edges": []},
        "context": {"nodes": [], "edges": []},
        "right": {"nodes": [], "edges": []}
    }

    # Define dictionaries for bond types and charges
    bonds = {1: "-", 2: "=", 3: "#", 1.5: ":"}
    charges = {1: "+", -1: "-"}

    # Extract atom mapping information from reactants and products
    reactant_atoms = {(atom.GetAtomMapNum(), atom.GetSymbol() + charges[atom.GetFormalCharge()] 
                       if atom.GetFormalCharge() != 0 else atom.GetSymbol()) 
                      for reactant in reaction.GetReactants() 
                      for atom in reactant.GetAtoms() if atom.GetAtomMapNum() is not None}
    
    product_atoms = {(atom.GetAtomMapNum(), atom.GetSymbol() + charges[atom.GetFormalCharge()] 
                      if atom.GetFormalCharge() != 0 else atom.GetSymbol())
                     for product in reaction.GetProducts() 
                     for atom in product.GetAtoms() if atom.GetAtomMapNum() is not None}

    # Extract bonds information from reactants and products
    reactant_bonds = {(min(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()), 
                       max(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()), b.GetBondTypeAsDouble()) 
                       for reactant in reaction.GetReactants() 
                       for b in reactant.GetBonds() 
                       if b.GetBeginAtom().GetAtomMapNum() is not None and b.GetEndAtom().GetAtomMapNum() is not None}
    
    product_bonds = {(min(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()), 
                      max(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()), b.GetBondTypeAsDouble()) 
                      for product in reaction.GetProducts() 
                      for b in product.GetBonds() 
                      if b.GetBeginAtom().GetAtomMapNum() is not None and b.GetEndAtom().GetAtomMapNum() is not None}

    # Get unchanged bonds and atoms by taking the intersection of reactant and product bonds and atoms
    unchanged_bonds = reactant_bonds.intersection(product_bonds)
    unchanged_atoms = reactant_atoms.intersection(product_atoms)

    # Sort changed bonds and atoms based on source atom ID
    changed_reactant_bonds = sorted(reactant_bonds - unchanged_bonds, key=lambda x: x[0])
    changed_product_bonds = sorted(product_bonds - unchanged_bonds, key=lambda x: x[0])
    changed_reactant_atoms = sorted(reactant_atoms - unchanged_atoms, key=lambda x: x[0])
    changed_product_atoms = sorted(product_atoms - unchanged_atoms, key=lambda x: x[0])
    unchanged_atoms = sorted(unchanged_atoms, key=lambda x: x[0])
    unchanged_bonds = sorted(unchanged_bonds, key=lambda x: x[0])

    # Add nodes and edges for unchanged bonds and atoms in the context
    for atom_id, label in unchanged_atoms:
        gml_rule["context"]["nodes"].append({
            "id": atom_id,
            "label": label
        })

    for bond in unchanged_bonds:
        source, target, label = bond
        bond_label = bonds[label]
        gml_rule["context"]["edges"].append({
            "source": source,
            "target": target,
            "label": bond_label
        })

    # Add nodes and edges for changed bonds and atoms in the left and right parts of the rule
    for atom_id, label in changed_reactant_atoms:
        gml_rule["left"]["nodes"].append({
            "id": atom_id,
            "label": label
        })

    for source, target, label in changed_reactant_bonds:
        bond_label = bonds[label]
        gml_rule["left"]["edges"].append({
            "source": source,
            "target": target,
            "label": bond_label
        })

    for atom_id, label in changed_product_atoms:
        gml_rule["right"]["nodes"].append({
            "id": atom_id,
            "label": label
        })

    for source, target, label in changed_product_bonds:
        bond_label = bonds[label]
        gml_rule["right"]["edges"].append({
            "source": source,
            "target": target,
            "label": bond_label
        })

    # Generate the GML format string for the rule
    gml_format = f"rule[\n ruleID {gml_rule['ruleID']}"

    for key in ["left", "context", "right"]:
        gml_format += f" \n {key} ["

        if "nodes" in gml_rule[key] and gml_rule[key]["nodes"]:
            for item in gml_rule[key]["nodes"]:
                gml_format += f"""\n   node [ id {item['id']} label "{item['label']}"]"""

        for item in sorted(gml_rule[key]["edges"], key=lambda x: (x['source'],x['target'])):
            gml_format += f"""\n   edge [ source {item['source']} target {item['target']} label "{item['label']}"]"""

        gml_format += "\n ]" 

    gml_format += "\n ]"

    return gml_format


from collections import defaultdict

def smarts_to_gml_v2(reaction_smarts, reaction_name):
    """
    Convert a SMARTS representation of a chemical reaction to a GML representation
    with sorted IDs.
    """
    reaction = AllChem.ReactionFromSmarts(reaction_smarts)
    gml_rule = defaultdict(lambda: {"nodes": set(), "edges": set()})
    gml_rule["ruleID"] = f'"{reaction_name}"'

    bonds = {1: "-", 2: "=", 3: "#", 1.5: ":"}
    charges = {1: "+", -1: "-", 0: ""}

    def get_molecule_info(mol):
        atoms = {(atom.GetAtomMapNum(), atom.GetSymbol() + charges[atom.GetFormalCharge()])
                 for atom in mol.GetAtoms() if atom.GetAtomMapNum()}
        bonds = {(min(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()),
                  max(b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()),
                  b.GetBondTypeAsDouble())
                 for b in mol.GetBonds()
                 if b.GetBeginAtom().GetAtomMapNum() and b.GetEndAtom().GetAtomMapNum()}
        return atoms, bonds

    reactants_info = [get_molecule_info(r) for r in reaction.GetReactants()]
    products_info = [get_molecule_info(p) for p in reaction.GetProducts()]

    reactant_atoms = set().union(*(atoms for atoms, _ in reactants_info))
    reactant_bonds = set().union(*(bonds for _, bonds in reactants_info))
    product_atoms = set().union(*(atoms for atoms, _ in products_info))
    product_bonds = set().union(*(bonds for _, bonds in products_info))

    unchanged_atoms = reactant_atoms & product_atoms
    unchanged_bonds = reactant_bonds & product_bonds

    gml_rule["left"]["nodes"] = reactant_atoms - unchanged_atoms
    gml_rule["left"]["edges"] = reactant_bonds - unchanged_bonds
    gml_rule["right"]["nodes"] = product_atoms - unchanged_atoms
    gml_rule["right"]["edges"] = product_bonds - unchanged_bonds
    gml_rule["context"]["nodes"] = unchanged_atoms
    gml_rule["context"]["edges"] = unchanged_bonds

    def format_section(section_name, section_data):
        nodes = sorted(section_data["nodes"])
        edges = sorted(section_data["edges"])
        result = [f" \n {section_name} ["]
        result.extend(f'   node [ id {id} label "{label}"]' for id, label in nodes)
        result.extend(f'   edge [ source {s} target {t} label "{bonds[l]}"]' for s, t, l in edges)
        result.append(" ]")
        return "\n".join(result)

    sections = ["left", "context", "right"]
    gml_format = [f"rule[\n ruleID {gml_rule['ruleID']}"]
    gml_format.extend(format_section(section, gml_rule[section]) for section in sections)
    gml_format.append("]")

    return "\n".join(gml_format)



# Atomic_List = ["H", "C", "O", "N", "P", "S", "K", "Na","Mg", "Ca", "Fe", "Zn", "Cu", "Mn", "Mo", "Co", "Ni", "Cl",  "I"]
Atomic_List = ["H", "C", "O", "N", "P", "S"]

class AtomicNumberTable(object):
    def __init__(self, atomic_list):
        super(AtomicNumberTable, self).__init__()
        self.atomic_list = atomic_list

    def __len__(self):
        return len(self.atomic_list)
    
    def get_index(self, symbol):
        if symbol not in self.atomic_list:
            return None
        return self.atomic_list.index(symbol)

    def get_symbol(self, index):
        if index < 0 or index >= len(self.atomic_list):
            return None
        return self.atomic_list[index]



def smile_to_data(smile):
    atomic_table = AtomicNumberTable(Atomic_List)
    mol = Chem.AddHs(Chem.MolFromSmiles(smile))
    # assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()
    # print(f"Number of atoms: {N}")
    # print(f"conformers: {mol.GetNumConformers()}")
    # print(f"position: {mol.GetConformer(0).GetPositions()}")
    # pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    atom_idx = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    
    for atom in mol.GetAtoms():
        atom_idx.append(atomic_table.get_index(atom.GetSymbol()))
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)
    print(f"z atomic number shape: {z.shape}    z atomic number: {z}")
    print(f"sp : {sp}  the shape of sp: {len(sp)} \n sp2 : {sp2}  the shape of sp2: {len(sp2)} \n sp3 : {sp3}  the shape of sp3: {len(sp3)} \n aromatic : {aromatic}  the shape of aromatic: {len(aromatic)} \n")
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]
        print(f"bond.GetBondType(): {bond.GetBondType()} \n edge_type: {edge_type} \n")
    print(f"row: {row}")
    print(f"col: {col}")
    edge_index = torch.tensor([row, col], dtype=torch.long)
    print(f"edge index: {edge_index}  edge_index shape: {edge_index.shape}")
    edge_type = torch.tensor(edge_type)
    print(f"edge_type: {edge_type}   edge_type shape: {edge_type.shape}")

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    print(f"perm: {perm}\n edge_index[0]*N: {edge_index[0]*N} \n edge_index[1]: {edge_index[1]} \n edge_index[0]*N+edge_index[1]: {edge_index[0]*N+edge_index[1]} \n edge_index[0]*N+edge_index[1].argsort(): {(edge_index[0]*N+edge_index[1]).argsort()}\n")
    edge_index = edge_index[:, perm]
    print(f"edge_index shape after permutation: {edge_index.shape} \n edge_index after permutation: {edge_index}")
    edge_type = edge_type[perm]
    print(f"edge_type shape after permutation: {edge_type.shape} \n edge_type after permutation: {edge_type}")

    row, col = edge_index
    print(f"row after permutation: {row} \n col after permutaion: {col}")
    hs = (z == 1).to(torch.float32)
    print(f"hs: {hs}\n hs[row]: {hs[row]} ]")
    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()
    print(f"num_hs: {num_hs}")
    x1 = F.one_hot(torch.tensor(atom_idx), num_classes=len(atomic_table))
    print(f"x1 shape: {x1.shape} atom_idx: {atom_idx}")
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],dtype=torch.float).t().contiguous()
    print(f"x2 shape: {x2.shape}") 
    x = torch.cat([x1.to(torch.float), x2], dim=-1)


    data = Data(atom_features=x, edge_index=edge_index, edge_type=edge_type,rdmol=copy.deepcopy(mol), smiles=smile)
 
    return data


def linear_encoding(smiDFS,name):
    if Chem.MolFromSmiles(smiDFS) is not None:
        try:
            output= smiles(smiDFS,name)
        except:
            
            print("Problem with SMILES for Chem:", smiDFS)
            pass
    else:
        try:
            output= graphDFS(smiDFS,name)
        except:
            print("Problem with SMILES for DFS:", smiDFS)
            pass
    return output

def collect_inputGraphs(smi_csv_path):
    input_graphs = []
    df = pd.read_csv(smi_csv_path)
    df.columns = df.columns.str.lower()
    for i in range(df.shape[0]):
        mol_name = df.iloc[i]['name']
        mol_smiles = df.iloc[i]['smiles']

        input_graphs.append(linear_encoding(mol_smiles, mol_name))

    return input_graphs


def collect_rules(rule_gml_path):
    rule_list = []
    rule_gml_path = os.path.join(rule_gml_path, "*.gml")
    files = sorted(glob.glob(rule_gml_path, recursive=True))
    for file in files:
        rule_list.append(mod.ruleGML(file))

    return rule_list

# Function to convert derivation graph to hypergraph representation
def dg_to_hypergraph(dg):
    """
    Converts a derivation graph to a hypergraph representation.

    Parameters:
    - dg (pymod object DerivationGraph:pymod.DG): Input derivation graph.

    Returns:
    List of hypergraph entries, where each entry is a list containing:
    - rule (set): Set of rules associated with the edge.
    - source (set): Set of source vertices connected to the edge.
    - target (set): Set of target vertices connected to the edge.
    """
    # Initialize an empty list to store hypergraph entries
    hypergraph = []

    # Iterate through edges in the derivation graph
    for e in dg.edges:
        # Extract rules, sources, and targets from each edge
        rule = set(r for r in e.rules)
        source = set(s.graph for s in e.sources)
        target = set(t.graph for t in e.targets)

        # Append the hypergraph entry to the list
        hypergraph.append([rule, source, target])

    return hypergraph


# Function to convert hypergraph to a set of hyperedges and vertices
def hg_to_evset(hypergraph):
    """
    Extracts hyperedges and vertices from a hypergraph.

    Parameters:
    - hypergraph (list): List of hypergraph entries, where each entry is a list containing:
      - rule (set): Set of rules associated with the edge.
      - source (set): Set of source vertices connected to the edge.
      - target (set): Set of target vertices connected to the edge.

    Returns:
    Tuple of sets containing:
    - hyperedges_set (set): Set of unique hyperedge IDs.
    - vertices_set (set): Set of unique vertex IDs.
    """
    hyperedges_set = set()
    vertices_set = set()

    for entry in hypergraph:
        # Extract unique hyperedge ID and update sets
        hyperedge_id = {list(entry[0])[0].id}
        # print(f"hyperedge object: {entry[0]} \n hyperedge object next layer : {list(entry[0])} \n {list(entry[0])[0].name}" )
        # print(f"rule left graph: {list(entry[0])[0].LeftGraph} \n rule right graph: {list(entry[0])[0].RightGraph}")
        # print(f"rule left graph name: {list(entry[0])[0].LeftGraph.Edge} \n rule right graph name: {[v for v in list(entry[0])[0].ContextGraph.vertices]}")

        hyperedges_set.update(hyperedge_id)
        vertices_set.update(entry[1])
        vertices_set.update(entry[2])

    return hyperedges_set, vertices_set


# Function to convert hypergraph to a biadjacency matrix
def hypergraph_to_biadjacency(hypergraph):
    """
    Converts a hypergraph to a biadjacency matrix.

    Parameters:
    - hypergraph (list): List of hypergraph entries, where each entry is a list containing:
      - rule (set): Set of rules associated with the edge.
      - source (set): Set of source vertices connected to the edge.
      - target (set): Set of target vertices connected to the edge.

    Returns:
    Tuple containing:
    - Q_hyperedges_vertices (torch.Tensor): Biadjacency matrix mapping hyperedges to vertices.
    - Q_vertices_hyperedges (torch.Tensor): Biadjacency matrix mapping vertices to hyperedges.
    - index_map_vertices (dict): Dictionary mapping vertex IDs to matrix indices.
    - index_map_hyperedges (dict): Dictionary mapping hyperedge IDs to matrix indices.
    """
    # Extract hyperedges and vertices from the hypergraph list
    hyperedges_set, vertices_set = hg_to_evset(hypergraph)

    # Sort hyperedges and vertices for consistent indexing
    hyperedges_set = sorted(hyperedges_set)
    vertices_set = sorted(vertices_set, key=lambda x: x.id)

    # Create dictionaries to map vertex and hyperedge indices to matrix indices
    index_map_vertices = {v: i for i, v in enumerate(vertices_set)}
    index_map_hyperedges = {h: i for i, h in enumerate(hyperedges_set)}

    # Initialize matrices for directed bipartite graph with infinity values
    Q_vertices_hyperedges = torch.full((len(vertices_set), len(hyperedges_set)), float('inf'))
    Q_hyperedges_vertices = torch.full((len(hyperedges_set), len(vertices_set)), float('inf'))

    # Fill in matrices based on the hypergraph structure
    for entry in hypergraph:
        # Extract hyperedge ID from the first rule in the set
        hyperedge_id = list(entry[0])[0].id

        # Update matrices based on source and target vertices
        for v in entry[1]:
            Q_vertices_hyperedges[index_map_vertices[v], index_map_hyperedges[hyperedge_id]] = 1

        for v in entry[2]:
            Q_hyperedges_vertices[index_map_hyperedges[hyperedge_id], index_map_vertices[v]] = 1

    return Q_hyperedges_vertices, Q_vertices_hyperedges, index_map_vertices, index_map_hyperedges


from typing import List

def encode(molecule: str) -> str:
    """
    Encodes the given molecule string after replacing non-parsable parts.
    
    Args:
        molecule (str): The molecular formula containing certain abbreviations.
    
    Returns:
        str: The encoded molecular string with non-parsable parts replaced.
    """
    # Dictionary mapping of non-parsable abbreviations to lanthanides (or placeholder values)
    non_parsable_to_lanthanides = {
        'Ad': 'La',   # Ad -> Lanthanum (La) as a placeholder
        'CoA': 'Ce',  # CoA -> Cerium (Ce) as a placeholder
        'NAD': 'Pr',  # NAD -> Praseodymium (Pr)
        # 'NAD+': 'Nd'  # NAD+ -> Neodymium (Nd)
    }

    # Replace non-parsable abbreviations in the molecule string
    for key, replacement in non_parsable_to_lanthanides.items():
        molecule = molecule.replace(f'{key}', f'{replacement}')
    
    return molecule

def decode(molecule: str) -> str:
    """
    Decodes the given molecule string by replacing lanthanides with their original abbreviations.


    Args:
        molecule (str): The encoded molecular string containing lanthanides.
    """
    lanthanides_to_non_parsable = {
        'La': 'Ad',  
        'Ce': 'CoA', 
        'Pr': 'NAD',  
       
    }

    # Replace non-parsable abbreviations in the molecule string
    for key, replacement in lanthanides_to_non_parsable.items():
        molecule = molecule.replace(f'{key}', f'{replacement}')
    
    return molecule
    
  

