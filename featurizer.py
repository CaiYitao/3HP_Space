import torch
from torch_geometric.data import Data
from abc import ABC, abstractmethod
import copy
import json
import numpy as np

from torch_scatter import scatter
# from torch_geometric.utils import scatter
import rdkit.Chem as Chem
from rdkit.Chem import DataStructs, rdmolops
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, Batch
from torch_geometric.data.hypergraph_data import HyperGraphData
from torch_geometric.loader import DataLoader
# from transformers import BertTokenizer, T5Tokenizer
from utils import *
from encoder import GraphEncoder


class BaseFeaturizer(ABC):
    def __init__(self):
        super(BaseFeaturizer, self).__init__()
    
    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError
    

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1   
    
def parse_smidfs(smi_dfs):
    replacements= {"CoA": "*","Ad": "*","NAD": "*"}
    if Chem.MolFromSmiles(smi_dfs) is not None:
        return Chem.MolFromSmiles(smi_dfs)
    else:
        for k,v in replacements.items():
            smi_dfs = smi_dfs.replace(k,v)
            print(f"smi_dfs after wildcard: {smi_dfs}")
        return Chem.MolFromSmiles(smi_dfs)
                   


class MolGraphFeaturizer(BaseFeaturizer):

    allowable_features = {
        'possible_atomic_num_list':       list(range(1, 119)) + ['misc'],
        'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
        'possible_chirality_list':        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list':    [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            'misc'
        ],
        'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
        'possible_is_aromatic_list':      [False, True],
        'possible_is_in_ring_list':       [False, True],
        'possible_bond_type_list':        [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            'misc'
        ],
        'possible_bond_dirs':             [  # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ],
        'possible_bond_stereo_list':      [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY,
        ], 
        'possible_is_conjugated_list':    [False, True]
    }

    def __init__(self):
        super(MolGraphFeaturizer, self).__init__()

    def __call__(self, smi_dfs):
        print(f"smi_dfs: {smi_dfs}")
        mol = parse_smidfs(smi_dfs)
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            # if self.config["name"] in ["ogb", "unimap"]:
            atom_feature = [
                safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                safe_index(self.allowable_features['possible_chirality_list'],atom.GetChiralTag()),
                safe_index(self.allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                safe_index(self.allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                safe_index(self.allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                safe_index(self.allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                safe_index(self.allowable_features['possible_hybridization_list'], atom.GetHybridization()),
                safe_index(self.allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
                self.allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                self.allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
   
            # print(f"atom symbol: {atom.GetSymbol()} {atom.GetAtomicNum()} \n atom_feature: {atom_feature}")
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
 

        # bonds
        if len(mol.GetBonds()) <= 0:  # mol has no bonds
            num_bond_features = 3 
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, num_bond_features), dtype=torch.long)
        else:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                # if self.config["name"] in ["ogb", "unimap"]:
                edge_feature = [
                    safe_index(self.allowable_features['possible_bond_type_list'], bond.GetBondType()),
                    self.allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()),
                    self.allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                ]

                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

        
            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
     

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data
    



class RuleGraphFeaturizer(BaseFeaturizer):
    allowable_features = {
        'possible_atomic_num_list':       list(range(1, 119)) + ['misc'],
        'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
        'possible_chirality_list':        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list':    [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            'misc'
        ],
        'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
        'possible_is_aromatic_list':      [False, True],
        'possible_is_in_ring_list':       [False, True],
        'possible_bond_type_list':        [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            'misc'
        ],
        'possible_bond_dirs':             [  # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ],
        'possible_bond_stereo_list':      [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY,
        ], 
        'possible_is_conjugated_list':    [False, True]
    }
    charges = {2:"2+",1: "+",0:"", -1: "-",-2: "2-"}  
    rule_dict = {"R1: (S)-lactate + NAD+ = pyruvate + NADH + H+":"[CH3:1][C:2]([H:8])([O:3][H:7])[C:4](=[O:5])[OH:6].[*+:9]>>[CH3:1][C:2](=[O:3])[C:4](=[O:5])[OH:6].[H+:7].[H:8][*:9]",
                 "R2: Ac-aldehyde + CoA + NAD+ = Ac-CoA + NADH + H+":"[CH3:1][C:2]([H:6])=[O:3].[*+:5].[S:4]([H:8])[*:7]>>[CH3:1][C:2](=[O:3])[S:4][*:7].[H+:8].[*:5][H:6]",
                 "R3: L-Aspartate + 2-Oxoglutarate <=> Oxaloacetate + L-Glutamate":"[H:20][N:1]([H:21])[C:2]([CH2:3][C:4](=[O:5])[OH:6])([H:22])([C:7](=[O:8])[OH:9]).[O:10]=[C:11]([OH:12])[CH2:13][CH2:14][C:15](=[O:16])[C:17](=[O:18])[OH:19]>>[C:2]([CH2:3][C:4](=[O:5])[OH:6])([C:7](=[O:8])[OH:9])=[O:16].[H:20][N:1]([H:21])[C:15]([CH2:14][CH2:13][C:11](=[O:10])[OH:12])([H:22])[C:17](=[O:18])[OH:19]",
                 "R4: Acyl-CoA + H2O <=> CoA + Carboxylate":"[C:1](=[O:2])[S:3][*:5].[H:7][O:4][H:6]>>[C:1](=[O:2])[O:4][H:7].[H:6][S:3][*:5]",
                 "R5: SAM + CO2 = SAM-CO2H":"[CH3:1][S+:2]([CH2:3][CH2:4][CH:5]([H:11])[NH2:6])[*:7].[O:8]=[C:9]=[O:10]>>[CH3:1][S+:2]([CH2:3][CH2:4][C@H:5]([NH2:6])[C:9](=[O:10])[O:8][H:11])[*:7]",
                 "R6: L-malate = fumarate + H2O":"[O:1]=[C:2]([OH:3])[CH:4]([H:11])[C@H:5]([O:6][H:10])[C:7](=[O:8])[OH:9]>>[O:1]=[C:2]([OH:3])/[CH:4]=[CH:5]/[C:7](=[O:8])[OH:9].[H:10][O:6][H:11]",
                 "R7: Lactoyl-CoA <=> Propenoyl-CoA + H2O":"[H:8][CH2:1][CH:2]([O:3][H:9])[C:4](=[O:5])[S:6][*:7]>>[CH2:1]=[CH:2][C:4](=[O:5])[S:6][*:7].[H:8][O:3][H:9]",
                 "R8: L-Aspartate <=> Fumarate + Ammonia":"[NH2:1][C@@H:2]([CH:3]([H:10])[C:4](=[O:5])[OH:6])[C:7](=[O:8])[OH:9]>>[CH:2](=[CH:3]/[C:4](=[O:5])[OH:6])\\[C:7](=[O:8])[OH:9].[NH2:1][H:10]",
                 "R9: L-lysine = (3S)-3,6-diaminohexanoate":"[NH2:1][CH2:2][CH2:3][CH2:4][CH:5]([H:11])[C@H:6]([NH2:7])[C:8](=[O:9])[OH:10]>>[NH2:1][CH2:2][CH2:3][CH2:4][C@H:5]([NH2:7])[CH:6]([H:11])[C:8](=[O:9])[OH:10]",
                 "R10: ATP + Acetate + CoA <=> AMP + PPi + Ac-CoA":"[CH3:1][C:2](=[O:3])[OH:4].[*:18][S:5][H:19].[CH3:6][O:7][P:8](=[O:9])([OH:10])[O:11][P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:20])[OH:21]>>[CH3:1][C:2](=[O:3])[S:5][*:18].[OH:4][P:8]([O:7][CH3:6])(=[O:9])[OH:10].[O:11]([H:19])[P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:20])[OH:21]",
                 "R11: ATP + Ac-CoA + HCO3- = ADP + malonyl-CoA + Pi":"[H:23][CH2:1][C:2](=[O:3])[S:4][*:24].[CH3:5][O:6][P:7](=[O:8])([OH:9])[O:10][P:11](=[O:12])([OH:13])[O:14][P:15](=[O:16])([OH:17])[OH:18].[O:19]=[C:20]([OH:21])[OH:22]>>[CH2:1]([C:2](=[O:3])[S:4][*:24])[C:20](=[O:19])[OH:22].[CH3:5][O:6][P:7](=[O:8])([OH:9])[O:10][P:11](=[O:12])([OH:13])[OH:21].[H:23][O:14][P:15](=[O:16])([OH:17])[OH:18]"}
    
    def __init__(self):
        super(RuleGraphFeaturizer, self).__init__()



    def __call__(self, rxn_name):
        print(f"rxn_name: {rxn_name}")
        rxn_smarts = self.rule_dict[rxn_name]
        if isinstance(rxn_smarts, str):
            reaction = AllChem.ReactionFromSmarts(rxn_smarts)
        else:
            reaction = rxn_smarts

        # Extract reactants, intermediates, and products from the reaction
        reactants = reaction.GetReactants()
        products = reaction.GetProducts()

        # Extract atoms and bonds from reactants and products
        reactant_atoms, reactant_bonds = self.extract_atoms_and_bonds(reactants)
        product_atoms, product_bonds = self.extract_atoms_and_bonds(products)
        # print(f"reactant_atoms: {reactant_atoms} number of reactant atoms: {len(reactant_atoms)}")  
        # print(f"reactant_bonds: {reactant_bonds}")
        # print(f"product_atoms: {product_atoms}")
        # print(f"product_bonds: {product_bonds}")
        self.num_atoms = len(reactant_atoms)
        # Determine unchanged atoms and bonds
        unchanged_atom_idxs = reactant_atoms.keys() & product_atoms.keys()
        unchanged_atoms = {idx: reactant_atoms[idx] for idx in unchanged_atom_idxs}
        unchanged_bond_keys = reactant_bonds.keys() & product_bonds.keys()
        unchanged_bonds = {bond_key: reactant_bonds[bond_key] for bond_key in unchanged_bond_keys}
        # print(f"unchanged_atom_idxs: {unchanged_atom_idxs}")
        # print(f"unchanged_atoms: {unchanged_atoms}")
        # print(f"unchanged_bond_keys: {unchanged_bond_keys}")
        # print(f"unchanged_bonds: {unchanged_bonds}")
        # Determine changed atoms and bonds
        changed_reactant_bonds = {idx: reactant_bonds[idx] for idx in reactant_bonds if idx not in unchanged_bond_keys}
        changed_product_bonds = {idx: product_bonds[idx] for idx in product_bonds if idx not in unchanged_bond_keys}
        changed_reactant_atoms = {idx: reactant_atoms[idx] for idx in reactant_atoms if idx not in unchanged_atom_idxs}
        changed_product_atoms = {idx: product_atoms[idx] for idx in product_atoms if idx not in unchanged_atom_idxs}
        # print(f"changed_reactant_bonds: {changed_reactant_bonds}")
        # print(f"changed_product_bonds: {changed_product_bonds}")
        # print(f"changed_reactant_atoms: {changed_reactant_atoms}")
        # print(f"changed_product_atoms: {changed_product_atoms}")
        # Convert changed atoms and bonds to PyTorch Geometric Data objects
        changed_reactant_data = self.convert_to_data(changed_reactant_atoms, changed_reactant_bonds)
        changed_product_data = self.convert_to_data(changed_product_atoms, changed_product_bonds)

        # Convert unchanged atoms and bonds to PyTorch Geometric Data objects
        unchanged_data = self.convert_to_data(unchanged_atoms, unchanged_bonds)

        # Concatenate node features
        x = torch.cat([changed_reactant_data.x, unchanged_data.x, changed_product_data.x], dim=0)

        # Offset edge indices
        offset_unchanged = changed_reactant_data.num_nodes 
        offset_product = offset_unchanged + unchanged_data.num_nodes
        unchanged_data.edge_index += offset_unchanged
        changed_product_data.edge_index += offset_product

        # Concatenate edge indices and attributes
        edge_index = torch.cat([changed_reactant_data.edge_index, unchanged_data.edge_index, changed_product_data.edge_index], dim=1)
        edge_attr = torch.cat([changed_reactant_data.edge_attr, unchanged_data.edge_attr, changed_product_data.edge_attr], dim=0)


        # Create one-hot membership features 
        reactant_membership = torch.tensor([[1, 0, 0]] * changed_reactant_data.num_nodes, dtype=torch.float)
        context_membership = torch.tensor([[0, 1, 0]] * unchanged_data.num_nodes, dtype=torch.float)
        product_membership = torch.tensor([[0, 0, 1]] * changed_product_data.num_nodes, dtype=torch.float)
        membership = torch.cat([reactant_membership, context_membership, product_membership])
        x = torch.cat([x, membership], dim=1)  # Append to node features

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def extract_atoms_and_bonds(self, molecules):
        atoms = {}
        bonds = {}

        for mol in molecules:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() is not None:
                    # print(f"atom: {atom.GetSymbol()} \n atom symbol + charges {atom.GetSymbol() + self.charges[atom.GetFormalCharge()]}")
                    atoms[(atom.GetAtomMapNum(), atom.GetSymbol() + self.charges[atom.GetFormalCharge()] 
                        if atom.GetFormalCharge() != 0 else atom.GetSymbol()) ] = atom

            for bond in mol.GetBonds():
                if bond.GetBeginAtom().GetAtomMapNum() is not None and bond.GetEndAtom().GetAtomMapNum() is not None:
                    bond_key = (min(bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()), 
                       max(bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()), bond.GetBondTypeAsDouble())
                    bonds[bond_key] = bond 


        # print(f"returned atoms: {atoms}")
        # print(f"returned bonds: {bonds}")
        return atoms, bonds

    def convert_to_data(self, atoms, bonds):
        atom_features = []
        edge_index = []
        edge_attr = []

        for atom_idx, atom in atoms.items():
            atom.UpdatePropertyCache() 
            atom_feature = [
                safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                safe_index(self.allowable_features['possible_chirality_list'], atom.GetChiralTag()),
                safe_index(self.allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                safe_index(self.allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                safe_index(self.allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                safe_index(self.allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                safe_index(self.allowable_features['possible_hybridization_list'], atom.GetHybridization()),
                safe_index(self.allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
                self.allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                self.allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
                atom.GetAtomMapNum()
            ]
            atom_features.append(atom_feature)

        for bond_key, bond in bonds.items():
            # print(f"bond_key: {bond_key}")
            start, end,_ = bond_key
            edge_index.append([start, end])
            bond_feature = [
                safe_index(self.allowable_features['possible_bond_type_list'], bond.GetBondType()),
                self.allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()),
                self.allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
            edge_attr.append(bond_feature)

        x = torch.tensor(atom_features, dtype=torch.long)

        m,n= self.num_atoms,11
        # print(f"m: {m}", f"n: {n}")
        # print(f"x: {x}")
        x = self.resize_and_map_tensor(x,m,n)
        # print(f"mapped atom features: {x}")
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()-1  # 1-based to 0-based index
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data


    def resize_and_map_tensor(self,A, M, N):
        # Check if A has the desired shape
       
        if A.numel() == 0:
            new_tensor = torch.zeros((M, N), dtype=A.dtype, device=A.device)
            indices = torch.arange(M, dtype=new_tensor.dtype).unsqueeze(1) +1 # Shape: (M, 1)

            # Scatter the indices into the last column of the tensor
            new_tensor.scatter_(1, torch.tensor([[N - 1]], dtype=torch.long).expand(M, 1), indices)
            
        else:
            assert A.shape[0] <= M , f"The atoms take part in the reaction should be smaller than the total atoms number: {M}"

            # Create a new tensor of shape (M, N) filled with zeros
            new_tensor = torch.zeros((M, N), dtype=A.dtype, device=A.device)
            # print(f"new_tensor shape: {new_tensor.shape}")
            # print(f"A shape: {A.shape} A: {A}")
            # divide the new tensor into two parts one is the atom features and the other is the mapnumber as mapping idx
            feat, map_idx = A, A[:, -1]
            # print(f"feat: {feat}")  
             # Reshape map_idx to match the dimensions of feat and change the map_idx to 0-based index
            map_idx = map_idx.view(-1, 1)-1
            # print(f"map_idx: {map_idx}")
            # Map the atom features into the new tensor 
     
            # scatter(feat, map_idx.view(-1, 1), dim=0, out=new_tensor)
            scatter(feat, map_idx, dim=0, out=new_tensor)
            # out = scatter(feat, map_idx.view(-1, 1), dim=1)

        # return out
        return new_tensor
    




class HyperGraphFeaturizer(BaseFeaturizer):
    rule_dict = {"R1: (S)-lactate + NAD+ = pyruvate + NADH + H+":"[CH3:1][C:2]([H:8])([OH:3])[C:4](=[O:5])[OH:6].[*+:9]>>[CH3:1][C:2](=[O:3])[C:4](=[O:5])[OH:6].[H+:7].[H:8][*:9]",
                 "R2: Ac-aldehyde + CoA + NAD+ = Ac-CoA + NADH + H+":"[CH3:1][C:2]([H:6])=[O:3].[*+:5].[S:4]([H:8])[*:7]>>[CH3:1][C:2](=[O:3])[S:4][*:7].[H+:8].[*:5][H:6]",
                 "R3: L-Aspartate + 2-Oxoglutarate <=> Oxaloacetate + L-Glutamate":"[H:20][N:1]([H:21])[C:2]([CH2:3][C:4](=[O:5])[OH:6])([H:22])([C:7](=[O:8])[OH:9]).[O:10]=[C:11]([OH:12])[CH2:13][CH2:14][C:15](=[O:16])[C:17](=[O:18])[OH:19]>>[C:2]([CH2:3][C:4](=[O:5])[OH:6])([C:7](=[O:8])[OH:9])=[O:16].[H:20][N:1]([H:21])[C:15]([CH2:14][CH2:13][C:11](=[O:10])[OH:12])([H:22])[C:17](=[O:18])[OH:19]",
                 "R4: Acyl-CoA + H2O <=> CoA + Carboxylate":"[C:1](=[O:2])[S:3][*:5].[H:7][O:4][H:6]>>[C:1](=[O:2])[O:4][H:7].[H:6][S:3][*:5]",
                 "R5: SAM + CO2 = SAM-CO2H":"[CH3:1][S+:2]([CH2:3][CH2:4][CH:5]([H:11])[NH2:6])[*:7].[O:8]=[C:9]=[O:10]>>[CH3:1][S+:2]([CH2:3][CH2:4][C@H:5]([NH2:6])[C:9](=[O:10])[O:8][H:11])[*:7]",
                 "R6: L-malate = fumarate + H2O":"[O:1]=[C:2]([OH:3])[CH:4]([H:11])[C@H:5]([O:6][H:10])[C:7](=[O:8])[OH:9]>>[O:1]=[C:2]([OH:3])/[CH:4]=[CH:5]/[C:7](=[O:8])[OH:9].[H:10][O:6][H:11]",
                 "R7: Lactoyl-CoA <=> Propenoyl-CoA + H2O":"[H:8][CH2:1][CH:2]([O:3][H:9])[C:4](=[O:5])[S:6][*:7]>>[CH2:1]=[CH:2][C:4](=[O:5])[S:6][*:7].[H:8][O:3][H:9]",
                 "R8: L-Aspartate <=> Fumarate + Ammonia":"[NH2:1][C@@H:2]([CH:3]([H:10])[C:4](=[O:5])[OH:6])[C:7](=[O:8])[OH:9]>>[CH:2](=[CH:3]/[C:4](=[O:5])[OH:6])\\[C:7](=[O:8])[OH:9].[NH2:1][H:10]",
                 "R9: L-lysine = (3S)-3,6-diaminohexanoate":"[NH2:1][CH2:2][CH2:3][CH2:4][CH:5]([H:11])[C@H:6]([NH2:7])[C:8](=[O:9])[OH:10]>>[NH2:1][CH2:2][CH2:3][CH2:4][C@H:5]([NH2:7])[CH:6]([H:11])[C:8](=[O:9])[OH:10]",
                 "R10: ATP + Acetate + CoA <=> AMP + PPi + Ac-CoA":"[CH3:1][C:2](=[O:3])[OH:4].[*:18][S:5][H:19].[CH3:6][O:7][P:8](=[O:9])([OH:10])[O:11][P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:20])[OH:21]>>[CH3:1][C:2](=[O:3])[S:5][*:18].[OH:4][P:8]([O:7][CH3:6])(=[O:9])[OH:10].[O:11]([H:19])[P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:20])[OH:21]",
                 "R11: ATP + Ac-CoA + HCO3- = ADP + malonyl-CoA + Pi":"[H:23][CH2:1][C:2](=[O:3])[S:4][*:24].[CH3:5][O:6][P:7](=[O:8])([OH:9])[O:10][P:11](=[O:12])([OH:13])[O:14][P:15](=[O:16])([OH:17])[OH:18].[O:19]=[C:20]([OH:21])[OH:22]>>[CH2:1]([C:2](=[O:3])[S:4][*:24])[C:20](=[O:19])[OH:22].[CH3:5][O:6][P:7](=[O:8])([OH:9])[O:10][P:11](=[O:12])([OH:13])[OH:21].[H:23][O:14][P:15](=[O:16])([OH:17])[OH:18]"}
    
    def __init__(self,cfg):
        super(HyperGraphFeaturizer, self).__init__()

        self.mol_featurizer = MolGraphFeaturizer()
        self.rule_featurizer = RuleGraphFeaturizer()
        self.mol_encoder = GraphEncoder(cfg,rule_graph=False)
        self.rule_encoder = GraphEncoder(cfg,rule_graph=True)
        self.config = cfg

    def __call__(self, dg):
        vertices_to_idx = {}
        mol_attr_data= []
        for idx, vertex in enumerate(dg.vertices):
            print(f"vertex.graph: {vertex.graph} ; vertex.graph.linearEncoding: {vertex.graph.linearEncoding}")
            vertices_to_idx[vertex.graph] = idx
            mol_attr_data.append(self.mol_featurizer(vertex.graph.linearEncoding))
        # print(f"mol_attr_data: {mol_attr_data}")
        hypergraph = dg_to_hypergraph(dg)
        hyperedge_name = []
        hyperedge_attr_data = []
        for idx, hyperedge in enumerate(hypergraph):
            rule = list(hyperedge[0])[0]
            # print(f"rule: {rule} rule.name: {rule.name}")
            hyperedge_name.append(rule.name)
            # hyperedge_attr_data.append(list(self.rule_featurizer(self.rule_dict[rule.name])))
            rule_data = self.rule_featurizer(rule.name)
            hyperedge_attr_data.append(rule_data)

        # print(f"hyperedge_attr_data: {hyperedge_attr_data}")
        # num_vertices = len(dg.vertices)
        # num_hyperedges = len(hyperedge_to_index)
        hyperedge_index = torch.empty((2,0), dtype=torch.long)
        # print(f"hyperedge_index: {hyperedge_index}")
        # n=0
        for i,edge in enumerate(hypergraph):
            num_react = len(edge[1])
            num_prod = len(edge[2])
            v_per_edge = num_react + num_prod
            new_edge_idx = torch.empty((2, v_per_edge), dtype=torch.long)
            # print(f"new_edge_idx: {new_edge_idx}")
            for i, v_set in enumerate(edge[1:]):
                # print(f"the Length of {"reactants" if i==0 else "products"}: {len(v_set)}")
                for j, v in enumerate(v_set):
                    # print(f"new_edge_idx after process: {new_edge_idx}\n i: {i} j: {j}  i * num_react + j: { i * num_react + j}")
                    new_edge_idx[0, i * num_react + j] = vertices_to_idx[v]
                    new_edge_idx[1, i * num_react + j] = i


            hyperedge_index = torch.cat((hyperedge_index, new_edge_idx), dim=1)
            # n += v_per_edge

        # print(f"hyperedge_index after process: {hyperedge_index} \n actual shape: {hyperedge_index.shape} \n expected shape: {2, n}")
        # mol_attr_data = torch.tensor(mol_attr_data)
        # hyperedge_attr_data = torch.tensor(hyperedge_attr_data)

        hg_data = Data(x=mol_attr_data, edge_index= hyperedge_index, edge_attr=hyperedge_attr_data,edge_name=hyperedge_name)
        # print(f"hg_data.x: {hg_data.x}")
        mol_loader = DataLoader(hg_data.x)
        # print(f"mol_loader length: {len(mol_loader)}")
        rule_loader = DataLoader(hg_data.edge_attr)
        mol_rep = torch.empty(0, self.config.dim_h)
        for i, mol_graph_data in enumerate(mol_loader):
            # print(f"mol_graph_data batch: {mol_graph_data.batch}")
            mol_out = self.mol_encoder(mol_graph_data)
            # output = output.unsqueeze(0)
            # print(f"output: {output} output shape: {output.shape} i: {i}")
            mol_rep = torch.cat([mol_rep, mol_out], 0)
        # print(f"mol_rep shape: {mol_rep.shape}")
        rule_rep = torch.empty(0, self.config.dim_h)
        for i, rule_graph_data in enumerate(rule_loader):
            # print("rule_graph_data batch:" ,rule_graph_data)
            # rule_graph_batch = Batch.from_data_list(rule_graph_data)
            rule_out = self.rule_encoder(rule_graph_data)
            rule_rep = torch.cat([rule_rep, rule_out], 0)
        
        hypergraph_data = HyperGraphData(x=mol_rep, edge_index=hg_data.edge_index, edge_attr=rule_rep)
        
        return hypergraph_data
        


        
