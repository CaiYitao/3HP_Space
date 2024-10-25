from rdkit import Chem
from typing import List, Optional, Tuple
import re
from utils import encode,decode
from rdkit import Chem
from typing import List, Optional, Tuple
import re

class AtomMapper:
    """Efficient class for mapping atoms in chemical reactions."""
    
    def __init__(self):
        self.map_pattern = re.compile(r':(\d+)]')
    
    def _create_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Create RDKit molecule preserving explicit hydrogens."""
        try:
            # Use parseSmiles to preserve explicit H
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
                
            # Sanitize the molecule while keeping explicit H
            Chem.SanitizeMol(mol, 
                sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_ADJUSTHS)
            return mol
        except Exception as e:
            print(f"Error parsing SMILES {smiles}: {str(e)}")
            return None

    def _map_atoms(self, mol: Chem.Mol, start_idx: int) -> Tuple[Chem.Mol, int]:
        """Map atoms including explicit hydrogens."""
        if mol is None:
            return None, start_idx
            
        # First map non-hydrogen atoms
        for atom in mol.GetAtoms():         
            atom.SetAtomMapNum(start_idx)
            start_idx += 1
            
        return mol, start_idx

    def process_smiles(self, smiles: str, start_idx: int = 1) -> str:
        """Process a single SMILES string preserving explicit H."""
        mol = self._create_mol(smiles)
        if mol is None:
            return smiles
            
        mapped_mol, _ = self._map_atoms(mol, start_idx)
        if mapped_mol is None:
            return smiles
        
        # Use non-canonical SMILES to preserve input order
        mapped_smiles = Chem.MolToSmiles(mapped_mol, 
                                        canonical=False, 
                                        allHsExplicit=False)
        
        # Convert map numbers to atom IDs preserving explicit H
        return self.map_pattern.sub(lambda m: f":{m.group(1)}]", mapped_smiles)

    def process_reactants(self, reactants: str) -> str:
        """Process multiple reactants while preserving explicit H."""
        if not reactants:
            return ""
            
        reactant_list = reactants.split(".")
        current_idx = 1
        processed = []
        
        for reactant in reactant_list:
            mol = self._create_mol(reactant)
            if mol is None:
                continue
                
            mapped_mol, current_idx = self._map_atoms(mol, current_idx)
            if mapped_mol is None:
                continue
                
            # Use non-canonical SMILES to preserve input structure
            mapped_smiles = Chem.MolToSmiles(mapped_mol, 
                                           canonical=False, 
                                           allHsExplicit=False)
            print(f"\n Mapped SMILES: {mapped_smiles} \n ")
            # print(f" MolfromSmiles: {Chem.MolFromSmiles(mapped_smiles)}")
            # processed_smiles = self.map_pattern.sub(
            #     lambda m: f":{m.group(1)}]", 
            #     mapped_smiles
            # )

            # print(f"\n Processed SMILES: {processed_smiles} \n ")
            # processed.append(processed_smiles)
            processed.append(mapped_smiles)
            
        return ".".join(processed)

def compare_smiles(original: str, processed: str):
    """Compare original and processed SMILES for validation."""
    orig_mol = Chem.MolFromSmiles(original, sanitize=False)
    proc_mol = Chem.MolFromSmiles(processed, sanitize=False)
    # print(f"\n Original Mol: {orig_mol} \n ")
    # print(f"\n Processed Mol: {proc_mol} \n ")
    if orig_mol is None or proc_mol is None:
        return False
        
    # Compare atom counts
    orig_atoms = [atom.GetSymbol() for atom in orig_mol.GetAtoms()]
    proc_atoms = [atom.GetSymbol() for atom in proc_mol.GetAtoms()]
    # print(f"\n Original Atoms: {orig_atoms} \n ")
    # print(f"\n Processed Atoms: {proc_atoms} \n ")
    return orig_atoms == proc_atoms


# def compare_smiles(original: str, processed: str) -> bool:
#     """Compare original and processed SMILES ensuring valid structures."""
#     orig_mol = Chem.MolFromSmiles(original)
#     proc_mol = Chem.MolFromSmiles(processed)
    
#     if orig_mol is None or proc_mol is None:
#         return False
        
#     # Compare atom counts including mapped hydrogens
#     orig_atoms = [(atom.GetSymbol(), len(atom.GetNeighbors())) 
#                  for atom in orig_mol.GetAtoms()]
#     proc_atoms = [(atom.GetSymbol(), len(atom.GetNeighbors())) 
#                  for atom in proc_mol.GetAtoms()]
    
#     # Verify all atoms are mapped in processed SMILES
#     all_mapped = all(atom.GetAtomMapNum() > 0 
#                     for atom in proc_mol.GetAtoms())
    
#     return sorted(orig_atoms) == sorted(proc_atoms) and all_mapped

def process_reaction_smiles(reaction_smiles: str, verbose: bool = False) -> Tuple[str, dict]:
    """Process complete reaction SMILES preserving explicit H."""
    mapper = AtomMapper()
    stats = {
        'total_molecules': 0,
        'processed_molecules': 0,
        'explicit_h_count': 0
    }
    
    try:
        if ">>" in reaction_smiles:
            reactants, products = reaction_smiles.split(">>")
            processed_reactants = mapper.process_reactants(reactants)
            processed_products = mapper.process_reactants(products)
            result = f"{processed_reactants}>>{processed_products}"
        else:
            result = mapper.process_reactants(reaction_smiles)
            
        # Calculate statistics
        stats['total_molecules'] = len(reaction_smiles.split('.'))
        stats['processed_molecules'] = len(result.split('.'))
        
        if verbose:
            print(f"Original SMILES:\n{reaction_smiles}")
            print(f"\nProcessed SMILES:\n{result}")
            
        return result, stats
        
    except Exception as e:
        print(f"Error processing reaction SMILES: {str(e)}")
        return reaction_smiles, stats


# Example usage and testing
def run_example():
    # Test data
    test_reactants = """C(O)(O)=O.[Ad][O][P](=[O])([O][H])[O][P](=[O])([O][H])[O][P](=[O])([O][H])[O][H].[CoA][S][C]([C]([N]([H])[H])([C]([H])([C](=[O])[S][CoA])[C](=[O])[O][H])[H])=[O]"""
    
    # Process with different options
    result, stats = process_reaction_smiles(encode(test_reactants), verbose=True)
    result=decode(result)
    print("\nOriginal SMILES:")
    print(test_reactants)
    print("\nProcessed SMILES:")
    print(result)
    print("\nStatistics:")
    
    show_comparison = []
    for r,t in zip(result.split("."), test_reactants.split(".")):
        r,t = encode(r),encode(t)
        show_comparison.append([r,t,compare_smiles(r,t)])
    print("Lets compare the result with original smiles:   ",show_comparison )
    
    for key, value in stats.items():
        print(f"{key}: {value}")
    #test if result can be parsed by rd kit
    for r in result.split("."):
        mol = Chem.MolFromSmiles(encode(r), sanitize=False)
        if mol is None:
            print("Error: Processed SMILES cannot be parsed by RDKit.")
    return result, stats




if __name__ == "__main__":
    import pandas as pd
    import os
    # result, stats = run_example()
    meta_data = pd.read_csv(os.path.join(os.getcwd(), "data/reaction_dataset.csv"))

    reactants = meta_data["Reactants"].tolist()

    # Batch processing
    def process_batch(smiles_list):
        mapper = AtomMapper()
        return [mapper.process_reactants(encode(smiles)) for smiles in smiles_list]

    mapped_reactants = process_batch(reactants)
    mapped_reactants = [decode(smiles) for smiles in mapped_reactants]
    # Save the mapped reactants to the DataFrame
    meta_data["mapped_reactants"] = mapped_reactants
    meta_data.to_csv(os.path.join(os.getcwd(), "data/reaction_dataset.csv"), index=False)