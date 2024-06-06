from rdchiral.main import rdchiralRun
from rdkit.Chem import AllChem

def smarts_to_gml_with_rdchiral(reaction_smarts):
    # Create an RDKit reaction object from SMARTS
    reaction = AllChem.ReactionFromSmarts(reaction_smarts)
    
    # Initialize RDChiral for the reaction
    rxn = rdchiralRun([reaction])

    # Extract the reactants and products
    reactants, products = rxn.run()

    # Initialize the GML rule dictionary
    gml_rule = {
        "ruleID": "reaction name",  # You can set the reaction name here
        "left": {"edges": []},
        "context": {"nodes": [], "edges": []},
        "right": {"edges": []}
    }

    # Create dictionaries to track node IDs and store atom labels
    atom_id_map = {}
    atom_labels = {}

    # Extract the bonds in the reaction
    for reactant, product in zip(reactants, products):
        for atom in reactant.GetAtoms():
            atom_id = atom.GetIdx()
            atom_id_map[atom_id] = len(atom_id_map) + 1
            atom_labels[atom_id] = atom.GetSymbol()

        for bond in reactant.GetBonds():
            begin_atom = bond.GetBeginAtom().GetIdx()
            end_atom = bond.GetEndAtom().GetIdx()
            if bond not in product.GetBonds():
                gml_rule["left"]["edges"].append({
                    "source": atom_id_map[begin_atom],
                    "target": atom_id_map[end_atom],
                    "label": "-"
                })

        for atom in product.GetAtoms():
            atom_id = atom.GetIdx()
            if atom_id not in atom_id_map:
                atom_id_map[atom_id] = len(atom_id_map) + 1
                atom_labels[atom_id] = atom.GetSymbol()

        for bond in product.GetBonds():
            begin_atom = bond.GetBeginAtom().GetIdx()
            end_atom = bond.GetEndAtom().GetIdx()
            if bond not in reactant.GetBonds():
                gml_rule["right"]["edges"].append({
                    "source": atom_id_map[begin_atom],
                    "target": atom_id_map[end_atom],
                    "label": bond.GetSmarts()
                })

    # Populate context with atom nodes and edges not participating in the reaction
    for atom_id, label in atom_labels.items():
        gml_rule["context"]["nodes"].append({
            "id": atom_id_map[atom_id],
            "label": label
        })

    # Print the resulting GML rule
    print("GML Rule:", gml_rule)

# Example reaction SMARTS
reaction_smarts = "[H:1][O:2][C:3][C:4][H:5]>>[H:1][O:2][H:5].[C:3]=[C:4]"

# Transform the reaction SMARTS into GML format with rdchiral
smarts_to_gml_with_rdchiral(reaction_smarts)
