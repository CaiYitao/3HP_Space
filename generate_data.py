import mod
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from utils import collect_rules, collect_inputGraphs

def get_unique_reactants(dg):
    """Get unique reactant SMILES/DFS from the derivation graph."""
    unique_reactants = set()
    for e in dg.edges:
        reactant_smiles = '.'.join(sorted(s.graph.linearEncoding for s in e.sources))
        unique_reactants.add(reactant_smiles)
    return list(unique_reactants)

def generate_dataset(dg, rules):
    """Generate the dataset from the derivation graph."""
    reactants = get_unique_reactants(dg)
    print('reactants:', reactants)  

    rule_mapping = {rule.name: f"R{i+1}" for i, rule in enumerate(rules)}
    print('rule_mapping:', rule_mapping)

    # Create a DataFrame with 'Reactants' column and rules as columns, initialized with NaN
    df = pd.DataFrame(index=reactants, columns=['Reactants'] + list(rule_mapping.values())).reset_index().rename(columns={'index': 'Reactants'})
    df.fillna(np.nan, inplace=True)
    
    # Fill the DataFrame
    for edge in dg.edges:
        source_smiles = '.'.join(sorted(s.graph.linearEncoding for s in edge.sources))
        for rule in rules:
            if source_smiles in df.index:
                if rule in edge.rules:
                    df.loc[source_smiles, rule_mapping[rule.name]] = 1
                elif pd.isna(df.loc[source_smiles, rule_mapping[rule.name]]):
                    df.loc[source_smiles, rule_mapping[rule.name]] = 0
    
    return df

def main():
    dg_path = '/home/mescalin/yitao/Documents/Code/3HP_Space/data/dg'  # Replace with the actual path to your DG file
    rules_path = '/home/mescalin/yitao/Documents/Code/3HP_Space/gml_rules'  # Replace with the actual path to your rules directory
    input_graphs_path = '/home/mescalin/yitao/Documents/Code/3HP_Space/known_mol.csv'  # Replace with the actual path to your input graphs directory
    rules = collect_rules(rules_path)
    input_graphs = collect_inputGraphs(input_graphs_path)

    dg = mod.DG.load(input_graphs, rules, dg_path)

    dataset = generate_dataset(dg, rules)
    
    # Save the dataset to a CSV file
    output_path = '/home/mescalin/yitao/Documents/Code/3HP_Space/data/reaction_dataset.csv'
    dataset.to_csv(output_path)
    print(f"Dataset saved to {output_path}")

if __name__ == '__main__':
    main()