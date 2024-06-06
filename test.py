# import gymnasium as gym
# from gymnasium import spaces
import torch
import torch.nn as nn
import pandas as pd
from utils import *
# from shortest_path_bipartite_ import *
import sys
# sys.path.append("/home/talax/xtof/local/Mod/lib64")
import os
sys.path.append("/home/mescalin/yitao/Documents/Code/CRN_IMA/OpenBioMed-main")
# sys.path.append("/home/mescalin/yitao/Documents/Code/CRN_IMA/OpenBioMed-main/open_biomed")
import subprocess
from mod import *
# from open_biomed.models.multimodal import MolFM
# import json
# from open_biomed.utils.data_utils import DataProcessorFast
# from open_biomed.utils import fix_path_in_config
# from rdkit import Chem
# from rdkit.Chem import Draw


import torch
from torch_geometric.data import Data

def hypergraph_to_pyg_data(hypergraph):
    # Extracting unique nodes and hyperedges
    nodes = set()
    hyperedges = set()
    for edge in hypergraph:
        hyperedges.add(tuple(edge[0].keys()))  # Extracting hyperedge
        for node_set in edge[1:]:
            for node in node_set:
                nodes.add(node)

    # Mapping nodes and hyperedges to unique indices
    nodes = sorted(list(nodes))
    print(f"hyperedges: {hyperedges}")
    node_to_index = {node: index for index, node in enumerate(nodes)}
    print(f"node_to_index: {node_to_index}")
    hyperedge_to_index = {hyperedge: index for index, hyperedge in enumerate(hyperedges)}
    print(f"hyperedge_to_index: {hyperedge_to_index}")
    # Constructing node features
    num_nodes = len(nodes)
    num_features = 1  # Update this based on your actual node features
    node_features = torch.randn(num_nodes, num_features)  # Placeholder for node features

    # Constructing hyperedge index
    num_hyperedges = len(hyperedges)
    num_nodes_per_hyperedge = len(next(iter(hyperedges)))
    print(f"num_hyperedges: {num_hyperedges}, num_nodes_per_hyperedge: {num_nodes_per_hyperedge}")
    hyperedge_index = torch.zeros(2, num_hyperedges * num_nodes_per_hyperedge, dtype=torch.long)
    print(f"hyperedge_index: {hyperedge_index}")
    for i, edge in enumerate(hypergraph):
        hyperedge = tuple(edge[0].keys())
        print(f"hyperedge: {hyperedge}")
        print(f"edge nodes: {edge[1:]}")
        for j, node_set in enumerate(edge[1:]):
            for node in node_set:
                print(f"j: {j}, length of edge: {len(edge[1:])}")
                hyperedge_index[0, i * num_nodes_per_hyperedge + j] = node_to_index[node]
                hyperedge_index[1, i * num_nodes_per_hyperedge + j] = 0 if j < len(edge[1:])//2 else 1
                print(f"hyperedge_index[0, i * num_nodes_per_hyperedge + j]: {hyperedge_index[0, i * num_nodes_per_hyperedge + j]}")
                print(f"hyperedge_index[1, i * num_nodes_per_hyperedge + j]: {hyperedge_index[1, i * num_nodes_per_hyperedge + j]}")
                print(f"hyperedge_index: {hyperedge_index}")
    # Constructing edge attributes (if available)
    # Here you need to extract and format the hyperedge attributes accordingly
    hyperedge_attrs = None

    # Create a PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=hyperedge_index,
                edge_attr=hyperedge_attrs, num_edges=num_hyperedges)

    return data




def main():


    # smis = [
    #     "CCCC[Si](Cl)(Cl)Cl", 
    #     "CO[C@@]1([C@@H]2N(C1=O)C(=C(CS2)COC(=O)N)C(=O)[O-])NC(=O)CC3=CC=CS3", 
    #     "CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)C(F)(F)F",
    # ]
    # texts = [
    #     "it appears as a colorless liquid with a pungent odor. Flash point 126Â°F. Corrosive to metals and skin. Used to make various silicon containing compounds.",
    #     "it is a cephalosporin carboxylic acid anion having methoxy, 2-thienylacetamido and carbamoyloxymethyl side-groups, formed by proton loss from the carboxy group of the semisynthetic cephamycin antibiotic cefoxitin. It is a conjugate base of a cefoxitin.",
    #     "it is a fatty acid derivative that is arachidonic acid in which the OH part of the carboxy group has been replaced by a trifluoromethyl group It has a role as an EC 3.1.1.4 (phospholipase A2) inhibitor. It is an organofluorine compound, a ketone, an olefinic compound and a fatty acid derivative. It derives from an arachidonic acid.",
    # ]

    # path = "/home/mescalin/yitao/Documents/Code/CRN_IMA/OpenBioMed-main"
    # mols = [Chem.MolFromSmiles(smi) for smi in smis]


    # config = json.load(open("../configs/mtr/molfm.json", "r"))
    # fix_path_in_config(config, path)
    # print("Config: ", config)
    # processor = DataProcessorFast(entity_type="molecule", config=config["data"]["mol"])
    # processor.featurizer.set_mol2text_dict(dict(zip(smis, texts)))
    # mols = processor(smis)
    # model = MolFM(config["network"])
    # state_dict = torch.load("../ckpts/fusion_ckpts/molfm.pth", map_location="cpu")["model"]
    # model.load_state_dict(state_dict)
    # model.eval()

    # with torch.no_grad():
    #     structure_feats = model.encode_mol(mols["structure"], proj=True)
    #     text_feats = model.encode_text(mols["text"])

    # for i in range(len(smis)):
    #     similarity = torch.cosine_similarity(structure_feats[i], text_feats)
    #     best = torch.argmax(similarity).item()
    #     print("Similarity for ", smis[i], "is", similarity.numpy(), ", Retrieved text is \"", texts[best], "\"")

    # Define your hypergraph data
    hypergraph = [
        [{'R9': 'L-lysine = (3S)-3,6-diaminohexanoate'}, {'LYS'}, {'p_{2,0}'}],
        [{'R6': 'L-malate = fumarate + H2O'}, {'MAL'}, {'H2O', 'p_{3,0}'}],
        [{'R3': 'L-Aspartate + 2-Oxoglutarate <=> Oxaloacetate + L-Glutamate'}, {'ASP', 'AKG'}, {'p_{6,0}', 'p_{6,1}'}],
        [{'R7': 'Lactoyl-CoA <=> Propenoyl-CoA + H2O'}, {'LAC-CoA'}, {'H2O', 'p_{8,0}'}]
    ]

    # Convert hypergraph to PyTorch Geometric Data object
    data = hypergraph_to_pyg_data(hypergraph)

    # Print the resulting data object
    print(data)
if __name__ == "__main__":
    main()