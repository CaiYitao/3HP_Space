from env import ChemicalReactionEnv,init_dg,Env
import sys
import subprocess
sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *
import numpy as np
import random
import pandas as pd
from utils import *
from CRN_IMA.featurizer import HyperGraphFeaturizer
import torch
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from model.layer import GraphGPSLayer
# from model.module import GraphGPSLayer

# REPLACEMENT_DICT = {"[CoA]": "CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C(N=CN=C32)N)O)OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCC",
#                     "[Ad]": "O(O(N(C32=NC=N)C=C3N=CN2(O1)C)C)CC1C",
#                     "[NAD+]":"C1=CC(=C[N+](=C1)C2C(C(C(O2)COP(=O)([O-])OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)O)O)O)O)C(=O)N",
#                     "[NAD][H]": "C1C=CN(C=C1C(=O)N)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)O)O)O)O"}

class Config:
    dict={'dim_h': 64, 
    'num_heads': 4, 
    'attn_dropout': 0.1, 
    'layer_norm': True, 
    'batch_norm': False, 
    'local_gnn_type': "GINE", 
    'global_model_type': "Transformer",
    'd_model': 10, 
    'd_state': 64, 
    'd_conv': 64, 
    'expand': 2,
    'dropout': 0.1}
    def __init__(self, **kwargs):
        for key, value in self.dict.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
    




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
    # rules_path = "/home/mescalin/yitao/Documents/Code/3HPspace"
    # rules = collect_rules(rules_path)
    known_mol = [asp,lys,akg,pyr,lac,aca,ac,mal,co2,pi,ppi,nh3,h2o,hco3,hp,coa,nadh,nadplus,accoa,laccoa]
    # mol_path = "/home/mescalin/yitao/Documents/Code/CRN_IMA/known_mol.csv"
    # # known_mol = [asp,lys,akg,pyr,lac,aca,ac,mal,co2,pi,ppi,nh3,hco3,hp,coa,nadh,nadplus,accoa,laccoa]
    # known_mol = collect_inputGraphs(mol_path)
    # known_mol = known_mol + [smiles("[NH2:1][c:2]1[cH:3][cH:4][c:5]([C:6](=[O:7])[OH:8])[cH:9][cH:10]1","phenol")]
    print(f"known_mol: {known_mol}")
    target_molecule="OCC1OC(O)C(C(C1O)O)O"
    # target_molecule="OCC1OC(O)C(C(C1O)O)O"

    dg_init = init_dg(known_mol)
    env = ChemicalReactionEnv(dg_init,rules,target_molecule)
    hg_featurizer = HyperGraphFeaturizer()
    # dg_reset = env.reset()
    # dg_reset.print()
    
    # post.flushCommands()
    # generate summary/summery.pdf
    # subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])
    episodes = 30
    random.seed(1337)
    action = []
    reward = []
    Done = []

    config = Config()
    # graph_mamba=GraphMambaLayer()
    graph_gps = GraphGPSLayer(config)
    # if torch.cuda.is_available():
    #     graph_mamba = graph_mamba.cuda()

    for i in range(episodes):
        a = random.randint(0,len(rules)-1)
        print(f"rule action: R{a+1} at the {i+1}th episode")

        dg_new,r,done= env.step(a)
        hg_data = hg_featurizer(dg_new)
        if torch.cuda.is_available():
            hg_data = hg_data.cuda()
        # if i == episodes-1:
            # print(f"hg_data: {hg_data}\n hg_data.x: {hg_data.x}\n hg_data.edge_index: {hg_data.edge_index}\n hg_data.edge_attr: {hg_data.edge_attr}")
        pickle.dump(hg_data, open("/home/mescalin/yitao/Documents/Code/CRN_IMA/hg_data.pkl", "wb"))
            # dataloader = DataLoader(hg_data.x)
            # print(f"length of dataloader: {len(dataloader)} the number of mols{len(hg_data.x)}")
            # mol_rep = []
            # for i, batch in enumerate(dataloader):

                # print(f"batch.x {batch.x} \n shape {batch.x.shape} \n batch.edge_index: {batch.edge_index} \n edge_attr: {batch.edge_attr} ")                # output = graph_mamba(batch)
                # output = graph_gps(batch)
                # mol_rep.append(output)    
            #     print(f"output: {output}")
      
            #     print(f"batch: {batch} ")

            #     print(f"batch.batch: {batch.batch} ")
            # # for _, (edge_attr,edge_name) in enumerate(zip(hg_data.edge_attr, hg_data.edge_name)):
            #     # edge_attr_loader = DataLoader(edge_attr)
            #     batch_edge_attr = Batch.from_data_list(edge_attr)
            #     print(f"edge_name: {edge_name}")    
            #     print(f"batch_edge_attr: {batch_edge_attr.x} \n shape: {batch_edge_attr.x.shape}")
            #     print(f"batch_edge_attr.edge_index: {batch_edge_attr.edge_index} \n shape: {batch_edge_attr.edge_index.shape}")
        
            # print(f"mol_rep: {mol_rep}")
        
        env.render()
        action.append(a)
        reward.append(r)
        Done.append(done)

    print(f"action: {action}")
    print(f"reward: {reward}")
    print(f"Done: {Done}")
    dg_reset = env.reset()
    env.render()



if __name__ == "__main__":
    main()