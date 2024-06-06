import wandb
import torch
import random

from runner import Runner
from buffer import ReplayBuffer, Episode
from env import Env,init_dg
from utils import *
import pandas as pd
import sys
sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *
from model.PPOagent import PPOAgent,PPORunner


# wandb_config = {
#     'dim_h': 128,
#     'num_heads': 6,
#     'attn_dropout': 0.1,
#     'layer_norm': True,
#     'batch_norm': False,
#     'dropout': 0.2,
#     'dim_hyperedge': 10,
#     'dim_in': 64,
#     'epochs': 100,
#     'max_episodes': 1000,
#     'max_steps_per_episode': 100,
#     'discount_factor': 0.99,
#     'actor_lr': 0.001,
#     'critic_lr': 0.002,
#     'target_update_interval': 1,
#     'n_epochs': 100,
#     'policy_noise': 0.2,
#     'noise_clip': 0.5,
#     'discount': 0.99,
#     'batch_size': 128,
#     'policy_freq': 2,
#     'use_amp': True,
#     'cross_attention': True,
#     'parallel_output': False,
#     'actor_optimizer': 'Adam',
#     'critic_optimizer': 'Adam',
#     'actor_scheduler_type': 'StepLR',
#     'critic_scheduler_type': 'StepLR',
#     'actor_scheduler_params': {'step_size': 20, 'gamma': 0.5},
#     'critic_scheduler_params': {'step_size': 20, 'gamma': 0.5},
#     'gradient_clip': 0.5,
#     'tau': 0.005,
#     'actor_loss_weight': 0.5,
#     'critic_loss_weight': 0.5,
#     'num_hglayers': 3,
#     'dim_hyperedge': 10,
#     'dim_in': 64,
#     'rule_dim': 64,
#     'mol_dim': 64,
#     'parallel_output': False,
#     'actor_optimizer': 'Adam',
#     'critic_optimizer': 'Adam',
#     'actor_scheduler_type': 'StepLR',
#     'critic_scheduler_type': 'StepLR',
#     'actor_scheduler_params': {'step_size': 20, 'gamma': 0.5},
#     'critic_scheduler_params': {'step_size': 20, 'gamma': 0.5},
#     'gradient_clip': 1.0,
#     'tau': 0.005,
#     'actor_loss_weight': 0.5,
#     'critic_loss_weight': 0.5,
  
# }

class Config:
    def __init__(self):
        # Graph GPS Layer
        self.bond_attr_dim = 3
        self.atom_attr_dim = 10
        self.dim_h = 128
        self.num_heads = 4
        self.num_layers = 3
        self.attn_dropout = 0.1
        self.layer_norm = True
        self.batch_norm = False
        self.dropout = 0.2
        self.local_gnn_type = 'GINE'
        self.global_model_type = 'Transformer'

        # HyperGraph Layer
        self.num_edge_heads = 2
        self.num_node_heads = 2
        self.dim_hyperedge = 10
        self.dim_in = 64
        self.num_hglayers = 2


        # Training
        self.epochs = 100
        self.num_episodes = 1000
        self.max_episodes = 1000
        self.max_steps = 1000
        self.discount_factor = 0.9
        self.lr = 0.0001
  
        self.n_epochs = 100
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.discount = 0.9
        self.batch_size = 64
        self.policy_freq = 2
        self.use_amp = True
        self.cross_attention = True
        self.parallel_output = False
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        self.clip_epsilon = 0.2
        self.num_updates = 50
        self.hg_attn_heads = 4  # Number of attention heads in the hypergraph Conv layer

        # Optimizers
        self.optimizer = 'Adam'
        self.optimizer_params = {'lr': 1e-3, 'weight_decay': 1e-5, 'betas': (0.9, 0.999)}
        # self.critic_optimizer = 'Adam'
        # self.critic_optimizer_params = {'lr': 5e-3, 'weight_decay': 1e-5, 'betas': (0.9, 0.999)}
        self.gradient_clip = 0.5
        

        # Learning rate scheduler
        self.scheduler = 'StepLR'

        self.scheduler_params = {'step_size': 20, 'gamma': 0.5}
 

        # Other hyperparameters
        self.tau = 0.005

        self.rule_dim = 11
        self.mol_dim = 721
        # self.max_action = 721
        self.buffer_capacity = 100000
        self.log_interval = 10



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
lac = smiles('CC(O)C(=O)O', 'LAC')



if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="CRN_RewardType_RXN_Happen_Or_Not", name="PPOAgent")
    # wandb.init(project="CRN_RewardType_RXN_Distance_to_Target", name="PPOAgent")


    config = Config()
    # Initialize wandb config with your settings
    wandb.config.update(config)
    # Create an instance of your environment
    # mols_dict = {"smiles":['OC(=O)CC(N)C(=O)O','NCCCCC(N)C(=O)O','OC(=O)C(=O)CCC(=O)O','CC(=O)C(=O)O','CC(O)C(=O)O','CC=O','CC(=O)O','OC(=O)C(O)CC(=O)O','O=C=O','OP(=O)(O)O','OP(=O)(O)OP(=O)(O)O','[NH3]','OC(=O)O','[H+]'],
    #              "name":['ASP','LYS','AKG','PYR','LAC','AcA','Ac','MAL','CO2','Pi','PPi','NH3','HCO3','H+']}
    # config.mol_dim = len(mols_dict["smiles"])
    # known_mol = pd.DataFrame(mols_dict)
    # known_mol.to_csv("known_mol.csv",index=False)
    mol_path0 = "/home/mescalin/yitao/Documents/Code/CRN_IMA/known_mol.csv"
    mol_path = "/home/mescalin/yitao/Documents/Code/CRN_IMA/namesmilesinchi_pretty.csv"
    # known_mol = [asp,lys,akg,pyr,lac,aca,ac,mal,co2,pi,ppi,nh3,hco3,hp,coa,nadh,nadplus,accoa,laccoa]
    known_mol0 = collect_inputGraphs(mol_path0)
    # print(f"known_mol0: {known_mol0}")
    known_mol = collect_inputGraphs(mol_path)
    # print(f"known_mol: {known_mol}")
    mol_pool = known_mol0 + known_mol
    # print(f"known_mol within know_mol0: {mol_pool}")
    # print(f"len known_mol: {len(mol_pool)}")
    rules_path = "/home/mescalin/yitao/Documents/Code/3HPspace"
    rules = collect_rules(rules_path)
    target_molecule="CC(=O)C(=O)O"
    init_mol = [h2o,lac,coa,accoa,nadh,nadplus]
    dg_init = init_dg(init_mol)
    # print("inputGraphs",inputGraphs)
    print("dg init graphDatabase", dg_init.graphDatabase)
    # reward_type = "RXN_Happen_Or_Not"
    reward_type = "RXN_Distance_to_Target" 
    env = Env(dg_init,rules,target_molecule,mol_pool, reward=reward_type)
    # Create an instance of TD3Agent
    agent = PPOAgent(config,env)
    buffer = ReplayBuffer(config)   
    # Create an instance of Runner
    runner = PPORunner(agent,config)
    # Start training
    runner.train()

