import wandb
import torch
import random
from CRN_IMA.model.TD3agent import TD3Agent
from runner import Runner
from buffer import ReplayBuffer, Episode
from env import Env,init_dg
from utils import *
import pandas as pd
import sys
sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *



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
        self.dim_hyperedge = 10
        self.dim_in = 64
        self.num_hglayers = 3


        # Training
        self.epochs = 100
        self.max_episodes = 1000
        self.max_steps_per_episode = 100
        self.discount_factor = 0.99
        # self.actor_lr = 0.001
        # self.critic_lr = 0.002
        self.target_update_interval = 1
        self.n_epochs = 100
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.discount = 0.99
        self.batch_size = 128
        self.policy_freq = 2
        self.use_amp = True
        self.cross_attention = True
        self.parallel_output = False

        # Optimizers
        self.actor_optimizer = 'Adam'
        self.actor_optimizer_params = {'lr': 1e-3, 'weight_decay': 1e-5, 'betas': (0.9, 0.999)}
        self.critic_optimizer = 'Adam'
        self.critic_optimizer_params = {'lr': 5e-3, 'weight_decay': 1e-5, 'betas': (0.9, 0.999)}
        self.gradient_clip = 0.5
        

        # Learning rate scheduler
        self.actor_scheduler_type = 'StepLR'
        self.critic_scheduler_type = 'StepLR'
        self.actor_scheduler_params = {'step_size': 20, 'gamma': 0.5}
        self.critic_scheduler_params = {'step_size': 20, 'gamma': 0.5}

        # Other hyperparameters
        self.tau = 0.005
        self.actor_loss_weight = 0.5
        self.critic_loss_weight = 0.5
        self.rule_dim = 11
        self.mol_dim = 19
        self.max_action = 10
        self.capacity = 100000

        # Rule Index Dictionary
        self.rule_idx_dict = {
            0: "R1: (S)-lactate + NAD+ = pyruvate + NADH + H+",
            1: "R2: Ac-aldehyde + CoA + NAD+ = Ac-CoA + NADH + H+",
            2: "R3: L-Aspartate + 2-Oxoglutarate <=> Oxaloacetate + L-Glutamate",
            3: "R4: Acyl-CoA + H2O <=> CoA + Carboxylate",
            4: "R5: SAM + CO2 = SAM-CO2H",
            5: "R6: L-malate = fumarate + H2O",
            6: "R7: Lactoyl-CoA <=> Propenoyl-CoA + H2O",
            7: "R8: L-Aspartate <=> Fumarate + Ammonia",
            8: "R9: L-lysine = (3S)-3,6-diaminohexanoate",
            9: "R10: ATP + Acetate + CoA <=> AMP + PPi + Ac-CoA",
            10: "R11: ATP + Ac-CoA + HCO3- = ADP + malonyl-CoA + Pi"
        }





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


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="ChemicalReactionNetworks")

    config = Config()
    # Initialize wandb config with your settings
    wandb.config.update(config)
    # Create an instance of your environment
    mols_dict = {"smiles":['OC(=O)CC(N)C(=O)O','NCCCCC(N)C(=O)O','OC(=O)C(=O)CCC(=O)O','CC(=O)C(=O)O','CC(O)C(=O)O','CC=O','CC(=O)O','OC(=O)C(O)CC(=O)O','O=C=O','OP(=O)(O)O','OP(=O)(O)OP(=O)(O)O','[NH3]','OC(=O)O','[H+]'],
                 "name":['ASP','LYS','AKG','PYR','LAC','AcA','Ac','MAL','CO2','Pi','PPi','NH3','HCO3','H+']}

    known_mol = pd.DataFrame(mols_dict)
    known_mol.to_csv("known_mol.csv",index=False)
    mol_path = "/home/mescalin/yitao/Documents/Code/CRN_IMA/known_mol.csv"
    # known_mol = [asp,lys,akg,pyr,lac,aca,ac,mal,co2,pi,ppi,nh3,hco3,hp,coa,nadh,nadplus,accoa,laccoa]
    known_mol = collect_inputGraphs(mol_path)
    print(f"known_mol: {known_mol}")
    rules_path = "/home/mescalin/yitao/Documents/Code/3HPspace"
    rules = collect_rules(rules_path)
    target_molecule="OCC1OC(O)C(C(C1O)O)O"
    init_mol = [h2o,coa,accoa,nadh,nadplus]
    dg_init = init_dg(init_mol)   
    env = Env(dg_init,rules,target_molecule,known_mol)
    # Create an instance of TD3Agent
    agent = TD3Agent(config)
    buffer = ReplayBuffer(config)   
    # Create an instance of Runner
    runner = Runner(agent,env,buffer, config)

    # Start training
    runner.run()
