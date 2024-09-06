import gymnasium as gym
from gymnasium import spaces
import torch
import pandas as pd
from utils import *
from shortest_path_bipartite import *
import sys
# sys.path.append("/home/talax/xtof/local/Mod/lib64")
import subprocess
import mod
from mod import *
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from scipy.spatial.distance import squareform, pdist
from utils import *


def DFS_to_smiles(DFS):
    """
    Transform a name string to a corresponding SMILES string using a replacement dictionary.

    Parameters:
    - DFS (str): The input name string.
    - replacement_dict (dict): A dictionary where keys are substrings to be replaced,
      and values are the corresponding SMILES representations.

    Returns:
    - str: The transformed SMILES string.
    """
    replacement_dict = {"CoA": "*",
                        "Ad": "*",
                        "NAD": "*"}
    for substring, smiles in replacement_dict.items():
        if substring in DFS:

            # Replace substring and remove spaces
            DFS = DFS.replace(substring, smiles)
    
    return DFS




def extract_real_smiles(graph):
    """Extract the real SMILES string from a graphDFS instance or a regular graph."""
    # print("graph:",graph,graph.name,graph.id)
    # print(f"graph type:\n smiles: {graph.linearEncoding}")

    # Check if it's a graphDFS instance
    if Chem.MolFromSmiles(graph.linearEncoding) is None:
        # Extract the real SMILES string from the graph.name field
        real_smiles = DFS_to_smiles(graph.linearEncoding)

    else:
        # Extract the SMILES from the regular graph's graph.smiles field
        real_smiles = graph.smiles
    # print("real_smiles:",real_smiles)
    return real_smiles



def init_dg(mols):
    """
    Initialize a derivation graph with a single molecule.

    Parameters:
    - smiles (str): SMILES representation of the molecule.

    Returns:
    - pymod.DG: The initialized derivation graph.
    """
    # Initialize the derivation graph
    # dg = DG(graphDatabase=inputGraphs)
    # dg = DG(graphDatabase=mols)
    dg = DG(graphDatabase=mols,labelSettings=LabelSettings(LabelType.Term, LabelRelation.Specialisation))

    # Add the molecule to the derivation graph
    with dg.build() as b:
        b.execute(addSubset(mols))

    return dg


class Env(gym.Env):
    """
    Chemical reaction environment for reinforcement learning.

    Parameters:
    - dg_initial (pymod.DG): The initial derivation graph.
    - rules (list): List of reaction rules.
    - target_molecule (str): SMILES representation of the target molecule.

    Attributes:
    - num_init_mol (int): Number of known initial molecules.
    - current_dg (pymod.DG): The current derivation graph.
    - rules (list): List of reaction rules.
    - action_space (gym.spaces.Discrete): The action space.
    - reward (float): The reward for the current step.
    - done (bool): True if the episode is over, False otherwise.

    """
    def __init__(self, dg_initial, rules,target_molecule,mols_pool,reward="RXN_Happen_Or_Not"):
        super(Env, self).__init__()

        # Define the initial derivation graph
        self.dg_initial = dg_initial


        self.current_dg = self.dg_initial
        
        # Define reaction rules as the action space
        self.rules = rules
        self.action_space = spaces.Discrete(len(self.rules))
        self.target_molecule = target_molecule
    
        # Define observation space
        # Assuming the observation is the derivation graph object
        # self.observation_space = spaces.Dict({
        #     'vertices': spaces.MultiBinary(self.current_dg.numVertices),
        #     'edges': spaces.MultiBinary(self.current_dg.numEdges)
        # })

        # Initialize other variables
        self.reward_type = reward
        self.reward = 0
        self.done = False  # Added variable for episode termination
        self.scale = 1e-1
     
        # self.mols_pool = pd.read_csv(mols_pool_path)
        self.mols_pool = mols_pool
        self.num_init_mol = self.dg_initial.numVertices
        self.num_mols = self.num_init_mol + len(self.mols_pool)

        # print("self.num_mols:",self.num_mols)

    def apply_reaction(self, rule_index,mol_idx):
        # Apply the selected reaction rule to update the derivation graph
        if type(mol_idx) == tuple:
            mol_idx, cof_idx = mol_idx
            mols_pool = self.mols_pool["mols"]
            cofactors = self.mols_pool["cofactors"]
            mol = mols_pool[mol_idx]
            cof = cofactors[cof_idx]
            mol_set = [mol,cof]
        else:
            mol = self.mols_pool[mol_idx]
            mol_set = [mol]
        rule = self.rules[rule_index]
        mstrat = (
            rightPredicate[
                lambda der: all(g.vLabelCount('C') <= 88 for g in der.right)
            ](rule) )
        
        dg = DG()

        with dg.build() as b:
            # b.execute(addUniverse(self.current_dg.graphDatabase))
            if self.current_dg.edges is not None:
                for e in self.current_dg.edges:
                    b.addHyperEdge(e)
            # print("slected mol:",mol,mol.linearEncoding)
            # print("graphDatabase 1:",self.current_dg.graphDatabase)
            graphDatabase = []
            for g in self.current_dg.graphDatabase:
                graphDatabase.append(g.linearEncoding)

            # res = b.execute(addUniverse(self.current_dg.graphDatabase) >> addSubset(mol_set) >> mstrat)
            if self.mol_not_in_graphDatabase(mol):
                
                res = b.execute(addUniverse(self.current_dg.graphDatabase) >> addSubset(self.dg_initial.graphDatabase + mol_set) >> mstrat)
                
            else:
                res = b.execute(addUniverse(self.current_dg.graphDatabase) >> mstrat)   
            # res = b.execute(addSubset(self.current_dg.graphDatabase+[mol]) >> mstrat)   
            self.subset = res.subset
        print("dg graphDatabase after add mol and/or reaction:",dg.graphDatabase)
        # Check if the subset is empty or has zero length
        if len(self.subset) == 0:
            self.done = True

        return dg


    
    # Function to compute the maximum similarity score and the column index of the corresponding molecule
    def compute_max_similarity(self, vertices_dict):
        # Initialize max similarity score and corresponding column index
        max_similarity_score = 0
        # print("vertices_dict:",vertices_dict)
        max_similarity_idx = None
        # Iterate over molecules in the derivation graph
        for _, g in enumerate(vertices_dict):
            # Compute similarity score between the molecule and the target molecule
           
            similarity_score = calculate_similarity(extract_real_smiles(g), self.target_molecule)
            # print("similarity_score:",similarity_score)
            # print("max_similarity_score:",max_similarity_score)
            # print("g.id:",g.id)
            # print("self.num_mol:",self.num_mols)
            # print("g.id>=self.num_init_mol:",g.id>=self.num_mols)
        
            # Update max similarity score and corresponding column index if the current score is higher
            if similarity_score > max_similarity_score and g.id>=self.num_mols:
                max_similarity_score = similarity_score
                max_similarity_idx = g
    

        return max_similarity_score, max_similarity_idx
    
    def compute_reward(self, dg):
        # Get the maximum similarity score and the column index of the corresponding molecule
        hypergraph = dg_to_hypergraph(dg)
        print("hypergraph:",hypergraph)

        if hypergraph is None:
            reward = 0       
        else:

            Q1, Q2, V, H = hypergraph_to_biadjacency(hypergraph)
            if not H or not V:
                reward = 0
            else:       
           
                max_similarity_score, max_similarity_idx = self.compute_max_similarity(V)

                if max_similarity_idx is None:
                    reward = 0
                else:
                    df_Q1 = pd.DataFrame(Q1.numpy(), index = H.keys(), columns = V.keys() )
                    df_Q2 = pd.DataFrame(Q2.numpy(), index = V.keys(), columns = H.keys() )
                    # print("DATAFRAME df_Q1:\n ",df_Q1)
                    # print("DATAFRAME df_Q2:\n ",df_Q2)
                    for i,_ in enumerate(H.keys()):
                        for j,key in enumerate(V.keys()):

                            # print(f"key: {key} key.linearEncoding: {key.linearEncoding}")
                            # print(f"'NAD' or 'CoA' or 'Ad' in key.linearEncoding:{'NAD' in key.linearEncoding or 'CoA' in key.linearEncoding or 'Ad' in key.linearEncoding}")
                            if "NAD" in key.linearEncoding or "CoA" in key.linearEncoding or "Ad" in key.linearEncoding:

                                Q1[i,j] = float('inf')
                                Q2[j,i] = float('inf')
                    # print(f"Q1 after process: {Q1} q2: {Q2} ")
                    df_Q1_after = pd.DataFrame(Q1.numpy(), index = H.keys(), columns = V.keys() )
                    df_Q2_after = pd.DataFrame(Q2.numpy(), index = V.keys(), columns = H.keys() )
                    # print("DATAFRAME df_Q1_after:\n ",df_Q1_after)
                    # print("DATAFRAME df_Q2_after:\n ",df_Q2_after)


                    # Implement the Torgansin Zimmerman tropical algebra all pairs shortest path algorithm to compute reward
                    P1_2m1, P2_2m1, Q1_2m1, Q2_2m1, D = torgansin_zimmerman(Q1, Q2)
                    # print("P1_2m1 shape:",P1_2m1.shape)
                    # print("P2_2m1 shape:",P2_2m1.shape)
                    P2_2m1 = P2_2m1.numpy()/2
                    df_P2_2m1 = pd.DataFrame(P2_2m1, index = V.keys(), columns = V.keys() )    
                 
                    graph_ids = [g.id for g in V.keys()]

                    # Sort the list of graph IDs
                    graph_ids_sorted = sorted(graph_ids)
                    # print("graph_ids_sorted:",graph_ids_sorted)
                    # Reindex the sorted list
                    reindexed_graph_ids = list(range(1, len(graph_ids_sorted) + 1))
                    # print("reindexed_graph_ids:",reindexed_graph_ids)
                    # Now reindexed_graph_ids contains the sorted and reindexed list of graph IDs

                    df_P2_2m1["Graph Idx"] = reindexed_graph_ids
                    # print("DATAFRAME df_P2_2m1:\n ",df_P2_2m1)
                    # print("number of known initial molecules:",self.num_init_mol)
                    # Extract the shortest paths of mol with maximum similarity score from known initial molecules
                    # print("max_similarity_idx:",max_similarity_idx)
                    shortest_path = df_P2_2m1.loc[df_P2_2m1["Graph Idx"]<= self.num_init_mol, max_similarity_idx].values
                    shortest_path = shortest_path[shortest_path != 0.0]
                    # print("shortest_path:",shortest_path)
                    # Get the minimum shortest path from the known initial mols subset as the distance score
                    distance_score = shortest_path.min().item()

                    # Calculate the reward as the product of similarity score and the reward from distance score
                    reward = max_similarity_score * (1 / (distance_score * self.scale))

        return reward


    def step(self, rule,mol):
        # Execute a single step in the environment based on the selected action
        # if self.done:
        #     raise ValueError("Cannot step in an episode that is over. Call reset first.")

        # Apply the selected reaction rule
        dg = self.apply_reaction(rule,mol)
        if self.reward_type == "RXN_Happen_Or_Not":
            if type(mol) == tuple:
                mol_g = self.mols_pool["mols"][mol[0]]
            else:
                mol_g = self.mols_pool[mol]
            print("subset of dg:",self.subset)
                               
            # Check that mol_g is isomorphic to an existing molecule in self.subset
            if not any(self.is_isomorphic(mol_g, other_g) for other_g in self.subset):
                self.reward = 0  # mol_g is not isomorphic to anything in self.subset
                # for g in self.subset:
                #     print("mol in subset:",g,g.linearEncoding)
            elif len(self.subset) >= 2:  # At least two molecules (including mol_g)
                self.reward = 1  # Reaction produced a new molecule
            else:
                self.reward = 0  # Only mol_g exists, no new molecule
   
        elif self.reward_type == "RXN_Distance_to_Target":
            self.reward = self.compute_reward(dg)

        # self.reward = 0
        # Update the current derivation graph
        self.current_dg = dg

        # Return the updated observation, reward, episode termination status, and additional information
        return self.current_dg, self.reward, self.done

    def reset(self):
        # Reset the environment to its initial state for a new episode
        self.current_dg = self.dg_initial
        self.reward = 0
        self.done = False  # Reset done variable

        # Return the initial observation
        return self.current_dg
    
    def render(self,edges=False):
        if edges:
            for e in self.current_dg.edges:
                e.print()
        self.current_dg.print()
        # flush summary file handle
        post.flushCommands()
        # generate summary/summery.pdf
        subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])
    

    def dump(self):

        self.current_dg.dump()
        # flush summary file handle
        # post.flushCommands()
        # # generate summary/summery.pdf
        # subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])



    def is_isomorphic(self, mol1, mol2):
        if not (is_smiles(mol1.linearEncoding) and is_smiles(mol2.linearEncoding)):
            return mol1.isomorphism(mol2)==1
        else:
            return rdkit_isomorphism_check(mol1, mol2)

        
    def mol_not_in_graphDatabase(self,mol):
        graphDatabase = []
        for g in self.current_dg.graphDatabase:
            graphDatabase.append(g.linearEncoding)
            if self.is_isomorphic(mol,g):
                return False
        return True
    
        

def is_smiles(encoding):
    try:
        mol = Chem.MolFromSmiles(encoding, sanitize=True)
        return mol is not None
    except Exception:
        return False

def rdkit_isomorphism_check(mol1, mol2):
    # Use RDKit for SMILES isomorphism checks 
    smiles1 = mol1.linearEncoding
    smiles2 = mol2.linearEncoding

    rdkit_mol1 = Chem.MolFromSmiles(smiles1) 
    rdkit_mol2 = Chem.MolFromSmiles(smiles2) 

    return rdkit_mol1.HasSubstructMatch(rdkit_mol2) and rdkit_mol2.HasSubstructMatch(rdkit_mol1)


class ChemicalReactionEnv(gym.Env):
    """
    Chemical reaction environment for reinforcement learning.

    Parameters:
    - dg_initial (pymod.DG): The initial derivation graph.
    - rules (list): List of reaction rules.
    - target_molecule (str): SMILES representation of the target molecule.

    Attributes:
    - num_init_mol (int): Number of known initial molecules.
    - current_dg (pymod.DG): The current derivation graph.
    - rules (list): List of reaction rules.
    - action_space (gym.spaces.Discrete): The action space.
    - reward (float): The reward for the current step.
    - done (bool): True if the episode is over, False otherwise.

    """
    def __init__(self, dg_initial, rules,target_molecule):
        super(ChemicalReactionEnv, self).__init__()

        # Define the initial derivation graph
        self.dg_initial = dg_initial
        self.num_init_mol = self.dg_initial.numVertices
        self.current_dg = self.dg_initial
        
        # Define reaction rules as the action space
        self.rules = rules
        self.action_space = spaces.Discrete(len(self.rules))
        self.target_molecule = target_molecule
        
        # Define observation space
        # Assuming the observation is the derivation graph object
        # self.observation_space = spaces.Dict({
        #     'vertices': spaces.MultiBinary(self.current_dg.numVertices),
        #     'edges': spaces.MultiBinary(self.current_dg.numEdges)
        # })

        # Initialize other variables
        self.reward = 0
        self.done = False  # Added variable for episode termination
        self.scale = 1e-1


    def _apply_reaction(self, rule_index):
        # Apply the selected reaction rule to update the derivation graph
        rule = self.rules[rule_index]
        mstrat = (
            rightPredicate[
                lambda der: all(g.vLabelCount('C') <= 88 for g in der.right)
            ](rule) )
        
        dg = DG(graphDatabase=self.current_dg.graphDatabase)

        with dg.build() as b:
            b.execute(addUniverse(self.current_dg.graphDatabase))
            if self.current_dg.edges is not None:
                for e in self.current_dg.edges:
                    b.addHyperEdge(e)
            res = b.execute(addUniverse(self.current_dg.graphDatabase) >> addSubset(inputGraphs) >> mstrat)   
            subset = res.subset


        # Check if the subset is empty or has zero length
        if len(subset) == 0:
            self.done = True

        return dg
    
    # Function to compute the maximum similarity score and the column index of the corresponding molecule
    def compute_max_similarity(self, vertices_dict):
        # Initialize max similarity score and corresponding column index
        max_similarity_score = 0
        # print("vertices_dict:",vertices_dict)
        max_similarity_idx = None
        # Iterate over molecules in the derivation graph
        for _, g in enumerate(vertices_dict):
            # Compute similarity score between the molecule and the target molecule
           
            similarity_score = calculate_similarity(extract_real_smiles(g), self.target_molecule)
            # print("similarity_score:",similarity_score)
            # print("max_similarity_score:",max_similarity_score)
            # print("g.id:",g.id)
            # print("self.num_init_mol:",self.num_init_mol)
            # print("g.id>=self.num_init_mol:",g.id>=self.num_init_mol)
            # Update max similarity score and corresponding column index if the current score is higher
            if similarity_score > max_similarity_score and g.id>=self.num_init_mol:
                max_similarity_score = similarity_score
                max_similarity_idx = g
    
        # print("returned max_similarity_score:",max_similarity_score)
        # print("returned max_similarity_idx:",max_similarity_idx)
        return max_similarity_score, max_similarity_idx
    
    def _compute_reward(self, dg):
        # Get the maximum similarity score and the column index of the corresponding molecule


        hypergraph = dg_to_hypergraph(dg)
        print("hypergraph:",hypergraph)

        if hypergraph is None:
            reward = 0
        
        else:

            Q1, Q2, V, H = hypergraph_to_biadjacency(hypergraph)
            max_similarity_score, max_similarity_idx = self.compute_max_similarity(V)

            # print(f"Q1: {Q1} q2: {Q2} V: {V} H: {H}")
            # print(f"Q1 shape: {Q1.shape} q2 shape: {Q2.shape} V shape: {len(V)} H shape: {len(H)}")
            df_Q1 = pd.DataFrame(Q1.numpy(), index = H.keys(), columns = V.keys() )
            df_Q2 = pd.DataFrame(Q2.numpy(), index = V.keys(), columns = H.keys() )
            # print("DATAFRAME df_Q1:\n ",df_Q1)
            # print("DATAFRAME df_Q2:\n ",df_Q2)
            for i,_ in enumerate(H.keys()):
                for j,key in enumerate(V.keys()):

                    # print(f"key: {key} key.linearEncoding: {key.linearEncoding}")
                    # print(f"'NAD' or 'CoA' or 'Ad' in key.linearEncoding:{'NAD' in key.linearEncoding or 'CoA' in key.linearEncoding or 'Ad' in key.linearEncoding}")
                    if "NAD" in key.linearEncoding or "CoA" in key.linearEncoding or "Ad" in key.linearEncoding:

                        Q1[i,j] = float('inf')
                        Q2[j,i] = float('inf')
            # print(f"Q1 after process: {Q1} q2: {Q2} ")
            df_Q1_after = pd.DataFrame(Q1.numpy(), index = H.keys(), columns = V.keys() )
            df_Q2_after = pd.DataFrame(Q2.numpy(), index = V.keys(), columns = H.keys() )
            # print("DATAFRAME df_Q1_after:\n ",df_Q1_after)
            # print("DATAFRAME df_Q2_after:\n ",df_Q2_after)


            if max_similarity_idx is None:
                reward = 0
                
            else:

                # Implement the Torgansin Zimmerman tropical algebra all pairs shortest path algorithm to compute reward
                P1_2m1, P2_2m1, Q1_2m1, Q2_2m1, D = torgansin_zimmerman(Q1, Q2)
                # print("P1_2m1 shape:",P1_2m1.shape)
                # print("P2_2m1 shape:",P2_2m1.shape)
                P2_2m1 = P2_2m1.numpy()/2
                df_P2_2m1 = pd.DataFrame(P2_2m1, index = V.keys(), columns = V.keys() )    

                df_P2_2m1["Graph ID"] = [g.id for g in V.keys()]
                # print("DATAFRAME df_P2_2m1:\n ",df_P2_2m1)
                # print("number of known initial molecules:",self.num_init_mol)
                # Extract the shortest paths of mol with maximum similarity score from known initial molecules
                shortest_path = df_P2_2m1.loc[df_P2_2m1["Graph ID"]< self.num_init_mol, max_similarity_idx].values

                # Get the minimum shortest path from the known initial mols subset as the distance score
                shortest_path = shortest_path[shortest_path != 0.0]
                # print("shortest_path:",shortest_path)
                distance_score = shortest_path.min().item()

                # print("distance_score:",distance_score)

                # Calculate the reward as the product of similarity score and the reward from distance score
                reward = max_similarity_score * (1 / (distance_score * self.scale))

                # print("reward:",reward)
        
        return reward


    def step(self, action):
        # Execute a single step in the environment based on the selected action
        # if self.done:
        #     raise ValueError("Cannot step in an episode that is over. Call reset first.")

        # Apply the selected reaction rule
        dg = self._apply_reaction(action)

        self.reward = self._compute_reward(dg)
        # self.reward = 0
        # Update the current derivation graph
        self.current_dg = dg

        # Return the updated observation, reward, episode termination status, and additional information
        return self.current_dg, self.reward, self.done

    def reset(self):
        # Reset the environment to its initial state for a new episode
        self.current_dg = self.dg_initial
        self.reward = 0
        self.done = False  # Reset done variable

        # Return the initial observation
        return self.current_dg
    
    def render(self,edges=True,rules=True):
        # if edges:
        #     for e in self.current_dg.edges:
        #         e.print()
        # if rules:
        #     for r in self.rules:
        #         r.print()
        self.current_dg.print()
        # flush summary file handle
        post.flushCommands()
        # generate summary/summery.pdf
        subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])




class IntrinsicRewardEnv(gym.Env):
    """
    Chemical reaction environment for reinforcement learning.

    Parameters:
    - dg_initial (pymod.DG): The initial derivation graph.
    - rules (list): List of reaction rules.
    - target_molecule (str): SMILES representation of the target molecule.

    Attributes:
    - num_init_mol (int): Number of known initial molecules.
    - current_dg (pymod.DG): The current derivation graph.
    - rules (list): List of reaction rules.
    - action_space (gym.spaces.Discrete): The action space.
    - reward (float): The reward for the current step.
    - done (bool): True if the episode is over, False otherwise.

    """
    def __init__(self, dg_initial, rules,target_molecule):
        super(IntrinsicRewardEnv, self).__init__()

        # Define the initial derivation graph
        self.dg_initial = dg_initial
        self.num_init_mol = self.dg_initial.numVertices
        self.current_dg = self.dg_initial

        # Define reaction rules as the action space
        self.rules = rules
        self.action_space = spaces.Discrete(len(self.rules))
        self.target_molecule = target_molecule
        # Define observation space
        # Assuming the observation is the derivation graph object
        # self.observation_space = spaces.Dict({
        #     'vertices': spaces.MultiBinary(len(self.dg_initial.vertices)),
        #     'edges': spaces.MultiBinary(len(self.dg_initial.edges))
        # })

        # Initialize other variables
        self.reward = 0
        self.done = False  # Added variable for episode termination

    def _apply_reaction(self, rule_index):
        # Apply the selected reaction rule to update the derivation graph
        rule = self.rules[rule_index]
        mstrat = (
            rightPredicate[
                lambda der: all(g.vLabelCount('C') <= 88 for g in der.right)
            ](rule) )
        
        dg = DG(graphDatabase=self.current_dg.graphDatabase)

        with dg.build() as b:
            b.execute(addUniverse(self.current_dg.graphDatabase))
            if self.current_dg.edges is not None:
                for e in self.current_dg.edges:
                    b.addHyperEdge(e)
            res = b.execute(mstrat)   
            subset = res.subset



def calculate_similarity(smiles1, smiles2, radius=2):
    # Convert SMILES strings to RDKit molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is not None or mol2 is not None:
        
        # Generate Morgan fingerprints with a specified radius
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=2048)

        # Calculate Tanimoto similarity coefficient
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        similarity = 0

    return similarity



def are_smiles_equal(smiles1, smiles2):
    """
    Check if two SMILES representations represent the same molecule.

    Parameters:
    - smiles1 (str): First SMILES representation.
    - smiles2 (str): Second SMILES representation.

    Returns:
    - bool: True if the molecules are the same, False otherwise.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    assert mol1 is not None and mol2 is not None, "Invalid SMILES representation(s)"
    # Canonicalize the representations for comparison
    canonical_smiles1 = Chem.MolToSmiles(mol1, isomericSmiles=False, canonical=True)
    canonical_smiles2 = Chem.MolToSmiles(mol2, isomericSmiles=False, canonical=True)

    return canonical_smiles1 == canonical_smiles2



def get_graph_id(smiles, dg):
    """
    Get the graph ID from a given SMILES representation in a derivation graph.

    Parameters:
    - smiles (str): SMILES representation of the target vertex.
    - dg (pymod.DG): The derivation graph.

    Returns:
    - int or None: The graph ID if found, or None if not found.
    """
    target_vertex_id = None

    for vertex in dg.vertices:
        if are_smiles_equal(smiles, vertex.graph.smiles):
            target_vertex_id = vertex.graph.id
            break

    return target_vertex_id


# def get_similarity_score(smiles1, smiles2):
#     """
#     Compute the Graph Laplacian Kernel between two molecule graphs.

#     Parameters:
#     - smiles1 (str): SMILES representation of the first molecule.
#     - smiles2 (str): SMILES representation of the second molecule.

#     Returns:
#     - float: Graph Laplacian Kernel similarity between the two molecules.
#     """
#     mol1 = Chem.MolFromSmiles(smiles1)
#     mol2 = Chem.MolFromSmiles(smiles2)

#     assert mol1 is not None and mol2 is not None, "Invalid SMILES representation(s)"
#     # Generate Morgan fingerprints for the molecules
#     fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
#     fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)

#     # Convert Morgan fingerprints to numpy arrays
#     arr1 = np.zeros((1,))
#     arr2 = np.zeros((1,))
#     DataStructs.ConvertToNumpyArray(fp1, arr1)
#     DataStructs.ConvertToNumpyArray(fp2, arr2)

#     # Compute Graph Laplacian Kernel
#     kernel_matrix = np.exp(-pdist(np.vstack([arr1, arr2]), 'cosine'))
#     similarity = kernel_matrix[0]

#     return similarity
