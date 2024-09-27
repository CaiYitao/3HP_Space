import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from chytorch.nn import MoleculeEncoder
from chytorch.utils.data import MoleculeDataset, collate_molecules, chained_collate,SMILESDataset
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from getReactionCenter import get_reaction_center
from utils import collect_rules
import wandb
from tqdm import tqdm
from collections import defaultdict
import mod

class ReactionCenterPredictor(nn.Module):
    def __init__(self, num_rules: int, hidden_size: int = 256, max_tokens: int = 128):
        super().__init__()
        self.molecule_encoder = MoleculeEncoder(d_model=hidden_size, max_tokens=max_tokens)
        self.rule_embedding = nn.Embedding(num_rules, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, max_tokens)

    def forward(self, molecules: torch.Tensor, rule_idx: torch.Tensor) -> torch.Tensor:
        mol_encoding = self.molecule_encoder(molecules)
        mol_repr = mol_encoding.mean(dim=1)
        rule_emb = self.rule_embedding(rule_idx)
        combined = mol_repr + rule_emb
        x = self.fc(combined)
        x = torch.relu(x)
        x = self.output(x)
        return torch.sigmoid(x)
    
    
class ReactionDataset(Dataset):
    def __init__(self, meta_data: pd.DataFrame, rules_dict: Dict[int, mod.Rule], max_tokens: int = 128):
        self.meta_data = meta_data
        self.rules_dict = rules_dict
        self.max_tokens = max_tokens
        self.smiles_dataset = SMILESDataset(meta_data['Reactants'].tolist(), cache={})
        self.reactants_dataset = MoleculeDataset(self.smiles_dataset)
        self.num_rules = len(rules_dict)
        
        # Create a dictionary to store indices for each reactant
        self.reactant_indices = defaultdict(list)
        for idx, reactant in enumerate(meta_data['Reactants']):
            self.reactant_indices[reactant].append(idx)
        
        # Create a list of unique (reactant, rule) pairs
        self.unique_pairs = []
        for reactant, indices in self.reactant_indices.items():
            for rule_idx in range(self.num_rules):
                self.unique_pairs.append((indices[0], rule_idx))
        
        # Shuffle the unique pairs
        np.random.shuffle(self.unique_pairs)

    def __len__(self) -> int:
        return len(self.unique_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        reactant_idx, rule_idx = self.unique_pairs[idx]
        
        # Get the reactants 
        reactants = self.reactants_dataset[reactant_idx]
        print('reactants',reactants)
        # Extract the SMILES string for the reactant
        reactant_smiles = self.meta_data['Reactants'].iloc[reactant_idx]
        reactant_smiles_list = reactant_smiles.split('.')
        
        rule = self.rules_dict[rule_idx + 1]  # +1 because rules_dict is 1-indexed
        
        # Pass each SMILES string in the list to get_reaction_center
        target = get_reaction_center(reactant_smiles_list, rule)
        # print("target",target)
        target = self.pad_target(target)
        
        return reactants, rule_idx, target

    def pad_target(self, target: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros(self.max_tokens, dtype=torch.float32)
        padded[:len(target)] = target.float()
        return padded
    
def create_train_val_test_split(dataset: ReactionDataset, train_ratio: float = 0.8, test_ratio: float = 0.1):
    num_samples = len(dataset)
    num_train = int(num_samples * train_ratio)
    num_test = int(num_samples * test_ratio)
    
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    test_indices = indices[num_train:num_train + num_test]
    val_indices = indices[num_train + num_test:]
    
    return (torch.utils.data.Subset(dataset, train_indices), 
            torch.utils.data.Subset(dataset, val_indices), 
            torch.utils.data.Subset(dataset, test_indices))

def collate_fn(batch):
    reactants, rule_indices, targets = zip(*batch)
    reactants = collate_molecules(reactants)
    rule_indices = torch.tensor(rule_indices, dtype=torch.long)
    targets = torch.stack([t for t in targets if t is not None])
    return reactants, rule_indices, targets

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    for reactants, rule_indices, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)", leave=False, position=0):
        reactants, rule_indices, targets = reactants.to(device), rule_indices.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(reactants, rule_indices)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        wandb.log({"Train Loss per step":loss.item()})
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for reactants, rule_indices, targets in tqdm(val_loader, desc="Validation"):
            reactants, rule_indices, targets = reactants.to(device), rule_indices.to(device), targets.to(device)
            
            outputs = model(reactants, rule_indices)
            print('output[0]:',outputs[0])
            print('shape of output',outputs.shape)
            print('shape of target', targets.shape)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Convert outputs to binary predictions
            predicted_applicability = (outputs > 0.5).float()  
            print('prediction[0]',predicted_applicability[0])
            print('target[0]',targets[0])
            # Check if all elements of predicted match the target
            correct += (predicted_applicability == targets).all(dim=1).sum().item()  # Count correct predictions


            total += targets.size(0)  # Count total number of samples
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def predict_applicability(model: nn.Module, reactants: torch.Tensor, rule_idx: torch.Tensor, 
                          threshold: float = 0.5) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        outputs = model(reactants, rule_idx)
        # Convert outputs to binary predictions based on the threshold
        predicted_applicability = (outputs > threshold).float()
        return predicted_applicability  # Return the binary predictions directly

def evaluate_applicability(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for reactants, rule_indices, targets in test_loader:  # Use test_loader here
            reactants, rule_indices, targets = reactants.to(device), rule_indices.to(device), targets.to(device)
            
            predicted_applicability = predict_applicability(model, reactants, rule_indices)
            correct += (predicted_applicability == targets).all(dim=1).sum().item()  # Check if predictions match targets
            total += targets.numel()  # Count total number of elements in targets
    
    accuracy = correct / total
    return accuracy

def main():

    wandb.init(project="3HP_Space_RCP")
    
    # Hyperparameters
    hidden_size = 256
    max_tokens = 128
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 5

    # Load meta dataset
    meta_data = pd.read_csv("data/reaction_dataset_encoded.csv")

    # Prepare rules dictionary
    rule_gml_path = "/home/mescalin/yitao/Documents/Code/3HP_Space/gml_rules"
    rules_dict = {i+1: rule for i, rule in enumerate(collect_rules(rule_gml_path))}
    
    # Create dataset and data loader
    dataset = ReactionDataset(meta_data, rules_dict, max_tokens)
    train_dataset, val_dataset, test_dataset = create_train_val_test_split(dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)  # Create test loader

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReactionCenterPredictor(num_rules=len(rules_dict), hidden_size=hidden_size, max_tokens=max_tokens).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0
    best_model = None

    # Training loop
 
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_loss = train_model(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Average Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        # wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            torch.save(best_model, f"model/epoch_{epoch+1}_val_accuracy_{val_accuracy:.4f}.pth")
            print(f"Best model saved at epoch {epoch+1} with validation accuracy: {val_accuracy:.4f}")

        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
    applicability_accuracy = evaluate_applicability(model, test_loader, device)
    wandb.log({"applicability_accuracy":applicability_accuracy})
    wandb.finish()

if __name__ == "__main__":
    main()

