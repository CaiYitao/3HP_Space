from chytorch.utils.data import MoleculeDataset, SMILESDataset
from torch.utils.data import DataLoader
from chytorch.nn import MoleculeEncoder
from getReactionCenter import get_reaction_center

from utils import encode
import pandas as pd
#load the dataset 
df = pd.read_csv('data/reaction_dataset_encoded.csv')

#encode the data in the dataset
# df['Reactants'] = df['Reactants'].apply(encode)
# save_path = 'data/reaction_dataset_encoded.csv'
# df.to_csv(save_path, index=False)
data = df['Reactants'].tolist()





encoder = MoleculeEncoder()
def main():

    ds = MoleculeDataset(SMILESDataset(data[:1], cache={}))
    dl = DataLoader(ds, batch_size=1)
 
    for batch in dl:
        x = encoder(batch)
        print('batch:', batch)
        print('batch.atoms shape:', batch.atoms.shape)
        print('batch.neighbors shape:', batch.neighbors.shape)
        print('batch.distances shape:', batch.distances.shape)
        print('x:', x)
        print('x.shape:', x.shape)

    

if __name__ == '__main__':
    main()