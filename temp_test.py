import pandas as pd
import mod


data = pd.read_csv('/home/mescalin/yitao/Documents/Code/3HP_Space/data/reaction_dataset.csv')
#print first 2 rows in full
print(data.head(2).to_string(index=False))
# get the smiles of reactants, and the rule applicable to the reactants which is 1 in the value we should get the index of the rule which means which column is the rule.there is no header in the csv file.
for index, row in data.iterrows():
    for col in range(1, len(row)):
        if row[col] == 1:
            rule_index = col
            break
    reactants = data['Reactants'][index]
    rule = data.columns[rule_index]
    print(reactants)
    print(rule)
    break

