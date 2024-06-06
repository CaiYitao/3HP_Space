
import sys
import subprocess
sys.path.append("/home/talax/xtof/local/Mod/lib64")
from mod import *





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

    # for m in inputGraphs:
    #     m.print()

    # rules(s)
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
    # reaction_smarts = "[H:1][O:2][C:3][C:4][H:5]>>[H:1][O:2][H:5].[C:3]=[C:4]"
    # reaction_name = "Hydrogenation"
    # rxn_arr = [reaction_smarts,reaction_name]
    # # Convert the reaction SMARTS into GML format rule
    # gml_format_rule = smarts_to_gml(rxn_arr)
    # rule = ruleGMLString(gml_format_rule)



    p = GraphPrinter()
    p.setReactionDefault()
    p.withIndex = True
    for r in inputRules:
        r.print(p)
    
    i=10
    strat = (addSubset(inputGraphs) >> repeat[i](inputRules))
    dg = DG(graphDatabase=inputGraphs)
    dg.build().execute(strat)
    # with dg.build() as b:
    #     b.subset=inputGraphs
    #     # b.apply(inputGraphs,r1,onlyProper=False)
    #     # b.apply(inputGraphs,r2,onlyProper=False)
    #     b.apply(inputGraphs,r3,onlyProper=False)

    # from data import HyperGraphFeaturizer 
    # import pickle  
    # hg_featurizer = HyperGraphFeaturizer()
    # hg_data = hg_featurizer(dg)
    # pickle.dump(hg_data, open(f"/home/mescalin/yitao/Documents/Code/3HPspace/hg_data{i}.pkl", "wb"))
    
    dg.print()

    # flush summary file handle
    post.flushCommands()
    # generate summary/summery.pdf
    subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])


if __name__ == "__main__":
    main()