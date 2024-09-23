import mod
from typing import List, Union
import subprocess

def get_reaction_center(reactants: List[Union[str, mod.Graph]], rule_gml: mod.Rule) -> List[int]:
    """
    Receive reactants and a reaction GML rule, and output a binary vector indicating
    the reaction center atoms.

    Parameters:
    - reactants: List of reactant SMILES strings or mod.Graph objects
    - rule_gml: GML string representation of the reaction rule

    Returns:
    - List[int]: Binary vector indicating reaction center atoms (1) and others (0)
    """
    # Convert SMILES to mod.Graph if necessary
    graphs = [mod.smiles(r, add=False) if isinstance(r, str) else r for r in reactants]
    print('graphs:', graphs)
    # Create the rule from GML
    rule = mod.Rule.fromGMLString(rule_gml)
    print('rule:', rule)
    # Create a DG and add the reactants
    mstrat = mod.rightPredicate[
        lambda der: all(g.vLabelCount('C') <= 88 for g in der.right)
    ](rule)

    # Create a DG and apply the rule
    dg = mod.DG()
    with dg.build() as b:
        res = b.execute(mod.addSubset(graphs) >> mstrat)

        print('res subset:', res.subset)
    # Check if the rule was applied successfully
        if len(res.subset) == 0:
            # Rule cannot be applied, return all zeros
            return [0] * sum(g.numVertices for g in graphs)
    dg.print()
 
    # Rule can be applied, create the binary vector
    binary_vector = []
    for graph in graphs:
        graph_vector = [0] * graph.numVertices
        for edge in dg.edges:
            if graph in edge.sources:
                vertex_mapper = mod.DGVertexMapper(edge)
                #get all the mapping info
                mapping_info = vertex_mapper.Result
                print('mapping_info:', (mapping_info.map,mapping_info.rule))
      
                
                left_graph = vertex_mapper.left
                right_graph = vertex_mapper.right
                #lets print the left and right graph with mod_post
                p = mod.GraphPrinter()
                p.setReactionDefault()
                p.withIndex = True
                for g in left_graph:
                    g.print(p)
                for g in right_graph:
                    g.print(p)




                print('left_graph:', left_graph)
                #get all the info of the left graph
                left_graph_info = []
                for v in left_graph.vertices:
                    left_graph_info.append((v.stringLabel, v.degree, v.id))
                print('left_graph_info:', left_graph_info)
                print('right_graph:', right_graph)
                right_graph_info = []
                for v in right_graph.vertices:
                    right_graph_info.append((v.stringLabel, v.degree, v.id))
                print('right_graph_info:', right_graph_info)

                for result in vertex_mapper:
                    for v in graph.vertices:
                        left_v = left_graph.vertices[v.id]
          
                        #get all the info of the left graph
                        left_v_info = left_v.stringLabel, left_v.degree, left_v.id
                        # print('left_v_info:', left_v_info)
                        if result.map[left_v] != left_v:
                            # Check if the vertex is actually changed in the reaction
                            right_v = result.map[left_v]
                            # print('right_v:', right_v)
                            #get all the info of the right graph
                            right_v_info = right_v.stringLabel, right_v.degree, right_v.id
                            # print('right_v_info:', right_v_info)
                    

                            if (left_v.stringLabel != right_v.stringLabel or
                                left_v.degree != right_v.degree):
                                graph_vector[v.id] = 1

         
        binary_vector.extend(graph_vector)
    # mod.post.flushCommands()
    #     # generate summary/summery.pdf
    # subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])
    return binary_vector

# Example usage:
def main():
    reactants = ["CC(=O)C(=O)O"]  
    rule_gml = """rule[
 ruleID "pyruvate = acetaldehyde + CO2   {r}  || ec num:  4.1.1.1"
 
 left [
   edge [ source 2 target 4 label "-"]
   edge [ source 4 target 6 label "-"]
   edge [ source 6 target 7 label "-"]
 ]
 
 context [
   node [ id 1 label "C"]
   node [ id 2 label "C"]
   node [ id 3 label "O"]
   node [ id 4 label "C"]
   node [ id 5 label "O"]
   node [ id 6 label "O"]
   node [ id 7 label "H"]
   edge [ source 1 target 2 label "-"]
   edge [ source 2 target 3 label "="]
   edge [ source 4 target 5 label "="]
 ]
 
 right [
   edge [ source 2 target 7 label "-"]
   edge [ source 4 target 6 label "="]
 ]
]"""
    result = get_reaction_center(reactants, rule_gml)
    print(result)

if __name__ == "__main__":
    main()