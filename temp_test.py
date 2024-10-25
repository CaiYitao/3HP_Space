import mod
from typing import List, Optional, Set, Union
import subprocess
import torch
import pandas as pd
import os
from utils import collect_rules


def getReactionCenter(mapRes: mod.DGVertexMapper.Result) -> mod.UnionGraph.Vertex:
    """
    Receive a vertex map from DGVertexMapper and output a set of educt vertices that
    form the reaction center.

    Note, this assumes that we are doing relatively normal chemistry stuff,
    otherwise we would need to iterate through the codomain graph as well to be sure we
    get everything.
    """
    m = mapRes.map
    gl = m.domain

    def findEdge(el: mod.UnionGraph.Edge) -> Optional[mod.UnionGraph.Edge]:
        vlSrc, vlTar = el.source, el.target

        vrSrc, vrTar = m[vlSrc], m[vlTar]

        if not vrSrc or not vrTar:
            return None
        for e in vrSrc.incidentEdges:
            if e.target == vrTar:
                return e
        return None

    res = set()

    for el in gl.edges:

        er = findEdge(el)

        if not er or el.stringLabel != er.stringLabel:
            res.add(el.source)

            res.add(el.target)

    for vl in gl.vertices:
        vr = m[vl]
        if not vr or vl.stringLabel != vr.stringLabel:
            res.add(vl)

    mapped_vertices = []
    for vert in gl.vertices:
        mapped_vertices.append((vert.stringLabel, vert.id))

    print("mapped reactants from get reaction center", mapped_vertices)
    return res


# def get_reaction_center(reactants: List[Union[str, mod.Graph]], rule: mod.Rule) -> torch.tensor:
# 	"""
# 	Receive reactants and a reaction GML rule, and output a binary vector indicating
# 	the reaction center atoms.

# 	Parameters:
# 	- reactants: List of reactant SMILES strings or mod.Graph objects
# 	- rule_gml: GML string representation of the reaction rule

# 	Returns:
# 	- Tensor[int]: Binary vector indicating reaction center atoms (1) and others (0)
# 	"""
# 	# Convert SMILES to mod.Graph if necessary
# 	graphs = [mod.smiles(r,allowAbstract=True) if isinstance(r, str) else r for r in reactants]
#     #get vertices label of mol graph and print


#     # Get and print the labels of vertices in the molecular graph
# 	for g in graphs:
# 		#vertex_labels = {v.stringLabel:v.id for v in g.vertices}
# 		vertex_labels = []
# 		for v in g.vertices:
# 			vertex_labels.append((v.stringLabel, v.id))
# 		print(f"atom labels for reactant {g.linearEncoding} before vertex mapping: {vertex_labels}")


# 	# Create a strategy to filter out reactions with more than 88 carbons on the right side
# 	mstrat = mod.rightPredicate[
# 		lambda der: all(g.vLabelCount('C') <= 88 for g in der.right)
# 	](rule)
# 	total_v = sum(g.numVertices for g in graphs)
# 	# Create a DG and apply the rule
# 	dg = mod.DG()
# 	with dg.build() as b:
# 		res = b.execute(mod.addSubset(graphs) >> mstrat)

# 	# Check if the rule was applied successfully
# 		if len(res.subset) == 0:
# 			# Rule cannot be applied, return all zeros
# 			return torch.tensor([0] * total_v)

# 	for e in dg.edges:

# 		maps = mod.DGVertexMapper(e)

# 		m = next(iter(maps), None)
# 		if m is not None:
# 			vs = getReactionCenter(m)
# 			reaction_center_vector = [v.id for v in vs]
# 			reaction_string = [(v.stringLabel,v.id) for v in vs]
# 			print('reaction_center_string:', reaction_string)

# 		break


# 	binary_reaction_center = torch.zeros(total_v, dtype=torch.int).index_fill_(0, torch.tensor(reaction_center_vector), 1)
# 	# print('total_v:', total_v)
# 	# all_vs = []
# 	# # p = mod.GraphPrinter()
# 	# # p.setReactionDefault()
# 	# # p.withIndex = True

# 	# for g in graphs:
# 	# 	# g.print(p)
# 	# 	for v in g.vertices:
# 	# 		all_vs.append(v)
# 	# 	for e in g.edges:
# 	# 		all_vs.append(e)
# 	# print('all the vertices in the reactants graphs:', all_vs)
# 	# print('all the vertices in the reactants graphs:', [v.stringLabel for v in all_vs])

# 	# print('binary_reaction_center:', binary_reaction_center)

# 	# mod.post.flushCommands()
#         # generate summary/summery.pdf
# 	# subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])
# 	return binary_reaction_center


def get_reaction_center(
    reactants: List[Union[str, mod.Graph]], rule: mod.Rule
) -> torch.tensor:
    """
    Receive reactants and a reaction GML rule, and output a binary vector indicating
    the reaction center atoms.

    Parameters:
    - reactants: List of reactant SMILES strings or mod.Graph objects
    - rule: GML string representation of the reaction rule

    Returns:
    - Tensor[int]: Binary vector indicating reaction center atoms (1) and others (0)
    """
    # Convert SMILES to mod.Graph if necessary
    graphs = [
        mod.smiles(r, allowAbstract=True) if isinstance(r, str) else r
        for r in reactants
    ]

    for r in reactants:
        g = mod.smiles(r, allowAbstract=True)

    total_v = sum(g.numVertices for g in graphs)

    # Create a strategy to filter out reactions with more than 88 carbons on the right side

    mstrat = mod.rightPredicate[
        lambda der: all(g.vLabelCount("C") <= 88 for g in der.right)
    ](rule)

    # Create a DG and apply the rule
    dg = mod.DG()
    with dg.build() as b:
        res = b.execute(mod.addSubset(graphs) >> mstrat)

    # Check if the rule was applied successfully
    if len(res.subset) == 0:
        # Rule cannot be applied, return all zeros
        return torch.tensor([0] * total_v)

    # Find the reaction center vertices using the mapping result

    for e in dg.edges:
        maps = mod.DGVertexMapper(e)
        m = next(iter(maps), None)

        if m is not None:
            external_id_map = {}
            cumulative_offset = 0
            gl = m.map.domain
            mapped_vertices = []
            for vert in gl.vertices:
                mapped_vertices.append((vert.stringLabel, vert.id))

            print("mapped reactants from get reaction center", mapped_vertices)
            # Build the mapping from internal to external IDs
            for g in gl:
                # print("g max external ID", g.maxExternalId)
                # print("g min external ID", g.minExternalId)
                print("g.vertices", [(v.stringLabel, v.id) for v in g.vertices])
                for i in range(g.minExternalId, g.maxExternalId + 1):
                    v = g.getVertexFromExternalId(i)
                    # print("v", v)
                    print(f"v.id: {v.id} label: {v.stringLabel}  i: {i} ")
                    if v:
                        global_id = cumulative_offset + v.id
                        print(
                            f"global_id: {global_id} v.id {v.id}  label: {v.stringLabel}  i: {i}  "
                        )
                        external_id_map[global_id] = i
                cumulative_offset += g.numVertices
                print("cumulative_offset:", cumulative_offset)
            print("external id map", external_id_map)
            test = [(v.stringLabel, v.id) for v in gl.vertices]
            # Get the reaction center vertices
            vs = getReactionCenter(m)
            reaction_string = [(v.stringLabel, v.id) for v in vs]
            print("reaction_center_string:", reaction_string)
            
            reaction_center_vector = []
            for v in vs:
                # Map internal vertex IDs to external if available
                print(
                    f"v.id: {v.id} label: {v.stringLabel}   v.id to external id:{external_id_map.get(v.id)}"
                )
                if external_id_map.get(v.id) is not None:
                    reaction_center_vector.append(external_id_map.get(v.id) - 1)
                else:
                    continue
        break

    # Create the binary vector for the reaction center, based on external IDs if available
    max_ext_id = max(external_id_map.values(), default=total_v)
 
    binary_reaction_center = torch.zeros(max_ext_id, dtype=torch.int)
    print("reaction_center_vector:", reaction_center_vector)
    # Fill the binary vector based on the reaction center vector

    binary_reaction_center.index_fill_(0, torch.tensor(reaction_center_vector), 1)

    return binary_reaction_center


# Example usage:
def main():

    data = pd.read_csv("data/reaction_dataset.csv")

    rule_gml_path = os.path.join(os.getcwd(), "gml_rules")

    rules_dict = {i + 1: rule for i, rule in enumerate(collect_rules(rule_gml_path))}

    reactants_set = data["mapped_reactants"].tolist()
    # reactants_set = data['Reactants'].tolist()
    # applicable_rules = data[]
    rs = [r for r in reactants_set[2].split(".")]
    print("rs", rs)

    # reactants = ['C(C(C)O)(O)=O', '[NAD+]']

    # rule_gml ="""rule[
    # ruleID "R1: (S)-lactate + NAD+ = pyruvate + NADH + H+"
    # left [
    # node [ id 9 label "NAD+"]
    # node [ id 7 label "H"]
    # edge [ source 2 target 3 label "-"]
    # edge [ source 3 target 7 label "-"]
    # edge [ source 2 target 8 label "-"]
    # ]
    # context [
    # node [ id 1 label "C"]
    # node [ id 2 label "C"]
    # node [ id 3 label "O"]
    # node [ id 4 label "C"]
    # node [ id 5 label "O"]
    # node [ id 6 label "O"]
    # node [ id 8 label "H"]
    # edge [ source 1 target 2 label "-"]
    # edge [ source 2 target 4 label "-"]
    # edge [ source 4 target 5 label "="]
    # edge [ source 4 target 6 label "-"]

    # ]
    # right [
    # node [ id 9 label "NAD"]
    # node [ id 7 label "H+"]
    # edge [ source 2 target 3 label "="]
    # edge [ source 8 target 9 label "-"]
    # ]
    # ]
    # """

    # rule = mod.Rule.fromGMLString(rule_gml)
    rule = rules_dict[4]
    result = get_reaction_center(rs, rule)
    # for res,rs in reactants_set:
    # result = get_reaction_center(rs, rule)

    print(result)


if __name__ == "__main__":
    main()
