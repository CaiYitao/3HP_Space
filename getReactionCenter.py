import mod
from typing import List, Optional, Set, Union
import subprocess
import torch

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


	return res


def get_reaction_center(reactants: List[Union[str, mod.Graph]], rule: mod.Rule) -> torch.tensor:
	"""
	Receive reactants and a reaction GML rule, and output a binary vector indicating
	the reaction center atoms.

	Parameters:
	- reactants: List of reactant SMILES strings or mod.Graph objects
	- rule_gml: GML string representation of the reaction rule

	Returns:
	- Tensor[int]: Binary vector indicating reaction center atoms (1) and others (0)
	"""
	# Convert SMILES to mod.Graph if necessary
	graphs = [mod.smiles(r,allowAbstract=True) if isinstance(r, str) else r for r in reactants]

	# Create a strategy to filter out reactions with more than 88 carbons on the right side
	mstrat = mod.rightPredicate[
		lambda der: all(g.vLabelCount('C') <= 88 for g in der.right)
	](rule)
	total_v = sum(g.numVertices for g in graphs)
	# Create a DG and apply the rule
	dg = mod.DG()
	with dg.build() as b:
		res = b.execute(mod.addSubset(graphs) >> mstrat)

	# Check if the rule was applied successfully
		if len(res.subset) == 0:
			# Rule cannot be applied, return all zeros
			return torch.tensor([0] * total_v)

	for e in dg.edges:

		maps = mod.DGVertexMapper(e)

		m = next(iter(maps), None)
		if m is not None:
			external_id_map = {}
            
			cumulative_offset = 0
            
			gl = m.map.domain
			
			for g in gl:

				for i in range(g.minExternalId, g.maxExternalId + 1):
					v = g.getVertexFromExternalId(i)

					if v:
						global_id = cumulative_offset + v.id

						external_id_map[global_id] = i
				cumulative_offset += g.numVertices
			vs = getReactionCenter(m)
			reaction_center_vector = []
			for v in vs:
				if external_id_map.get(v.id) is not None:
					reaction_center_vector.append(external_id_map.get(v.id) - 1)
				else:
					continue
			# reaction_center_vector = [v.id for v in vs]

					
		break

    
	binary_reaction_center = torch.zeros(total_v, dtype=torch.int).index_fill_(0, torch.tensor(reaction_center_vector), 1)
	# print('total_v:', total_v)
	# all_vs = []
	# # p = mod.GraphPrinter()
	# # p.setReactionDefault()
	# # p.withIndex = True
    
	# for g in graphs:
	# 	# g.print(p)
	# 	for v in g.vertices:
	# 		all_vs.append(v)
	# 	for e in g.edges:
	# 		all_vs.append(e)
	# print('all the vertices in the reactants graphs:', all_vs)	
	# print('all the vertices in the reactants graphs:', [v.stringLabel for v in all_vs])
	
	# print('binary_reaction_center:', binary_reaction_center)
		
	# mod.post.flushCommands()
        # generate summary/summery.pdf
	# subprocess.run(["/home/talax/xtof/local/Mod/bin/mod_post"])
	return binary_reaction_center

# Example usage:
def main():
	reactants = ["C[C@@H](C(=O)O)O","[NAD+]"]  
	rule_gml = """rule[
	ruleID "R1: (S)-lactate + NAD+ = pyruvate + NADH + H+" 
	left [
	node [ id 9 label "NAD+"]
	node [ id 7 label "H"]
	edge [ source 2 target 3 label "-"]
	edge [ source 3 target 7 label "-"]
	edge [ source 2 target 8 label "-"]
	] 
	context [
	node [ id 1 label "C"]
	node [ id 2 label "C"]
	node [ id 3 label "O"]
	node [ id 4 label "C"]
	node [ id 5 label "O"]
	node [ id 6 label "O"]
	node [ id 8 label "H"]
	edge [ source 1 target 2 label "-"]
	edge [ source 2 target 4 label "-"]
	edge [ source 4 target 5 label "="]
	edge [ source 4 target 6 label "-"]

	] 
	right [
	node [ id 9 label "NAD"]
	node [ id 7 label "H+"]
	edge [ source 2 target 3 label "="]
	edge [ source 8 target 9 label "-"]
	]
	]
	"""
    
	
	
	rule = mod.Rule.fromGMLString(rule_gml)
    
	result = get_reaction_center(reactants, rule)
    
	print(result)
	




if __name__ == "__main__":
	main()

