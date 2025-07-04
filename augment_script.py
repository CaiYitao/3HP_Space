import random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol

import chemTech.utils.helpers as helpers
from chemTech.utils.helpers import MolConversionError


def run_rxn(s0, rxn_smarts):
    try:
        m0 = helpers.mol_from_smiles(s0)
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

        # run reaction and randomly pick an output molecule
        products = rxn.RunReactants((m0,))
        if len(products) == 0:
            return ''
        m1 = products[random.randint(0, len(products) - 1)][0]

        fix_aromaticity(m1)

        Chem.SanitizeMol(m1)
        s1 = Chem.MolToSmiles(m1)
    except MolConversionError:
        print(f'Mol Conversion Error. There is an error in the smiles string: {s0}')
        return ''
    except Chem.KekulizeException:
        print(f'Could not kekulize: Returned empty str. Original: {s0}')
        return ''
    except ValueError:
        print(f'Value Error exception for: {s0} returned empty str.')
        return ''

    return s1 if s1 != s0 else ''


def fix_aromaticity(m1):
    for a in m1.GetAromaticAtoms():
        if not a.IsInRing():
            a.SetIsAromatic(False)
            for b in a.GetBonds():
                if not b.IsInRing():
                    b.SetBondType(Chem.rdchem.BondType.SINGLE)


def pos_trivalent_00(s0: str, src: str, tar: str) -> str:
    rxn_smarts = f'[C:1][{src};D2:2]=[C:3] >> [C:1][{tar}:2]=[C:3]'
    return run_rxn(s0, rxn_smarts)


rxn_000 = '[#6h:1] >> [#6:1]F'
rxn_001 = '[#6h:1] >> [#6:1]Cl'
rxn_002 = '[#6:1]F >> [#6:1]Cl'
rxn_003 = '[#6:1]F >> [#6:1][H]'
rxn_004 = '[#6:1]Cl >> [#6:1][H]'
rxn_005 = '[#6:1]Cl >> [#6:1]F'

rxn_006 = '[#6:1][OH] >> [#6:1][NH2]'
rxn_007 = '[#6:1][OH] >> [#6:1]NC'
rxn_008 = '[#6:1][OH] >> [#6:1]S'
rxn_009 = '[#6:1][NH2] >> [#6:1]O'
rxn_010 = '[#6:1][NH2] >> [#6:1]NC'
rxn_011 = '[#6:1][NH2] >> [#6:1]S'
rxn_012 = '[#6:1][NH][CH3] >> [#6:1]O'
rxn_013 = '[#6:1][NH][CH3] >> [#6:1][NH2]'
rxn_014 = '[#6:1][NH][CH3] >> [#6:1]S'
rxn_015 = '[#6:1][SH] >> [#6:1]O'
rxn_016 = '[#6:1][SH] >> [#6:1][NH2]'
rxn_017 = '[#6:1][SH] >> [#6:1]NC'

rxn_018 = '[C:1][O;D2:2][C:3] >> [C:1][S:2][C:3]'
rxn_019 = '[C:1][O;D2:2][C:3] >> [C:1][C:2][C:3]'
rxn_020 = '[C:1][O;D2:2][C:3] >> [C:1][N:2][C:3]'
rxn_021 = '[C:1][S;D2:2][C:3] >> [C:1][O:2][C:3]'
rxn_022 = '[C:1][S;D2:2][C:3] >> [C:1][C:2][C:3]'
rxn_023 = '[C:1][S;D2:2][C:3] >> [C:1][N:2][C:3]'
rxn_024 = '[C:1][C;D2:2][C:3] >> [C:1][O:2][C:3]'
rxn_025 = '[C:1][C;D2:2][C:3] >> [C:1][S:2][C:3]'
rxn_026 = '[C:1][C;D2:2][C:3] >> [C:1][N:2][C:3]'
rxn_027 = '[C:1][N;D2:2][C:3] >> [C:1][O:2][C:3]'
rxn_028 = '[C:1][N;D2:2][C:3] >> [C:1][S:2][C:3]'
rxn_029 = '[C:1][N;D2:2][C:3] >> [C:1][C:2][C:3]'

rxn_030 = '[C:1][N;D2:2]=[C:3] >> [C:1][C:2]=[C:3]'
rxn_031 = '[C:1][N;D2:2]=[C:3] >> [C:1][P:2]=[C:3]'
rxn_032 = '[C:1][C;D2:2]=[C:3] >> [C:1][N:2]=[C:3]'
rxn_033 = '[C:1][C;D2:2]=[C:3] >> [C:1][P:2]=[C:3]'
rxn_034 = '[C:1][P;D2:2]=[C:3] >> [C:1][N:2]=[C:3]'
rxn_035 = '[C:1][P;D2:2]=[C:3] >> [C:1][C:2]=[C:3]'

rxn_036 = '[C:1][C:2]#[N:3] >> [C:1][N+:2]#[C-:3]'
rxn_036_2 = pos_trivalent_00

rxn_037 = '[#6:1][C:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][Si:2]([#6:3])([#6:4])([#6:5])'
rxn_038 = '[#6:1][C:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][N+:2]([#6:3])([#6:4])([#6:5])'
rxn_039 = '[#6:1][C:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][P+:2]([#6:3])([#6:4])([#6:5])'
rxn_040 = '[#6:1][Si:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][C:2]([#6:3])([#6:4])([#6:5])'
rxn_041 = '[#6:1][Si:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][N+:2]([#6:3])([#6:4])([#6:5])'
rxn_042 = '[#6:1][Si:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][P+:2]([#6:3])([#6:4])([#6:5])'
rxn_043 = '[#6:1][N+:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][C:2]([#6:3])([#6:4])([#6:5])'
rxn_044 = '[#6:1][N+:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][Si:2]([#6:3])([#6:4])([#6:5])'
rxn_045 = '[#6:1][N+:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][P+:2]([#6:3])([#6:4])([#6:5])'
rxn_046 = '[#6:1][P+:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][C+:2]([#6:3])([#6:4])([#6:5])'
rxn_047 = '[#6:1][P+:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][Si+:2]([#6:3])([#6:4])([#6:5])'
rxn_048 = '[#6:1][P+:2]([#6:3])([#6:4])([#6:5]) >> [#6:1][N+:2]([#6:3])([#6:4])([#6:5])'

rxn_049 = '[C;R:1][C;R:2]=[C;R][C;R:4] >> [C;R:1][S;R:2][C;R:4]'
rxn_050 = '[C;R:1][S;R;D2:2][C;R:4] >> [C:1][C:2]=[C][C:4]'

rxn_051 = '[#6:1][C:2](=O)O >> [#6:1][S:2](=O)(=O)O'
rxn_052 = '[#6:1][S:2](=O)(=O)[O:3] >> [#6:1][S:2](=O)(=O)[N:3]'

rxn_053 = '[C:1][C:2](=[O:3])[C:4] >> [C:1][C:2](=[S:3])[C:4]'
rxn_054 = '[C:1][C:2](=[O:3])[C:4] >> [C:1][C:2](=[N:3]C)[C:4]'
rxn_055 = '[C:1][C:2](=[O:3])[C:4] >> [C:1][C:2](=[S:3](F)(F))[C:4]'
rxn_056 = '[C:1][C:2](=[S;D1:3])[C:4] >> [C:1][C:2](=[O:3])[C:4]'
rxn_057 = '[C:1][C:2](=[S;D1:3])[C:4] >> [C:1][C:2](=[N:3]C)[C:4]'
rxn_058 = '[C:1][C:2](=[S;D1:3])[C:4] >> [C:1][C:2](=[C:3](F)(F))[C:4]'
rxn_059 = '[C:1][C:2](=[N;D2:3]C)[C:4] >> [C:1][C:2](=[O:3])[C:4]'
rxn_060 = '[C:1][C:2](=[N;D2:3]C)[C:4] >> [C:1][C:2](=[S:3])[C:4]'
rxn_061 = '[C:1][C:2](=[N;D2:3]C)[C:4] >> [C:1][C:2](=[C:3](F)(F))[C:4]'

rxn_062 = '[#6:1][C:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[O:4][C:5][#6:6]'
rxn_063 = '[#6:1][C:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2]([F:3])=[N:4][C:5][#6:6]'
rxn_064 = '[#6:1][C:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][S:2](=[O:3])(=O)[N:4][C:5][#6:6]'
rxn_065 = '[#6:1][C:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2]([C:3](F)(F)(F))[N:4][C:5][#6:6]'
rxn_066 = '[#6:1][C:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2](=[S:3])[N:4][C:5][#6:6]'
rxn_067 = '[#6:1][C:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][P:2](=[O:3])[N:4][C:5][#6:6]'
rxn_068 = '[#6:1][C:2](=[O:3])[O:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[N:4][C:5][#6:6]'
rxn_069 = '[#6:1][C:2](=[O:3])[O:4][C:5][#6:6] >> [#6:1][C:2]([F:3])=[N:4][C:5][#6:6]'
rxn_070 = '[#6:1][C:2](=[O:3])[O:4][C:5][#6:6] >> [#6:1][S:2](=[O:3])(=O)[N:4][C:5][#6:6]'
rxn_071 = '[#6:1][C:2](=[O:3])[O:4][C:5][#6:6] >> [#6:1][C:2]([C:3](F)(F)(F))[N:4][C:5][#6:6]'
rxn_072 = '[#6:1][C:2](=[O:3])[O:4][C:5][#6:6] >> [#6:1][C:2](=[S:3])[N:4][C:5][#6:6]'
rxn_073 = '[#6:1][C:2](=[O:3])[O:4][C:5][#6:6] >> [#6:1][P:2](=[O:3])[N:4][C:5][#6:6]'
rxn_074 = '[#6:1][C:2]([F:3])=[N:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[N:4][C:5][#6:6]'
rxn_075 = '[#6:1][C:2]([F:3])=[N:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[O:4][C:5][#6:6]'
rxn_076 = '[#6:1][C:2]([F:3])=[N:4][C:5][#6:6] >> [#6:1][S:2](=[O:3])(=O)[N:4][C:5][#6:6]'
rxn_077 = '[#6:1][C:2]([F:3])=[N:4][C:5][#6:6] >> [#6:1][C:2]([C:3](F)(F)(F))[N:4][C:5][#6:6]'
rxn_078 = '[#6:1][C:2]([F:3])=[N:4][C:5][#6:6] >> [#6:1][C:2](=[S:3])[N:4][C:5][#6:6]'
rxn_079 = '[#6:1][C:2]([F:3])=[N:4][C:5][#6:6] >> [#6:1][P:2](=[O:3])[N:4][C:5][#6:6]'
rxn_080 = '[#6:1][S:2](=[O:3])(=O)[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[N:4][C:5][#6:6]'
rxn_081 = '[#6:1][S:2](=[O:3])(=O)[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[O:4][C:5][#6:6]'
rxn_082 = '[#6:1][S:2](=[O:3])(=O)[NH:4][C:5][#6:6] >> [#6:1][C:2]([F:3])=[N:4][C:5][#6:6]'
rxn_083 = '[#6:1][S:2](=[O:3])(=O)[NH:4][C:5][#6:6] >> [#6:1][C:2]([C:3](F)(F)(F))[N:4][C:5][#6:6]'
rxn_084 = '[#6:1][S:2](=[O:3])(=O)[NH:4][C:5][#6:6] >> [#6:1][C:2](=[S:3])[N:4][C:5][#6:6]'
rxn_085 = '[#6:1][S:2](=[O:3])(=O)[NH:4][C:5][#6:6] >> [#6:1][P:2](=[O:3])[N:4][C:5][#6:6]'
rxn_086 = '[#6:1][CH:2]([C:3](F)(F)(F))[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[N:4][C:5][#6:6]'
rxn_087 = '[#6:1][CH:2]([C:3](F)(F)(F))[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[O:4][C:5][#6:6]'
rxn_088 = '[#6:1][CH:2]([C:3](F)(F)(F))[NH:4][C:5][#6:6] >> [#6:1][C:2]([F:3])=[N:4][C:5][#6:6]'
rxn_089 = '[#6:1][CH:2]([C:3](F)(F)(F))[NH:4][C:5][#6:6] >> [#6:1][S:2](=[O:3])(=O)[N:4][C:5][#6:6]'
rxn_090 = '[#6:1][CH:2]([C:3](F)(F)(F))[NH:4][C:5][#6:6] >> [#6:1][C:2](=[S:3])[N:4][C:5][#6:6]'
rxn_091 = '[#6:1][CH:2]([C:3](F)(F)(F))[NH:4][C:5][#6:6] >> [#6:1][P:2](=[O:3])[N:4][C:5][#6:6]'
rxn_092 = '[#6:1][C:2](=[S;D1:3])[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[N:4][C:5][#6:6]'
rxn_093 = '[#6:1][C:2](=[S;D1:3])[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[O:4][C:5][#6:6]'
rxn_094 = '[#6:1][C:2](=[S;D1:3])[NH:4][C:5][#6:6] >> [#6:1][C:2]([F:3])=[N:4][C:5][#6:6]'
rxn_095 = '[#6:1][C:2](=[S;D1:3])[NH:4][C:5][#6:6] >> [#6:1][S:2](=[O:3])(=O)[N:4][C:5][#6:6]'
rxn_096 = '[#6:1][C:2](=[S;D1:3])[NH:4][C:5][#6:6] >> [#6:1][C:2]([C:3](F)(F)(F))[N:4][C:5][#6:6]'
rxn_097 = '[#6:1][C:2](=[S;D1:3])[NH:4][C:5][#6:6] >> [#6:1][P:2](=[O:3])[N:4][C:5][#6:6]'
rxn_098 = '[#6:1][PH:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[N:4][C:5][#6:6]'
rxn_099 = '[#6:1][PH:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2](=[O:3])[O:4][C:5][#6:6]'
rxn_100 = '[#6:1][PH:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2]([F:3])=[N:4][C:5][#6:6]'
rxn_101 = '[#6:1][PH:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][S:2](=[O:3])(=O)[N:4][C:5][#6:6]'
rxn_102 = '[#6:1][PH:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2]([C:3](F)(F)(F))[N:4][C:5][#6:6]'
rxn_103 = '[#6:1][PH:2](=[O:3])[NH:4][C:5][#6:6] >> [#6:1][C:2](=[S:3])[N:4][C:5][#6:6]'

rxn_104 = '[#6:1][c:2]1[n;D2:3][n;D2:4][c:5]([#6:7])[o:6]1 >> [#6:1][n:2]1[n:3][n:4][c:5]([#6:7])[c:6]1'
rxn_105 = '[#6:1][c:2]1[n;D2:3][n;D2:4][c:5]([#6:7])[o:6]1 >> [#6:1][c:2]1[n:3][o:4][c:5]([#6:7])[n:6]1'
rxn_106 = '[#6:1][c:2]1[n;D2:3][n;D2:4][c:5]([#6:7])[o:6]1 >> [#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[nH:6]1'
rxn_107 = '[#6:1][n:2]1[n:3][n:4][c:5]([#6:7])[c;D2:6]1 >> [#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[o:6]1'
rxn_108 = '[#6:1][n:2]1[n:3][n:4][c:5]([#6:7])[c;D2:6]1 >> [#6:1][c:2]1[n:3][o:4][c:5]([#6:7])[n:6]1'
rxn_109 = '[#6:1][n:2]1[n:3][n:4][c:5]([#6:7])[c;D2:6]1 >> [#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[nH:6]1'
rxn_110 = '[#6:1][c:2]1[n:3][o:4][c:5]([#6:7])[n:6]1 >> [#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[o:6]1'
rxn_111 = '[#6:1][c:2]1[n:3][o:4][c:5]([#6:7])[n:6]1 >> [#6:1][n:2]1[n:3][n:4][c:5]([#6:7])[c:6]1'
rxn_112 = '[#6:1][c:2]1[n:3][o:4][c:5]([#6:7])[n:6]1 >> [#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[nH:6]1'
rxn_113 = '[#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[nH:6]1 >> [#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[o:6]1'
rxn_114 = '[#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[nH:6]1 >> [#6:1][n:2]1[n:3][n:4][c:5]([#6:7])[c:6]1'
rxn_115 = '[#6:1][c:2]1[n:3][n:4][c:5]([#6:7])[nH:6]1 >> [#6:1][c:2]1[n:3][o:4][c:5]([#6:7])[n:6]1'

rxn_116 = '[#6:1][C:2](=[O:3])[H] >> [#6:1][C:2](=[S:3])[H]'
rxn_117 = '[#6:1][C:2](=[O:3])[H] >> [#6:1][C:2](=[N:3]C)[H]'
rxn_118 = '[#6:1][C:2](=[O:3])[H] >> [#6:1][C:2](=[C:3](F)(F))[H]'
rxn_119 = '[#6:1][C:2](=[S;D1:3])[H] >> [#6:1][C:2](=[O:3])[H]'
rxn_120 = '[#6:1][C:2](=[S;D1:3])[H] >> [#6:1][C:2](=[N:3]C)[H]'
rxn_121 = '[#6:1][C:2](=[S;D1:3])[H] >> [#6:1][C:2](=[C:3](F)(F))[H]'
rxn_122 = '[#6:1][C:2](=[C:3](F)(F))[H] >> [#6:1][C:2](=[O:3])[H]'
rxn_123 = '[#6:1][C:2](=[C:3](F)(F))[H] >> [#6:1][C:2](=[S:3])[H]'
rxn_124 = '[#6:1][C:2](=[C:3](F)(F))[H] >> [#6:1][C:2](=[N:3]C)[H]'

rxn_125 = '[c:1]1[c:2][c:3][c;D2:4][c;D2][c:5]1 >> [c:1]1[c:2][c:3][s:4][c:5]1'
rxn_126 = '[c:1]1[c:2][c:3][s:4][c:5]1 >> [c:1]1[c:2][c:3][c:4][c][c:5]1'
rxn_127 = '[c:1]1[c:2][c:3][c;D2:4][c:5][c:6]1 >> [c:1]1[c:2][c:3][n:4][c:5][c:6]1'
rxn_128 = '[c:1]1[c:2][c:3][n;D2;!H:4][c:5][c:6]1 >> [c:1]1[c:2][c:3][c:4][c:5][c:6]1'
rxn_129 = '[c:1]1[c;D2:2][c:3][c;D2:4][c:5][c:6]1 >> [c:1]1[n:2][c:3][n:4][c:5][c:6]1'
rxn_130 = '[c:1]1[n;D2;!H:2][c:3][n;D2;!H:4][c:5][c:6]1 >> [c:1]1[c:2][c:3][c:4][c:5][c:6]1'


pos_aug = []
indices = [f'{i:03}' for i in range(131)]
for idx in indices:
    fun_def = f'def pos_{idx}(s0: str) -> str:\n   rxn_smarts = rxn_{idx}\n   return run_rxn(s0, rxn_smarts)'
    exec(fun_def)
    pos_aug.append(eval(f'pos_{idx}'))
