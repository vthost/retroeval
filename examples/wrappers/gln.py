import sys
import argparse
from retroeval.model.factory import create_single_step_model

if __name__ != "__main__":
    # sys.argv += ['-f_atoms', 'assets/gln/cooked_schneider50k/atom_list.txt']  - done in factory now
    # this one must be reloaded with f_atoms before loading any gnn functionality, otherwise node feaature dims are wrong
    import GLN.gln.mods.mol_gnn.mg_clib.mg_lib as mg_lib
    import GLN.gln.test.model_inference as model_inference
    from GLN.gln.test.model_inference import RetroGLN

from retroeval.model import SingleStepModelWrapper


class GLNWrapper(SingleStepModelWrapper):

    def __init__(self, args):
        super(GLNWrapper, self).__init__(args)
        self.model_file = args.model_file
        self.template_file = args.template_file  # original GLN's cmd_args.dropbox
        self.beam_size = self.topk

        super(GLNWrapper, self).__post_init__()

    def load_model(self):
        return RetroGLN(self.template_file, self.model_file)

    def _run(self, mol_smiles: str):
        results = self.model.run(mol_smiles, self.beam_size, self.topk)
        if results is None:
            return {'reactants': [], 'scores': []}
        results['scores'] = results['scores'].tolist()
        return results


if __name__ == "__main__":
    m = create_single_step_model("gln")
    result = m.run("CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC")
    print(result)
