# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
from examples.utils.model import load_model
from retroeval.model import SingleStepModelWrapper
from retroeval.model.factory import create_single_step_model


# for checkpoints
# (1) for subclasses of examples.template.TemplateModel
# (2) created with examples.utils.model.save_model
class TemplateModelWrapper(SingleStepModelWrapper):
    def __init__(self, args):
        super(TemplateModelWrapper, self).__init__(args)

        self.model_file = args.model_file

        super(TemplateModelWrapper, self).__post_init__()
        
    def load_model(self):
        return load_model(self.model_file)

    def _run_all(self, mol_smiles_list):
        return self.model.run_all(mol_smiles_list)

    # put the logic into run all because for a list it runs more effectively that way
    # and we don't want to replicate it for a single molecule
    def _run(self, mol_smiles: str):
        return self._run_all([mol_smiles])[0]


if __name__ == "__main__":
    m = create_single_step_model("mlp")
    p = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC"
    result = m.run(p)
    print(result)
