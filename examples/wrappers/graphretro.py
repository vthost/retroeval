import torch
import os
from rdkit import RDLogger, Chem
from seq_graph_retro.utils.edit_mol import canonicalize, generate_reac_set
from seq_graph_retro.models import EditLGSeparate
from seq_graph_retro.search import BeamSearch

from retroeval.model import SingleStepModelWrapper
from retroeval.model.factory import create_single_step_model

lg = RDLogger.logger()
lg.setLevel(4)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def canonicalize_prod(p):
    pcanon = canonicalize(p)
    pmol = Chem.MolFromSmiles(pcanon)
    [atom.SetAtomMapNum(atom.GetIdx()+1) for atom in pmol.GetAtoms()]
    p = Chem.MolToSmiles(pmol)
    return p


class GraphRetroWrapper(SingleStepModelWrapper):

    def __init__(self, args):
        args.softmax = True
        super(GraphRetroWrapper, self).__init__(args)

        self.beam_width = 10  #self.topk  # set high (and > self.topk) since outputs many duplicates

        super(GraphRetroWrapper, self).__post_init__()

    def load_edits_model(self, args):
        edits_step = args.edits_step
        edits_loaded = torch.load(os.path.join(args.exp_dir, args.edits_exp,
                                               "checkpoints", edits_step + ".pt"),
                                  map_location=self.device)
        model_name = args.edits_exp.split("_")[0]

        return edits_loaded, model_name

    def load_lg_model(self, args):
        lg_loaded = torch.load(os.path.join(args.exp_dir, args.lg_exp,
                                            "checkpoints", args.lg_step + ".pt"),
                               map_location=self.device)
        model_name = args.lg_exp.split("_")[0]

        return lg_loaded, model_name

    def load_model(self):
        args=self.args
        edits_loaded, edit_net_name = self.load_edits_model(args)
        lg_loaded, lg_net_name = self.load_lg_model(args)

        edits_config = edits_loaded["saveables"]
        lg_config = lg_loaded['saveables']
        # lg_toggles = lg_config['toggles']

        rm = EditLGSeparate(edits_config=edits_config, lg_config=lg_config, edit_net_name=edit_net_name,
                    lg_net_name=lg_net_name, device=self.device)
        rm.load_state_dict(edits_loaded['state'], lg_loaded['state'])
        rm.to(self.device)
        rm.eval()
        beam_model = BeamSearch(model=rm, beam_width=self.beam_width, max_edits=1)
        return beam_model

    def _run(self, mol_smiles: str):
        beam_model = self.model
        reacts, scores = [], []
        try:
        #  beam_model.run_search throws exceptions eg
        #         c_atom_starts = index_select_ND(c_atom, dim=0, index=graph_tensors[-1][:, 0])
        # IndexError: index 0 is out of bounds for dimension 1 with size 0
            p = canonicalize_prod(mol_smiles)
            top_k_nodes = beam_model.run_search(p, max_steps=6)

            for node in top_k_nodes:
                pred_edit = node.edit
                pred_label = node.lg_groups
                if isinstance(pred_edit, list):
                    if not pred_edit:
                        continue
                    pred_edit = pred_edit[0]
                try:
                    pred_set = generate_reac_set(p, pred_edit, pred_label, verbose=False)
                    pred_set = ".".join(pred_set)
                    if pred_set not in reacts:
                        reacts += [pred_set]
                        scores += [node.prob]   # math.exp(node.prob/2)# node.prob is edit + lg prob, both are logs
                        if len(reacts) == self.topk:
                            break
                except BaseException as e:
                    print("Exception while running graphretro (step 2):", e, flush=True)
        except Exception as e:
            print("Exception while running graphretro (step 1):", e, flush=True)

        return {'reactants': reacts, 'scores': scores}


if __name__ == "__main__":
    m = create_single_step_model("graphretro")
    p = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC"
    result = m.run(p)
    print(result)
