import torch
from abc import ABCMeta, abstractmethod

from retroeval.utils.template import TemplateRunner


class TemplateModel(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, out_dim, templates=None, max_try=0):  # for evaluation, templates must be set
        super(TemplateModel, self).__init__()

        self.out_dim = out_dim
        self.i_unk = out_dim - 1  # hint to UNK template
        self.templates = templates
        self.max_try = max_try

        assert templates is None or len(templates) + 1 == out_dim  # + 1 for unknown

    @abstractmethod
    def encode_smiles(self, mol_smiles_list):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def run(self, mol_smiles):
        return self.run_all([mol_smiles])[0]

    # we use this as main functionality since we assume the model's forward is more efficient on batches
    def run_all(self, mol_smiles_list, topk=10):
        x = self.encode_smiles(mol_smiles_list).to(next(self.parameters()).device)
        x = self.forward(x)
        x, idx_batch = torch.sort(x, dim=-1, descending=True)
        results = []
        max_try = x.shape[-1] if not self.max_try else min(x.shape[-1], self.max_try)
        for i, idx in enumerate(idx_batch):
            n_try, n_succ = 0, 0
            reacts, tpl_idx, scores = [], [], []
            while n_succ < topk and n_try < max_try:
                i_pred = idx[n_try]
                n_try += 1

                if i_pred == self.i_unk:
                    continue

                result = TemplateRunner.run(mol_smiles_list[i], self.templates[i_pred])
                if result is not None and result:
                    n_succ += 1
                    reacts += result
                    scores += [x[i][n_try-1]] * len(result)
                    tpl_idx += [i_pred] * len(result)

            results += [{"reactants": reacts, "templates": tpl_idx, "scores": scores}]

        return results


if __name__ == "__main__":
    from examples.utils.model import load_model
    model = load_model(f"models/mlp/mlp_uspto-50k_epoch_132.pt", config=0)
    trg = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC"
    fs = model.run_all([trg, trg])