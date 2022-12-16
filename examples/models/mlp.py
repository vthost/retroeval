import torch.nn as nn
from examples.models.template import TemplateModel
from examples.utils.model import NONLINEARITY
from examples.utils.mol import FP, calc_fp_dim, encode_smiles


class RetroMLP(TemplateModel):

    def __init__(self, fp_dim, hidden_dim, out_dim, templates=None, fp_types=FP.values(), fp_radii=[], layers=1, activation="elu",
                 batch_norm=True, dropout=None):
        super(RetroMLP, self).__init__(out_dim, templates)

        self.fp_dim = fp_dim
        self.fp_types = fp_types
        self.fp_radii = fp_radii

        in_dim = calc_fp_dim(fp_dim, fp_types)
        modules = []

        for l in range(layers):
            modules += [nn.Linear(hidden_dim if l else in_dim, hidden_dim if l < layers-1 else out_dim)]
            if batch_norm:
                modules += [nn.BatchNorm1d(hidden_dim if l < layers-1 else out_dim)]
            modules += [NONLINEARITY[activation]] if l < layers-1 else [nn.Softmax(-1)]
            if dropout is not None:
                modules += [nn.Dropout(dropout)]

        self.layers = nn.Sequential(*modules)

    def encode_smiles(self, mol_smiles_list):
        return encode_smiles(mol_smiles_list, self.fp_dim, fp_types=self.fp_types, fp_radii=self.fp_radii)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":

    model = RetroMLP(128, 128, 8)
    trg = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC"
    fp = model.encode_smiles([trg, trg])  # need two samples to avoid batch norm error
    result = model.forward(fp)
    print("success")  # we need to set templates to use "run"
