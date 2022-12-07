from abc import ABCMeta, abstractmethod


# This interface is currently only conceptually here and not intended to be subclassed directly.
# Please use SingleStepModelWrapper.
class SingleStepModel(metaclass=ABCMeta):

    # return a dictionary minimally containing this keys/information, SORTED please (highest score first)
    # return {'reactants': list of reactants-smiles strings, 'scores': list-of-float-scores},
    # one string and score per result
    @abstractmethod
    def run(self, mol_smiles: str):
        raise NotImplementedError()

    # extra method since for some models batch processing is more efficient when several samples are given
    # in this case overwrite this
    def run_all(self, mol_smiles_list):
        return list(map(self.run, mol_smiles_list))
