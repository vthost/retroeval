import logging
import torch.cuda
from abc import abstractmethod

from retroeval.model.model import SingleStepModel
from retroeval.utils.mol import canonicalize_all


class SingleStepModelWrapper(SingleStepModel):

    def __init__(self, args):
        super(SingleStepModelWrapper, self).__init__()
        self.args = args
        self.model = None

        self.topk = args.topk if hasattr(args, 'topk') else 50
        self.softmax = args.softmax if hasattr(args, 'softmax') else True

        if torch.cuda.is_available():
            self.device = f"cuda:{args.gpu}" if hasattr(args, 'gpu') and args.gpu >= 0 else "cuda:0"
        else:
            self.device = "cpu"

        use_prob_thr = args.use_prob_thr if hasattr(args, 'use_prob_thr') else False
        prob_thr = args.prob_thr if hasattr(args, 'prob_thr') else 0.0
        # may assume done by model in case not set;
        # esp includes check if result can be canonicalized at all and filters out if not
        # If you want to use it, e.g., set args.canonicalize = True before calling super(YourWrapper, self)__init__()
        canonicalize = args.canonicalize if hasattr(args, 'canonicalize') else True

        def result_filter(reactants_smiles, prob):
            # optional filtering, not used in our evaluation
            if use_prob_thr and prob < prob_thr:
                return reactants_smiles, prob, False

            if canonicalize:
                reactants_smiles = canonicalize_all(reactants_smiles)
            if not reactants_smiles:  # None if could be canonicalized
                return "", prob, False

            return reactants_smiles, prob, True

        self.result_filter = result_filter

    def __post_init__(self):
        logging.info(f"Loading model dump")
        self.model = self.load_model()

        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()  # we assume a trained model

    # return the model (eg an instance of torch.module) which can be used as self.model
    # and then referenced in run_model
    @abstractmethod
    def load_model(self):
        pass

    # return a dictionary minimally containing this information, SORTED please (highest score first):
    # return {'reactants': list of reactant-smiles-strings, 'scores': list-of-float-scores}
    @abstractmethod
    def _run(self, mol_smiles: str):
        pass

    # models for which batch processing is more efficient should overwrite this
    # and then may call it in _run(...): return self._run_all([mol_smiles])
    def _run_all(self, mol_smiles_list):
        return list(map(self._run, mol_smiles_list))

    def _post_process_result(self, result):
        scores = result['scores']

        # may add something to support unsorted results, but this creates overhead...
        # scores = result['scores']
        # rs = result['reactants']
        # idx = np.argsort(np.array(scores))[::-1]
        # results = [(rs[i], scores[i]) for i in idx]

        results = zip(result['reactants'], scores)
        results = map(lambda r: self.result_filter(*r), results)  # for better efficiency this could be done (and stopped) within the below loop

        reacts, scores = [], []
        for react, score, res in results:
            if not score: continue
            if not res or \
                    react in reacts: continue  # filter result & for duplicates
            reacts += [react]
            scores += [score]

            if len(reacts) == self.topk:  # better don't assume the model does this
                break

        if self.softmax:
            scores = torch.nn.Softmax(-1)(torch.Tensor(scores)).tolist()

        return {'reactants': reacts, 'scores': scores}

    def run(self, mol_smiles: str):
        result = self._run(mol_smiles)  # we assume results is sorted
        return self._post_process_result(result)

    def run_all(self, mol_smiles_list):
        results = self._run_all(mol_smiles_list)
        return list(map(self._post_process_result, results))