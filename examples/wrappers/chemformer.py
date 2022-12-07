import torch
import math
import argparse
from retroeval.model import SingleStepModelWrapper
from retroeval.model.factory import create_single_step_model
import molbart.util as util
from molbart.decoder import DecodeSampler


def check_seq_len(tokens, mask, max_seq_len):
    """ Warn user and shorten sequence if the tokens are too long, otherwise return original

    Args:
        tokens (List[List[str]]): List of token sequences
        mask (List[List[int]]): List of mask sequences

    Returns:
        tokens (List[List[str]]): List of token sequences (shortened, if necessary)
        mask (List[List[int]]): List of mask sequences (shortened, if necessary)
    """

    seq_len = max([len(ts) for ts in tokens])
    if seq_len > max_seq_len:
        print(f"WARNING -- Sequence length {seq_len} is larger than maximum sequence size")

        tokens_short = [ts[:max_seq_len] for ts in tokens]
        mask_short = [ms[:max_seq_len] for ms in mask]

        return tokens_short, mask_short

    return tokens, mask


class ChemformerWrapper(SingleStepModelWrapper):

    def __init__(self, args):
        args.canonicalize = True
        args.softmax = True
        super(ChemformerWrapper, self).__init__(args)
        # copy and set system-specific arguments
        self.args.model_path = args.model_file
        self.args.vocab_path = "models/chemformer/bart_vocab.txt"
        self.args.chem_token_start_idx = util.DEFAULT_CHEM_TOKEN_START

        self.tokeniser = None  # will be initialized during loading

        super(ChemformerWrapper, self).__post_init__()

    def collate_data(self, target_smiles):
        if not isinstance(target_smiles, list):
            target_smiles = [target_smiles]

        prods_output = self.tokeniser.tokenise(target_smiles, pad=True)

        prods_tokens = prods_output["original_tokens"]
        prods_mask = prods_output["original_pad_masks"]
        prods_tokens, prods_mask = check_seq_len(prods_tokens, prods_mask, self.model.max_seq_len)

        prods_token_ids = self.tokeniser.convert_tokens_to_ids(prods_tokens)
        prods_token_ids = torch.tensor(prods_token_ids).transpose(0, 1)
        prods_pad_mask = torch.tensor(prods_mask, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            "encoder_input": prods_token_ids,
            "encoder_pad_mask": prods_pad_mask,
        }
        return collate_output

    def load_model(self):
        self.tokeniser = util.load_tokeniser(self.args.vocab_path, self.args.chem_token_start_idx)
        sampler = DecodeSampler(self.tokeniser, util.DEFAULT_MAX_SEQ_LEN)

        model = util.load_bart(self.args, sampler)
        model.num_beams = self.topk
        sampler.max_seq_len = model.max_seq_len
        return model

    def _run_all(self, mol_smiles_lst):
        data = self.collate_data(mol_smiles_lst)
        device_batch = {key: val.to(self.device) if type(val) == torch.Tensor else val for key, val in data.items()}

        model = self.model.to(self.device)

        with torch.no_grad():
            smiles, log_lhs = model.sample_molecules(device_batch, sampling_alg="beam")

        results = []
        for i, reacts in enumerate(smiles):
            results += [{"reactants": reacts, "scores": [p for p in log_lhs[i]]}]

        return results

    def _run(self, mol_smiles: str):
        return self._run_all([mol_smiles])[0]


if __name__ == "__main__":
    m = create_single_step_model("chemformer")
    p = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC"
    result = m.run(p)
    print(result)
