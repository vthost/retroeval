import torch
import numpy as np
from Graph2SMILES.models.graph2seq_series_rel import Graph2SeqSeriesRel
from torch.utils.data import DataLoader
from Graph2SMILES.utils.data_utils import G2SDataset as G2S1
from Graph2SMILES.utils.data_utils import get_graph_features_from_smi, load_vocab, tokenize_smiles, len2idx
from Graph2SMILES.preprocess import get_token_ids

from retroeval.model import SingleStepModelWrapper
from retroeval.model.factory import create_single_step_model


class G2SDataset(G2S1):
    def __init__(self, args, src_lines):
        self.args = args

        self.a_scopes = []
        self.b_scopes = []
        self.a_features = []
        self.b_features = []
        self.a_graphs = []
        self.b_graphs = []
        self.a_scopes_lens = []
        self.b_scopes_lens = []
        self.a_features_lens = []
        self.b_features_lens = []

        self.src_token_ids = []             # loaded but not batched
        self.src_lens = []
        # self.tgt_token_ids = []
        # self.tgt_lens = []

        self.data_indices = []
        self.batch_sizes = []
        self.batch_starts = []
        self.batch_ends = []

        self.vocab = load_vocab(args.vocab_file)
        self.vocab_tokens = [k for k, v in sorted(self.vocab.items(), key=lambda tup: tup[1])]

        # logging.info(f"Loading preprocessed features from {file}")
        feat = preprocess(src_lines, self.vocab, 512)  # todo
        for attr in ["a_scopes", "b_scopes", "a_features", "b_features", "a_graphs", "b_graphs",
                     "a_scopes_lens", "b_scopes_lens", "a_features_lens", "b_features_lens",
                     "src_token_ids", "src_lens", "tgt_token_ids", "tgt_lens"]:
            setattr(self, attr, feat[attr])

        # mask out chiral tag (as UNSPECIFIED)
        self.a_features[:, 6] = 2

        assert len(self.a_scopes_lens) == len(self.b_scopes_lens) == \
               len(self.a_features_lens) == len(self.b_features_lens) == \
               len(self.src_token_ids) == len(self.src_lens), \
               f"Lengths of source and target mismatch!"

        self.a_scopes_indices = len2idx(self.a_scopes_lens)
        self.b_scopes_indices = len2idx(self.b_scopes_lens)
        self.a_features_indices = len2idx(self.a_features_lens)
        self.b_features_indices = len2idx(self.b_features_lens)

        del self.a_scopes_lens, self.b_scopes_lens, self.a_features_lens, self.b_features_lens

        self.data_size = len(self.src_token_ids)
        self.data_indices = np.arange(self.data_size)

        # logging.info(f"Loaded and initialized G2SDataset, size: {self.data_size}")


def get_seq_features_from_line(src_line, max_src_len):  # -> Tuple[np.ndarray, int, np.ndarray, int]:
    global G_vocab

    src_tokens = src_line.strip().split()
    if not src_tokens:
        src_tokens = ["C", "C"]             # hardcode to ignore

    src_token_ids, src_lens = get_token_ids(src_tokens, G_vocab, max_len=max_src_len)

    src_token_ids = np.array(src_token_ids, dtype=np.int32)

    return src_token_ids, src_lens


def prep_g2s(src_lines, max_src_len: int, num_workers: int = 1):

    seq_features_and_lengths = [
        get_seq_features_from_line(src_line, max_src_len) for src_line in src_lines
    ]

    seq_features_and_lengths = list(seq_features_and_lengths)

    # logging.info(f"Done seq featurization, time: {time.time() - start}. Collating")
    src_token_ids, src_lens = zip(*seq_features_and_lengths)

    src_token_ids = np.stack(src_token_ids, axis=0)
    src_lens = np.array(src_lens, dtype=np.int32)

    graph_features_and_lengths = [get_graph_features_from_smi((i, "".join(src_line.split()), False))
                                  for i, src_line in enumerate(src_lines)]

    graph_features_and_lengths = list(graph_features_and_lengths)
    # logging.info(f"Done graph featurization, time: {time.time() - start}. Collating and saving...")
    a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, a_features, a_features_lens, \
        b_features, b_features_lens, a_graphs, b_graphs = zip(*graph_features_and_lengths)

    a_scopes = np.concatenate(a_scopes, axis=0)
    b_scopes = np.concatenate(b_scopes, axis=0)
    a_features = np.concatenate(a_features, axis=0)
    b_features = np.concatenate(b_features, axis=0)
    a_graphs = np.concatenate(a_graphs, axis=0)
    b_graphs = np.concatenate(b_graphs, axis=0)

    a_scopes_lens = np.array(a_scopes_lens, dtype=np.int32)
    b_scopes_lens = np.array(b_scopes_lens, dtype=np.int32)
    a_features_lens = np.array(a_features_lens, dtype=np.int32)
    b_features_lens = np.array(b_features_lens, dtype=np.int32)

    return {
        'src_token_ids': src_token_ids,
        'src_lens': src_lens,
        'tgt_token_ids': np.array([[]*len(src_lens)], dtype=np.int32),
        'tgt_lens': np.array([512]*len(src_lens), dtype=np.int32),
        'a_scopes': a_scopes,
        'b_scopes': b_scopes,
        'a_features': a_features,
        'b_features': b_features,
        'a_graphs': a_graphs,
        'b_graphs': b_graphs,
        'a_scopes_lens': a_scopes_lens,
        'b_scopes_lens': b_scopes_lens,
        'a_features_lens': a_features_lens,
        'b_features_lens': b_features_lens
    }


def tokenize(line):

    tokenize_line = tokenize_smiles

    line = "".join(line.strip().split())
    return tokenize_line(line)


def preprocess(smiless, vocab, max_src_len):
    smiless = [tokenize(smiles) for smiles in smiless]

    global G_vocab
    G_vocab = vocab

    return prep_g2s(smiless, max_src_len=max_src_len)


class Graph2SMILESWrapper(SingleStepModelWrapper):
    def __init__(self, args):
        args.canonicalize = True
        args.softmax = True
        super(Graph2SMILESWrapper, self).__init__(args)

        self.model_type = args.model_type
        self.model_file = args.model_file

        self.mpn_type = "dgcn"
        self.batch_type = 'tokens'
        self.predict_batch_size = 4096
        self.beam_size = 30
        self.temperature = 1.0
        self.predict_min_len = 1
        self.predict_max_len = 512

        super(Graph2SMILESWrapper, self).__post_init__()
        
    def load_model(self):
        if torch.cuda.is_available():
            state = torch.load(self.model_file)
        else:
            state = torch.load(self.model_file, map_location=torch.device('cpu'))
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]
        self.pretrain_args = pretrain_args

        for attr in ["mpn_type", "rel_pos"]:
            try:
                getattr(pretrain_args, attr)
            except AttributeError:
                pass

        assert self.model_type == pretrain_args.model, f"Pretrained model is {pretrain_args.model}!"
        model_class = Graph2SeqSeriesRel
        self.dataset_class = G2SDataset
        # if self.model_type == "s2s":
        #     model_class = Seq2Seq
        #     self.dataset_class = S2SDataset
        # elif self.model_type == "g2s_series_rel":
        #     model_class = Graph2SeqSeriesRel
        #     self.dataset_class = G2SDataset
        #     # args.compute_graph_distance = True
        #     # assert args.compute_graph_distance
        # else:
        #     raise ValueError(f"Model {args.model} not supported!")

        vocab = load_vocab(pretrain_args.vocab_file)
        self.vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

        model = model_class(pretrain_args, vocab)
        model.load_state_dict(pretrain_state_dict)

        return model

    def _run_all(self, mol_smiles_lst):

        test_dataset = self.dataset_class(self.pretrain_args, mol_smiles_lst)
        test_dataset.batch(
            batch_type=self.batch_type,
            batch_size=self.predict_batch_size
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )

        all_predictions = []
        with torch.no_grad():
            for test_idx, test_batch in enumerate(test_loader):

                test_batch.to(self.device)
                results = self.model.predict_step(
                    reaction_batch=test_batch,
                    batch_size=test_batch.size,
                    beam_size=self.beam_size,
                    n_best=self.topk*3,
                    temperature=self.temperature,
                    min_length=self.predict_min_len,
                    max_length=self.predict_max_len
                )

                for i, predictions in enumerate(results["predictions"]):
                    smis = []
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [self.vocab_tokens[idx] for idx in predicted_idx[:-1]]
                        smi = "".join(predicted_tokens)
                        smis.append(smi)
                    scores = [s.item() for s in results["scores"][i]]

                    all_predictions += [{'reactants': smis, 'scores': scores}]

        return all_predictions

    def _run(self, mol_smiles: str):
        return self._run_all([mol_smiles])[0]


if __name__ == "__main__":
    m = create_single_step_model("g2s")
    p = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC"
    result = m.run(p)
    print(result)



