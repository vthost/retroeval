import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import functools
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader

from examples.utils.data import load_dataset
from examples.utils.model import init_model, load_model, save_model, delete_model
from examples.models.mlp import RetroMLP
from retroeval.utils.file import save_args, get_metrics_str

print = functools.partial(print, flush=True)


def top_k_batch(preds, gt, ks=[1]):
    probs, idx = torch.topk(preds, k=max(ks))
    p_idxs = idx.cpu().numpy().tolist()
    t_idx = gt.cpu().numpy().tolist()

    ks = sorted(ks)
    top_ks = [0]*len(ks)
    for p_i, p_idx in enumerate(p_idxs):
        for i, k in enumerate(ks):
            if t_idx[p_i] in p_idx[:k]:
                top_ks[i] += 1
                if k < ks[-1]:
                    for j, k2 in enumerate(ks[i+1:]):
                        top_ks[i+1+j] += 1
                break
    return top_ks


class SingleStepFPDataset(Dataset):
    def __init__(self, X_fp, y):
        super(SingleStepFPDataset, self).__init__()
        self.X_fp = X_fp
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_fp[idx], self.y[idx]


def train_epoch(model, data, optimizer, device, loss_fn, it):
    losses = []
    model.train()
    for X_batch, y_batch in tqdm(data):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        p_batch = model.forward(X_batch)
        loss_v = loss_fn(p_batch, y_batch)
        loss_v.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        losses.append(loss_v.item())
        it.set_postfix(loss=np.mean(losses[-10:]) if losses else None)
    return losses


def eval_epoch(model, data, device, loss_fn, ks=[1, 3, 5, 10]):
    model.eval()
    loss = 0.0
    topks = []
    for X_batch, y_batch in tqdm(data):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        with torch.no_grad():
            y_hat = model.forward(X_batch)
            loss += loss_fn(y_hat, y_batch).item()
            topks += [torch.Tensor(top_k_batch(y_hat, y_batch, ks))]

    loss = loss / len(data.dataset)
    topks = torch.sum(torch.stack(topks, dim=0), dim=0)/len(data.dataset)

    r = {f"top_{k}": topks[i].item() for i, k in enumerate(ks)}
    r["loss"] = loss
    return r


# assumes model points to be a TemplateModel,
# and its method encode_smiles to return a fingerprint (i.e., not affected by learning)
def train(model, model_config,
          dataset,
          save_prefix="",
          batch_size=1024,
          lr=0.0005,
          epochs=10000,
          patience=50,
          weight_decay=0.0,
          train_with_valid=True,
          eval_metric="top_1",  # needs to be something which is better if higher
          ks=[1, 3, 5, 10],
          checkpoint=None,
          args=None,
          cache_data=True,
          verbose=True):

    if save_prefix:
        os.makedirs(save_prefix[:save_prefix.rindex('/')], exist_ok=True)
        save_args(save_prefix, args)

    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data, _, tpl_idx, tpls = load_dataset(dataset, tpls=True)

    if checkpoint:
        model, model_config = load_model(checkpoint)
    else:
        mod, cls = model[:model.rindex(".")], model[model.rindex(".")+1:]
        model_config = json.load(open(model_config, "r"))
        model_config["out_dim"] = len(tpls) + 1  # account for unknown test templates
        model = init_model(mod, cls, model_config)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, verbose=verbose)

    path = f"./data/{dataset}/{dataset}_{model_config['fp_dim']}" \
           f"_{'.'.join(model_config['fp_types'])}" \
           f"_{'.'.join(list(map(str, model_config['fp_radii']))) if isinstance(model_config['fp_radii'], list) else model_config['fp_radii']}.pt"
    if os.path.isfile(path):
        data = torch.load(path)
    else:
        data = {part: model.encode_smiles(dat) for part, dat in data.items()}
        if cache_data:
            torch.save(data, path)

    if train_with_valid:
        data["train"] = torch.cat([data["train"], data["valid"]], dim=0)
        tpl_idx["train"] = tpl_idx["train"] + tpl_idx["valid"]
    for part in ["train", "valid", "test"]:
        data[part] = DataLoader(SingleStepFPDataset(data[part], tpl_idx[part]),
                                batch_size=batch_size, shuffle=part == 'train')

    best_epoch, best_val = -1, -1
    it = trange(epochs)
    for e in it:
        train_epoch(model, data["train"], optimizer, device, loss_fn, it)
        # trai = eval_epoch(model, data["train"], device, loss_fn, ks)
        eval = eval_epoch(model, data["valid"], device, loss_fn, ks)
        test = eval_epoch(model, data["test"], device, loss_fn, ks)

        scheduler.step(eval["loss"])
        if eval[eval_metric] > best_val:
            if save_prefix:
                save_model(model, model_config, f"{save_prefix}_epoch_{e}.pt")
                delete_model(f"{save_prefix}_epoch_{best_epoch}.pt")
            best_epoch = e
            best_val = eval[eval_metric]  #eval["loss"]  #eval["top_1"] + eval["top_10"]
            if verbose:
                print(f"New best model at epoch {e}!")

        if verbose:
            # print(f"e{e} train >> " + get_metrics_str(trai))
            print(f"e{e} valid >> " + get_metrics_str(eval))
            print(f"e{e} test >> " + get_metrics_str(test))
            print()

        if e > best_epoch + patience:
            print(f"Stopping at epoch {e}! \nBest model: {best_val} (at e{best_epoch})")
            if save_prefix:
                model, model_config = load_model(f"{save_prefix}_epoch_{best_epoch}.pt", config=True)
                model_config["templates"] = tpls  # not needed in train but later for running, hence only save here
                save_model(model, model_config, f"{save_prefix}_epoch_{best_epoch}.pt")
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="training for retrosynthesis models")
    parser.add_argument('--model', default='examples.models.mlp.RetroMLP',
                        type=str, help="specify the model's module and class" )
    parser.add_argument('--model_config', default='./examples/configs/mlp.json',
                        type=str, help="specify the model's config file containing the arguments for the constructor" )
    parser.add_argument('--data', default='uspto-50k',
                        type=str, help='specify the dataset')
    parser.add_argument('--train_with_valid', default=1, type=int,
                        help="train with valid data")
    parser.add_argument('--cache', default=1, type=int,
                        help="save data preprocessed")
    parser.add_argument('--save_dir', default='./models/mlp/',
                        type=str, help='specify where to save the trained models')
    parser.add_argument('--exp_id', default='',
                        type=str, help='specify an id for the file name')
    parser.add_argument('--checkpoint', default="",
                        type=str, help='specify a checkpoint to load')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help="specify the batch size")
    parser.add_argument('--lr', default=0.0005, type=float,
                        help="specify the learning rate")
    parser.add_argument('--wd', default=0.0, type=float,
                        help="specify the weight decay")
    parser.add_argument('--patience', default=50, type=int,
                        help="specify the patience")
    parser.add_argument('--epochs', default=10000, type=int,
                        help="specify the max number of epochs")
    parser.add_argument('--verbose', default=1, type=int,
                        help="print out")

    args = parser.parse_args()

    train(args.model,
          args.model_config,
          args.data,
          save_prefix=args.save_dir + f"/mlp_{args.data}{args.exp_id}" if args.save_dir else "",
          batch_size=args.batch_size,
          lr=args.lr,
          weight_decay=args.wd,
          epochs=args.epochs,
          train_with_valid=args.train_with_valid,
          checkpoint=args.checkpoint,
          args=args,
          cache_data=args.cache,
          verbose=args.verbose)

