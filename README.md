# Evaluating Retrosynthesis Models

This repository provides simple tools to run and evaluate single-step retrosynthesis models in both the single and the multi-step scenario.
The code accompanies the paper [Retrosynthesis Prediction Revisited](https://openreview.net/pdf?id=kLzFuf4GoC-) (Tu et al., AI4Science @NeurIPS'22).


We created a wrapper interface for single-step models and also provide code that makes it usable in the multi-step tool [AIZynthFinder](https://github.com/MolecularAI/aizynthfinder). Then we use [PARoutes](https://github.com/MolecularAI/PaRoutes) to analyze the results. 
Examples for some models suggested in the literature and a short explanation on how to add your own wrapper are provided in [`examples/wrappers`](examples/wrappers). 

*Edit Note:
When recomputing the numbers for the paper, we observed that we had accidentally used the wrong templates
in the last two charts in Fig. 4 in our original calculations. The correct numbers are 93%/7% in both.*

## Setup 

We use a [conda](https://www.anaconda.com/) environment `retroeval` and provide [a setup script](scripts/setup.sh) which creates that. 
Before running the setup, please adapt the `pytorch-cuda` version to yours (see [here](https://pytorch.org/)). 
Also note that the script contains commands to install the model packages for our example wrappers. 
These dependencies are commented out per default since they are not needed, a simple MLP (i.e., NeuralSym as suggested by Segler and Waller, 2017) [is provided in this repository](examples/models/README.md).
Note that, to run a custom wrapper, you might have to install additional dependencies. 

Then run:
[``./scripts/setup.sh``](scripts/setup.sh)

We tested the environment on Ubuntu and Mac OS (Intel).


## Data & Models

All datasets and checkpoints we created are available [here](http://neurips.s3.us-east.cloud-object-storage.appdomain.cloud/index.html).

Run [``./scripts/download.sh``](scripts/download.sh) to obtain the datasets. Since the full rt and rd datasets are larger (each nearly 1GB), they are per default commented out in the script.

**Please Note**
We will make USPTO-ms available soon as well.

Run ``./scripts/checkpoints.sh`` to obtain the checkpoints we used.
Please see [`examples/wrappers`](examples/wrappers/README.md) for links to the code and descriptions provided by others.
## Usage

The [`scripts`](scripts) directory contains the code we used for dataset creation, analysis, and, especially, the following example commands for the experiments we ran.
Please note that the `.sh` files contain only basic options, see the python scripts in `retroeval` and [`retroeval/utils/config.py`](retroeval/utils/config.py) for more configuration options.
### Single-step Run 
`./scripts/ssrun.sh $SSMODEL $CHECKPOINT $EVALDATA`
  - `$SSMODEL` the model, it must be registered through a JSON description in [`assets/wrapper-args`](assets/wrapper-args). Our examples cover `{chemformer, chemformer_lg, g2s, graphretro, mlp}`, see also [`examples/wrappers`](examples/wrappers/README.md).
  - `$CHECKPOINT` is used to locate the checkpoint in the model's JSON description; `{uspto-50k, rt, rd}` are available for most example models (i.e., we use the training data to identify them, but you can choose a descriptor of your choice). 
  - `$EVALDATA` the dataset you want to run over (now), for evaluation, one of `{uspto-50k, rt-1k, rd-1k, rt, rd, uspto-ms}`. 
  If you want to use custom data, see `DATASET_INFO` in [`retroeval/utils/config.py`](retroeval/utils/config.py).

  Per default, the results are stored in `results-ss-${EXP_ID}/${EVALDATA}/${SSMODEL}.pkl`, where we use `CHECKPOINT` as `$EXP_ID`.
### Single-step Evaluation
`./scripts/sseval.sh $SSMODELS $EXP_ID $EVALDATA `
  - `$SSMODELS` the models, comma separated (e.g., `graphretro,mlp`).
  - `$EXP_ID` used to locate the results. 

The command is similar for our dataset with multiple solutions per product:
 
`./scripts/sseval_multi.sh $SSMODELS $EXP_ID $EVALDATA`

### Multi-step Run
`./scripts/msrun.sh $AIZCONFIG "${MODEL}.${CHECKPOINT}" $TARGETS $STOCK`
  - `$AIZCONFIG` the configuration file for AIZynthFinder (e.g., [`assets/aizynthfinder/config.yml`](assets/aizynthfinder/config.yml)).
  - `"${MODEL}.${CHECKPOINT}"` is our format for a wrapper-based 'policy' for AIZynthFinder. It must be registered through a JSON description in [`assets/wrapper-args`](assets/wrapper-args) exactly as for the single-step experiments; and, additionally, 
through a simple line in the configuration for AIZynthfinder, under the item `policy`. You can add additional arguments for the wrapper's constructor under the latter item, however, the ones in the JSON description take precendence.

  - `$TARGETS` the list of molecules for which routes should be found, one of `{rt-tpl-100, rt-1k, rd-1k}`.
  - `$STOCK` the file containing the molecules assumed to be purchasable, one of `{rt-tpl-100, rt-1k, rd-1k}`. Note that custom ones must be registered in the configuration. 
  
  See the [AIZynthfinder documentation](https://molecularai.github.io/aizynthfinder/) for more information about the latter three artifacts.

  The results are stored in `results-ms-${EXP_ID}/${TARGETS}/${POLICY}_${SEARCH_ALGO}` per default, 
  where we again use `$CHECKPOINT` as `$EXP_ID`, and `${SEARCH_ALGO}` describes the search algorithm used (default here `Retro*` (Chen et al.,2020)). 

### Multi-step Evaluation
The results are evaluated using PARoutes:

`./scripts/mseval.sh $POLICIES $TARGETS $EXP_ID` 

- `$POLICIES` comma-separated string of policies `${POLICY}_${SEARCH_ALGO}` as above; if empty (`""`), all results in the result directory (default `results-ms-${EXP_ID}/${TARGETS}`) are evaluated.


## Issues, Suggestions?

Please open an issue. There are definitely multiple possible improvements and extensions, 
many of which we would like to see ourselves here in this repository in the future. 
We maintain a list in [`assets/docs`](assets/docs/extensions.txt).
Ideas are welcome!


## References

`(Tu et al., 2022)`
Hongyu Tu, Shantam Shorewala,  Tengfei Ma, and Veronika Thost. Retrosynthesis Prediction Revisited. NeurIPS 2022 Workshop, AI for Science: Progress and Promises, 2022.
</br>
`(Segler and Waller, 2017)`
Marwin HS Segler and Mark P Waller. Neural-symbolic machine learning for retrosynthesis and reaction prediction. Chemistry–A European Journal, 23(25):5966–5971, 2017.
</br>
`(Chen et al., 2020)`
Binghong Chen et al. "Retro*: learning retrosynthetic planning with neural guided A* search." International Conference on Machine Learning. PMLR, 2020.
