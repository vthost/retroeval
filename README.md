# Evaluating Retrosynthesis Models

This repository provides simple tools to run and evaluate single-step retrosynthesis models in both the single and the multi-step scenario.
The code accompanies the paper [Retrosynthesis Prediction Revisited](https://openreview.net/pdf?id=kLzFuf4GoC-) (Tu et al., AI4Science @NeurIPS'22).


We created a wrapper interface for single-step models and also provide code that makes it usable in the multi-step tool [AIZynthFinder](https://github.com/MolecularAI/aizynthfinder). Then we use [PARoutes](https://github.com/MolecularAI/PaRoutes) to analyze the results. 
Examples for some models suggested in the literature and a short explanation on how to add your own wrapper are provided in [`examples/wrappers`](examples/wrappers). 


**Please Note**
We will make all datasets and model information available soon.


## Setup 

We use a [conda](https://www.anaconda.com/) environment `retroeval` and provide a setup script which creates that. 
Before running the [setup](scripts/setup.sh), please check the `cudatoolkit` version. 
Also note that the script installs the model packages for our example wrappers. 
If you do not want to have all these dependencies, comment out the last part of the script. 
They are not needed, a simple MLP (i.e., NeuralSym as suggested by Segler and Waller, 2017) is provided in this repository.
Note that, to run a custom wrapper, you might have to install additional dependencies. 

Then run:
``./scripts/setup.sh``

We tested the environment on Ubuntu and Mac OS (Intel).


## Data & Models

Coming soon.

[//]: # (This repository contains [our smaller test sets]&#40;data&#41; and [checkpoints for MLP].)

[//]: # ()
[//]: # (Run ``./scripts/download.sh`` to obtain the full datasets and checkpoints for all models. Note that the latter together require around XX of storage, therefore they are per default commented out in the script.)

[//]: # ()
[//]: # ([//]: # &#40;The data and models&#41;)
[//]: # ()
[//]: # ([//]: # &#40;- The data we used can be downloaded [here]&#40;TODO-LINK&#41;, please move it into this repository at root level.&#41;)
[//]: # ()
[//]: # ([//]: # &#40;- We provide our model checkpoints for the examples [here]&#40;TODO-LINK&#41;, please move them into this repository at root level &#40;or create a symbolic link to somewhere external, the full file is large `ln -s $EXTERNAL/models models`&#41;. &#41;)
[//]: # (The configurations we used are documented in [`assets/docs`]&#40;a-todo/exp_config.txt&#41;. )

[//]: # ()
[//]: # ([//]: # &#40;Please see [`examples/wrappers`]&#40;examples/wrappers/README.md&#41; for links to the checkpoints provided by others.&#41;)

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
`./scripts/sseval.sh $SSMODELS $EVALDATA $EXP_ID`
  - `$SSMODELS` the models, comma separated (e.g., `graphretro,mlp`).
  - `$EXP_ID` used to locate the results. 

The command is similar for our dataset with multiple solutions per product:
 
`./scripts/sseval_multi.sh $SSMODELS $EVALDATA $EXP_ID`

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