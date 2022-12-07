
DIR_WRAPPER_ARGS = "./assets/wrapper-args"


# we do not use the model since we evaluate several together
# our current code saves the results in 'model_pred_lst.pkl' inside the below directory
def get_default_ss_results_dir(data, part, exp_id, model=""):
    if part in ["valid", "eval", "train"] and not data.endswith(part):  # latter should not be the case, but who knows
        return f"./results-ss-{exp_id}/{data}-{part}"
    return f"./results-ss-{exp_id}/{data}"


# we do not use the method since we evaluate several together
# our current code saves the results (all aizynthfinder & paroutes output) in a directory `method` in the below one
def get_default_ms_results_dir(data, part, exp_id, method=""):
    post_fix = f"/{method}" if method else ""
    if part in ["valid", "eval", "train"] and not data.endswith(part):  # latter should not be the case
        return f"./results-ms-{exp_id}/{data}-{part}{post_fix}"
    return f"./results-ms-{exp_id}/{data}{post_fix}"


############################################


MODEL_TO_PLOT = {
    "neuralsymt1": "N",
    "neuralsymp": "NP",
    "neuralsympp": "NPP",
    "gln": "GLN",
    "mhn": "MHN",
    "chemformer": "CF",
    "chemformer_lg": "CFL",
    "g2s": "G2S",
    "graphretro": "GR",
    "retroxpert":"RX",
}

MODELS_IN_ORDER = list(MODEL_TO_PLOT.keys())

############################################


DI_COL_PATH = 0
DI_COL_FILE = 1
DI_COL_PROD = 2
DI_COL_REACT = 3
DI_COL_RXN = 4
DI_COL_TPL = 5
# add csv data that doesn't follow our dataset format here, including data columns
# single step datasets
DATASET_INFO = {
    # This is an example how the uspto-50k data from graphretro would be added, we preprocessed that data into our format
    # (https://github.com/vsomnath/graphretro/tree/main/datasets/uspto-50k)
    # 'uspto-50k': {
    #     "test": ("./data/uspto-50k", "canonicalized_test.csv", None, None, "reactants>reagents>production", None)
    # }
}

MULTI_SOL_DATASET_INFO = {
    'uspto-ms': ("./data/uspto-ms", "uspto-ms.pkl")
}

############################################

# are called differently from regular single-step metrics and given entire pandas dataframe

METRICS_ROUTE = ["mss"]