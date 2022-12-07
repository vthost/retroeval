import os
import json


def save_args(path_prefix, args):
    if args:
        json.dump(args.__dict__, open(path_prefix + '_args.json', 'w'), indent=3)


def dict_to_plt(dictionary, path=""):
    s = "x\ty\n"
    dictionary = dict(sorted(dictionary.items()))
    for k, v in dictionary.items():
        s += f"{k}\t{v}\n"

    if path:
        os.makedirs(path[:path.rindex("/")], exist_ok=True)
        with open(path, "w") as f:
            f.write(s)
    return s


def df_to_plot(df, x_col, y_col, path=""):
    s = "x\ty\n"
    for _, r in df.iterrows():
        s += f"{r[x_col]}\t{r[y_col]}\n"

    if path:
        os.makedirs(path[:path.rindex("/")], exist_ok=True)
        with open(path, "w") as f:
            f.write(s)
    return s


def df_to_files(df, file_name, format="csv,latex"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    df[df.select_dtypes(include=['number']).columns] *= 100

    if "csv" in format:
        df.to_csv(file_name + ".csv", index=False, float_format='%.1f')
    if "latex" in format:
        with open(file_name + ".tex", "w") as f:
            f.write(df.to_latex(index=False, float_format='%.1f'))

