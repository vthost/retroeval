import pandas as pd
from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.utils import remove_atom_mapping


# preprocess, originally used for uspto-50k
def process(in_file, out_file, rxn_col="reactants>reagents>production"):  # other columns are id, class
    data = pd.read_csv(in_file)
    data.drop(rxn_col, axis=1)
    rxns = [ChemicalReaction(rxn) for rxn in data[rxn_col]]
    data["smiles"] = [rxn.rsmi for rxn in rxns]
    data["retro_template"] = [rxn.generate_reaction_template()[1].smarts for rxn in rxns]  # returns canonical_template, self.retro_template
    data["template_hash"] = [rxn.retro_template.hash_from_smiles() for rxn in rxns]
    data["product"] = [remove_atom_mapping(rxn.products_smiles) for rxn in rxns]
    data["reactants"] = [remove_atom_mapping(rxn.reactants_smiles) for rxn in rxns]

    data.to_csv(out_file, index=False)
# we want: id,smiles,retro_template,template_hash,product,reactants  # level,patent_id,route_id,


def post_process(file):
    data = pd.read_csv(file)
    print("before duplicate check:", len(data))
    data = data.drop_duplicates(subset='smiles', keep="first")
    print("after duplicate check:", len(data))

    print("before product length check:", len(data))
    data = data[[len(p) > 2 for p in data['product']]]
    print("after product length check:", len(data))

    data.to_csv(file, index=False)

# processing: train
# before duplicate check: 40008
# after duplicate check: 39802
# before product length check: 39802
# after product length check: 39798 > 4
#
# processing: valid
# before duplicate check: 5001
# after duplicate check: 4995
# before product length check: 4995
# after product length check: 4995
#
# processing: test
# before duplicate check: 5007
# after duplicate check: 5005
# before product length check: 5005
# after product length check: 5005


def unique_test_processing():
    name = "uspto-50k"
    data = {}
    for part in ['train', 'valid', 'test']:
        path = f'./data/{name}/{name}-{part}-rxns.csv'
        data[part] = pd.read_csv(path)

    for part in ['valid', 'test']:
        print("before duplicate check", part, len(data[part]))
        data[part] = data[part][~data[part]["smiles"].isin(data['train']["smiles"])]
        print("after duplicate check", part, len(data[part]))

    part = 'test'
    print("before duplicate check", part, len(data[part]))
    data[part] = data[part][~data[part]["smiles"].isin(data['valid']["smiles"])]
    print("after duplicate check", part, len(data[part]))

# before duplicate check valid 4995
# after duplicate check valid 4950 > 45
# before duplicate check test 5005
# after duplicate check test 4955 > 50
# before duplicate check test 4955
# after duplicate check test 4949 > 6
# > These numbers match those from Lin et al.'22, apart from 45 vs. 44,
# but they might have additional filtering for incorrect reactions


if __name__ == "__main__":
    # for part in ['train', 'valid', 'test']:
    #     print()
    #     print("processing:", part)
    #     p1 = f"./data/uspto-50k/canonicalized_{part}.csv"
    #     p2 = f"./data/uspto-50k/uspto-50k-{part}-rxns.csv"
    #     process(p1, p2)
    #
    # for part in ['train', 'valid', 'test']:
    #     print()
    #     print("processing:", part)
    #     p2 = f"./data/uspto-50k/uspto-50k-{part}-rxns.csv"
    #     post_process(p2)
    #
    # unique_test_processing()

    pass
