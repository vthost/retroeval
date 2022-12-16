import pandas as pd
from rxnutils.chem.reaction import ChemicalReaction
from rxnutils.chem.utils import remove_atom_mapping


# preprocess, originally used for uspto-50k
def process(in_file, out_file):  # other columns are id, class
    rxns = pd.read_pickle(in_file)  # list of atom-mapped reaction smiles strings
    p = in_file.replace("_canon","").replace(".pickle", "_ids.pickle")
    ids = pd.read_pickle(p)  # we additionally saved the original ids

    rxns = [ChemicalReaction(rxn) for rxn in rxns]  # [rxn_col]]

    data = pd.DataFrame()
    data["id"] = ids
    data["smiles"] = [rxn.rsmi for rxn in rxns]
    data["retro_template"] = [rxn.generate_reaction_template()[1].smarts for rxn in rxns]  # returns canonical_template, self.retro_template
    data["template_hash"] = [rxn.retro_template.hash_from_smiles() for rxn in rxns]
    data["product"] = [remove_atom_mapping(rxn.products_smiles) for rxn in rxns]
    data["reactants"] = [remove_atom_mapping(rxn.reactants_smiles) for rxn in rxns]

    data.to_csv(out_file, index=False)
# we want: id,smiles,retro_template,template_hash,product,reactants  # level,patent_id,route_id,


if __name__ == "__main__":
    for part in ['train', 'valid', 'test']:
        print()
        print("processing:", part)
        # https://github.com/coleygroup/rxn-ebm/tree/master/rxnebm/data/cleaned_data
        p1 = f"./../rxn-ebm/rxnebm/data/cleaned_data/50k_clean_rxnsmi_noreagent_allmapped_canon_{part}.pickle"
        p2 = f"./data/uspto-50k/uspto-50k-{part}-rxns.csv"
        process(p1, p2)

