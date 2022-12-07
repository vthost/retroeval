# PaRoutes

We use PaRoutes for analysis as provided at https://github.com/MolecularAI/PaRoutes.

For our data extraction, we made slight changes from the procedure suggested in `PaRoutes/setup`:

- `extract_uspto_data.py`: we commented all code filtering the data according to the number of template occurrences
- `find_non_overlaps.py`: we took all routes from all patents (instead of only one route per patent; we filter later but only for the test sets)
- In the very last step we use adapted route-selection scripts provided in this directory

The entire extraction procedure is given in `run.sh`.