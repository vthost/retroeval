from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from argparse import Namespace
import numpy as np
import pandas as pd

from aizynthfinder.chem import SmilesBasedRetroReaction

if TYPE_CHECKING:
    from aizynthfinder.utils.type_utils import Any, Sequence, List, Tuple
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.chem import TreeMolecule
    from aizynthfinder.chem.reaction import RetroReaction

from aizynthfinder.context.policy.expansion_strategies import ExpansionStrategy
from retroeval.model import SingleStepModelWrapper


class RetroWrapperBasedExpansionStrategy(ExpansionStrategy):
    """
    A retroeval-wrapper-based expansion strategy.

    :param key: model.checkpoint
    :param config: the configuration of the tree search
    """

    _required_kwargs = []

    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        super().__init__(key, config, **kwargs)

        self._logger.info(
            f"Creating RetroWrapperBasedExpansionStrategy to {self.key}"
        )

        from retroeval.model.factory import create_single_step_model

        if "topk" not in kwargs:
            kwargs["topk"] = self._config.cutoff_number
        key = key.split('.')
        self.model = create_single_step_model(key[0], checkpoint=key[1], **kwargs)

    # pylint: disable=R0914
    def get_actions(
        self, molecules: Sequence[TreeMolecule]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies and given cutoffs

        :param molecules: the molecules to consider
        :return: the actions and the priors of those actions
        """

        possible_actions = []
        priors = []

        for mol in molecules:
            all_transforms_and_prob = self._predict(mol, self.model)

            probs = np.array(all_transforms_and_prob['scores'], dtype=np.float64)
            probable_transforms_idx = self._cutoff_predictions(probs)

            for rank, idx in enumerate(probable_transforms_idx):
                prob = float(probs[idx].round(4))
                priors += [prob]
                metadata = {"policy_probability": prob,
                            "policy_probability_rank": rank,
                            "policy_name": self.key}
                possible_actions.append(
                    SmilesBasedRetroReaction(
                        mol,
                        reactants_str=all_transforms_and_prob['reactants'][idx],
                        metadata=metadata
                    )
                )
        return possible_actions, priors  # type: ignore

    def _cutoff_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Get the top transformations, by selecting those that have:
            * cumulative probability less than a threshold (cutoff_cumulative)
            * or at most N (cutoff_number)
        """
        sortidx = np.argsort(predictions)[::-1]
        # retroeval: Since different models return very different probabilities better we do not filter based on those
        # cumsum: np.ndarray = np.cumsum(predictions[sortidx])
        # if 0:  #any(cumsum >= self._config.cutoff_cumulative):
        #     maxidx = int(np.argmin(cumsum < self._config.cutoff_cumulative))
        # else:
        maxidx = len(predictions)
        maxidx = min(maxidx, self._config.cutoff_number) or 1
        return sortidx[:maxidx]

    @staticmethod
    def _predict(mol: TreeMolecule, model: SingleStepModelWrapper):  # Note, we do not return an ndarray
        results = None
        try:
            results = model.run(mol.smiles)
        except Exception as e:
            print("Error while running single-step model:", e)
        return {"reactants:": [], "scores": []} if results is None else results
