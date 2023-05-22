from typing import Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from hisel import hsic, select, categorical
from collections.abc import Mapping

LassoSelection = select.Selection


class Parameters(Mapping):
    def __iter__(self):
        if not hasattr(self, '__dataclass_fields__'):
            raise StopIteration
        for v in self.__dataclass_fields__:
            yield v

    def __len__(self):
        if not hasattr(self, '__dataclass_fields__'):
            return 0
        return len(self.__dataclass_fields__)

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class SearchParameters(Parameters):
    num_permutations: Optional[int] = None
    im_ratio: float = .05
    max_iter: int = 2
    parallel: bool = True
    random_state: Optional[int] = None


@dataclass
class HSICLassoParameters(Parameters):
    mi_threshold: float = .0001
    hsic_threshold: float = .01
    batch_size: int = 5000
    minibatch_size: int = 200
    number_of_epochs: int = 4
    use_preselection: bool = True
    device: Optional[str] = None


continuous_dtypes = [
    float,
    np.float32,
    np.float64,
]

discrete_dtypes = [
    bool,
    int,
    np.int32,
    np.int64,
]


def select_features(
        xdf: pd.DataFrame,
        ydf: Union[pd.DataFrame, pd.Series],
        hsiclasso_parameters: Optional[HSICLassoParameters] = None,
        categorical_search_parameters: Optional[SearchParameters] = None,
):
    n, d = xdf.shape
    continuous_features = [
        col for col in xdf.columns
        if xdf[col].dtype in continuous_dtypes
    ]
    discrete_features = [
        col for col in xdf.columns
        if xdf[col].dtype in discrete_dtypes
    ]

    if hsiclasso_parameters is None:
        hsiclasso_parameters = HSICLassoParameters()
    if categorical_search_parameters is None:
        categorical_search_parameters = SearchParameters()
    continuous_lasso_selection: LassoSelection = select.select(
        xdf[continuous_features], ydf, **hsiclasso_parameters)

    categorical_search_selection: categorical.Selection = categorical.select(
        xdf[discrete_features], ydf, **categorical_search_parameters)

    selected_features = categorical_search_selection.features + \
        continuous_lasso_selection.features
    results = dict(
        continuous_lasso_selection=continuous_lasso_selection,
        categorical_search_selection=categorical_search_selection,
        selected_features=selected_features
    )
    return results
