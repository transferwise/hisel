import pandas as pd
import hisel
from arfs.feature_selection.allrelevant import Leshy
from xgboost import XGBRegressor


def print_results(selection, method):
    print('\n\n##########################################################')
    print(
        f'The following features have been selected using {method}:')
    print(f'{selection}')


def main():
    df = pd.read_csv('mydata.csv')
    # df is expected with categorical variables already encoded as integers
    # and possibly with continuous variables normalised.
    # The one-dimensional target is expected to be in hte last column of the dataframe.
    xdf = df.iloc[:, :-1]
    yser = df.iloc[:, -1]

    # Plain KSG-based selection
    selection, scores = hisel.select.ksgmi(xdf, yser, threshold=.01)
    print_results(selection, 'KSG')

    # HISEL with parameters specification
    categorical_search_parameters = hisel.feature_selection.SearchParameters(
        num_permutations=1,
        im_ratio=.03,
        max_iter=2,
        parallel=True,
        random_state=None,
    )
    hsiclasso_parameters = hisel.feature_selection.HSICLassoParameters(
        mi_threshold=.00001,
        hsic_threshold=0.005,
        batch_size=5000,
        minibatch_size=500,
        number_of_epochs=3,
        use_preselection=True,
        device=hisel.kernels.Device.CPU  # if cuda is available you can pass GPU
    )
    results = hisel.feature_selection.select_features(
        xdf, yser, hsiclasso_parameters, categorical_search_parameters)
    print_results(results.selected_features, 'HISEL')

    # Selection with Boruta
    n_estimators = 'auto'
    importance = "native"
    max_iter = 100
    random_state = None
    verbose = 0
    keep_weak = False
    regressor = XGBRegressor(random_state=42)
    leshy = Leshy(
        regressor,
        n_estimators=n_estimators,
        importance=importance,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
        keep_weak=keep_weak,
    )
    leshy.fit(xdf, yser)
    leshy_selection = leshy.selected_features_
    print_results(leshy_selection, 'BORUTA')


if __name__ == '__main__':
    main()
