import pandas as pd
import hisel


def main():
    # Minimial example of `hisel` usage with specification of parameters
    df = pd.read_csv('mydata.csv')
    xdf = df.iloc[:, :-1]
    yser = df.iloc[:, -1]
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
    print('\n\n##########################################################')
    print(
        f'The following features are relevant for the prediction of {yser.name}:')
    print(f'{results.selected_features}')


if __name__ == '__main__':
    main()
