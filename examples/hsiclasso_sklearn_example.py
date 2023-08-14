import numpy as np
from sklearn.feature_selection import mutual_info_regression, f_regression
import hisel


# This example is inspired by
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py


def main():
    n = 5000
    num_discr_features = 10
    num_cont_features = 10
    relevant_discr_feature = 3
    relevant_cont_feature = 12
    xdiscr = np.random.randint(low=0, high=2, size=(n, num_discr_features))
    xcont = np.random.uniform(size=(n, num_cont_features))
    x = np.concatenate((xdiscr, xcont), axis=1)
    def fun(phase, t): return np.sin(phase * np.pi * t)
    noise = np.random.normal(loc=0, scale=.05, size=(n, 1))
    # The discrete variable modulates the phase,
    # The continuous variable gives the evaluation point
    y = noise + fun(x[:, [relevant_discr_feature]],
                    x[:, [relevant_cont_feature]])

    expected_features = [relevant_discr_feature, relevant_cont_feature]

    selector = hisel.select.HSICSelector(
        x, y,
        xfeattype=hisel.select.FeatureType.BOTH,
        yfeattype=hisel.select.FeatureType.CONT,
        catcont_split=10,
    )

    selected_features = selector.select(
        number_of_features=2,
        number_of_epochs=3,
        return_index=True
    )

    mi = mutual_info_regression(x, np.squeeze(y))
    mi /= np.max(mi)
    mi_selection = np.argsort(mi)[-2:]

    f_test, _ = f_regression(x, np.squeeze(y))
    f_test /= np.max(f_test)
    f_test_selection = np.argsort(f_test)[-2:]

    print('\n\n##########################################################')
    print(f'Expected features: {expected_features}')  # [3, 12]
    print(f'HISC Lasso selection: {selected_features}')  # Should print [3, 12]
    print(f'Mutual info selection: {mi_selection}')  # Should print [3, 12]
    print(f'F-Test selection: {f_test_selection}')  # Expected to be wrong


if __name__ == '__main__':
    main()
