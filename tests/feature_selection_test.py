import unittest
import numpy as np
import pandas as pd

from hisel import feature_selection


class FeatSelTest(unittest.TestCase):
    def test(self):
        d_cont = 25
        d_discr = 15
        n = 5000
        n_rel_cont = 5
        n_rel_discr = 2
        xcont = np.random.uniform(size=(n, d_cont))
        xdiscr = np.random.randint(low=0, high=10, size=(n, d_discr))
        acont = np.random.permutation(np.concatenate((
            np.random.uniform(low=-1, high=1, size=(n_rel_cont,)),
            np.zeros(shape=(d_cont - n_rel_cont,))
        )))
        expected_cont, = np.where(np.abs(acont) > 0)
        tcont = acont.reshape(1, 1, d_cont)
        ycont = (tcont @ np.expand_dims(xcont, axis=2))[:, 0, 0]
        adiscr = np.random.permutation(np.concatenate((
            np.random.uniform(low=-1, high=1, size=(n_rel_discr,)),
            np.zeros(shape=(d_discr - n_rel_discr,))
        )))
        expected_discr, = np.where(np.abs(adiscr) > 0)
        tdiscr = adiscr.reshape(1, 1, d_discr)
        _ydiscr = (tdiscr @ np.expand_dims(xdiscr, axis=2))[:, 0, 0]
        ydiscr = (-np.ones(n, dtype=int) +
                  2 * np.array(_ydiscr > np.quantile(_ydiscr, .5), dtype=int)
                  )
        y = ydiscr * ycont
        xdfcont = pd.DataFrame(xcont, columns=[f'c{i}' for i in range(d_cont)])
        expected_cont_names = sorted(
            xdfcont.iloc[:, expected_cont].columns.tolist())
        xdfdiscr = pd.DataFrame(
            xdiscr, columns=[f'd{i}' for i in range(d_discr)])
        expected_discr_names = sorted(
            xdfdiscr.iloc[:,        expected_discr].columns.tolist())
        xdf = pd.concat([xdfcont, xdfdiscr], axis=1)
        ydf = pd.Series(y, name='y')
        results = feature_selection.select_features(xdf, ydf)
        selected_cont = results.continuous_lasso_selection.hsic_selection
        selected_discr = results.categorical_search_selection.features

        print('\n\nFeature Selection Test')
        print(f'Expected cont:\n{expected_cont_names}')
        print(f'Selected continuous features:\n{sorted(selected_cont)}')
        print(f'Expected discr:\n{expected_discr_names}')
        print(f'Selected discrete features:\n{sorted(selected_discr)}')
        self.assertTrue(
            len(selected_cont) > len(expected_cont) - 2
        )
        self.assertTrue(
            len(expected_discr) > len(expected_discr) - 2
        )


if __name__ == '__main__':
    unittest.main()
