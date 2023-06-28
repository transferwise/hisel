import unittest
import numpy as np
import pandas as pd

from hisel import categorical


class CategoricalSearchTest(unittest.TestCase):
    def test_discretise(self):
        y = np.random.uniform(size=(1000, 1))
        flat = categorical._discretise(y[:, 0])
        keepdim = categorical._discretise(y)
        self.assertTrue(
            np.all(keepdim[:, 0] == flat)
        )

    def generate_data(self):
        d = np.random.randint(low=10, high=15)
        n = np.random.randint(low=4000, high=5000)
        h = np.random.randint(low=5, high=10)
        n_rel = 3
        random_state = np.random.randint(low=0, high=100)

        x = np.random.randint(low=0, high=h, size=(n, d))
        a = np.random.permutation(np.concatenate((
            np.random.randint(low=-9, high=9, size=(n_rel, )),
            np.zeros(shape=(d - n_rel,), dtype=int)
        )))
        expected, = np.where(np.abs(a) > 0)
        t = a.reshape(1, 1, d)
        y = (t @ np.expand_dims(x, axis=2))[:, 0, 0]
        return x, y, expected

    def test_search_and_select(self):
        random_state = np.random.randint(low=1, high=50)
        x, y, expected = self.generate_data()
        selected = self._test_search(
            x, y, expected, random_state)
        selected, selection = self._test_select(
            x, y, expected, selected, random_state)

    def _assert_expected(self, expected, selected):
        n_rel = len(expected)
        self.assertTrue(
            len(selected) >= len(expected) - 1,
            'Too few selected features!'
        )
        symdiff = set(selected).symmetric_difference(set(expected))
        threshold = 1 + int(n_rel < 3) + n_rel // 3
        self.assertTrue(
            len(symdiff) < threshold,
            'Too large difference between selected and expected'
        )

    def _test_search(self, x, y, expected, random_state=None):
        n, d = x.shape
        n_rel = len(expected)
        num_permutations = 15
        im_ratio = .01
        max_iter = 1

        selected = categorical.search(
            x, y,
            num_permutations=num_permutations,
            im_ratio=im_ratio,
            max_iter=max_iter,
            parallel=False,
            random_state=random_state,
        )
        print(f'expected:\n{sorted(expected)}')
        print(f'selected:\n{sorted(selected)}')
        self._assert_expected(expected, selected)
        return selected

    def _test_select(self, x, y, expected, selected=None, random_state=None):
        n, d = x.shape
        n_rel = len(expected)
        num_permutations = 20
        im_ratio = .01
        max_iter = 1

        xdf = pd.DataFrame(x, columns=[f'f{i}' for i in range(d)])
        ydf = pd.Series(y)
        selection = categorical.select(
            xdf,
            ydf,
            num_permutations=num_permutations,
            im_ratio=im_ratio,
            max_iter=max_iter,
            parallel=False,
            random_state=random_state,
        )
        if selected is None:
            selected = categorical.search(
                x, y,
                num_permutations=num_permutations,
                im_ratio=im_ratio,
                max_iter=max_iter,
                parallel=True,
                random_state=random_state,
            )
        print(f'expected:\n{sorted(expected)}')
        print(f'selected:\n{sorted(selected)}')
        self._assert_expected(expected, selected)
        recon = [int(f.replace('f', '')) for f in selection.features]
        self.assertEqual(
            set(selection.indexes),
            set(selected)
        )
        self.assertEqual(
            set(recon),
            set(selected)
        )
        return selected, selection


if __name__ == '__main__':
    unittest.main()
