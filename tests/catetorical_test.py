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

    def test_search(self):
        d = np.random.randint(low=10, high=12)
        n = np.random.randint(low=4000, high=5000)
        h = np.random.randint(low=5, high=10)
        n_rel = d // 3
        num_permutations = 1 * d
        im_ratio = .05
        max_iter = 1
        random_state = np.random.randint(low=0, high=100)

        x = np.random.randint(low=0, high=h, size=(n, d))
        a = np.random.permutation(np.concatenate((
            np.random.randint(low=-9, high=9, size=(n_rel, )),
            np.zeros(shape=(d - n_rel,), dtype=int)
        )))
        expected, = np.where(np.abs(a) > 0)
        t = a.reshape(1, 1, d)
        y = (t @ np.expand_dims(x, axis=2))[:, 0, 0]
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
        self.assertTrue(
            len(selected) >= len(expected) - 1
        )
        self.assertTrue(
            len(set(selected).symmetric_difference(
                set(expected))) < 1 + n_rel // 3
        )
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
        print(f'selection:\n{sorted(selection.features)}')
        recon = [int(f.replace('f', '')) for f in selection.features]
        self.assertEqual(
            set(selection.indexes),
            set(selected)
        )
        self.assertEqual(
            set(recon),
            set(selected)
        )


if __name__ == '__main__':
    unittest.main()
