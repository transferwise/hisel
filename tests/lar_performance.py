import datetime
import numpy as np
from hisel import lar
use_pyhsiclasso = True
try:
    from pyHSICLasso import nlars
except (ModuleNotFoundError, ImportError):
    use_pyhsiclasso = False

def speed_test(
        n: int,
        d: int,
        a: int,
        number_of_experiments:int = 1,
        ):
    x = np.random.uniform(size=(n, d))
    beta = np.random.permutation(np.vstack(
        [
            np.random.uniform(low=0., high=1., size=(a, 1)),
            np.zeros((d-a, 1), dtype=np.float32)
        ]
    ))
    y = x @ beta + np.random.uniform(low=-1e-2, high=1e-2, size=(n, 1))
    hisel_runtimes = []
    for _ in range(number_of_experiments):
        t0 = datetime.datetime.now()
        feats = lar.solve(x, y, a)
        t1 = datetime.datetime.now()
        hisel_runtime = t1 - t0
        hisel_runtimes.append(
                hisel_runtime.seconds + 1e-6 * hisel_runtime.microseconds
                )
    hisel_mean = np.mean(hisel_runtimes)
    hisel_std = np.std(hisel_runtimes)
    print(
            f'hisel run time: {hisel_mean:.4f} +/- {hisel_std:.4f}  seconds\n')

    
    if use_pyhsiclasso:
        pyhsic_runtimes = []
        for _ in range(number_of_experiments):
            t0 = datetime.datetime.now()
            feats = nlars.nlars(x, x.T @ y, a, 3)
            t1 = datetime.datetime.now()
            pyhsic_runtime = t1 - t0
            pyhsic_runtimes.append(
                    pyhsic_runtime.seconds + 1e-6 * pyhsic_runtime.microseconds
                    )
        pyhsic_mean = np.mean(pyhsic_runtimes)
        pyhsic_std = np.std(pyhsic_runtimes)
        print(
                f'pyhsic run time: {pyhsic_mean:.4f} +/- {pyhsic_std:.4f}  seconds\n')

if __name__ == '__main__':
    n  = 100000
    d = 500
    a = 100
    number_of_experiments = 5
    # With these parameters, on my laptop, I get:
    # hisel run time: 14.1813 +/- 0.0975  seconds
    # pyhsic run time: 15.9536 +/- 0.2564  seconds
    speed_test(n, d, a, number_of_experiments)
    

