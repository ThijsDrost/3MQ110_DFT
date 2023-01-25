import numpy as np
from scipy.sparse import spdiags, linalg

dataloc = r'D:\Uni\Master\Mulitscale_methods\DFT\Code\Data'


def data_loc(loc: str):
    return rf'{dataloc}/{loc}'


n_vals = np.logspace(2, 6, 51, dtype=int)
h_vals = np.logspace(-5, 0, 51)
x_0s = np.logspace(-10, -100, 10)
for x_0 in x_0s:
    print(x_0)
    uno = np.zeros((len(n_vals), len(h_vals)))
    dos = np.zeros((len(n_vals), len(h_vals)))
    tres = np.zeros((len(n_vals), len(h_vals)))
    for i, n_val in enumerate(n_vals):
        for j, h_val in enumerate(h_vals):
            x_val = np.linspace(x_0, x_0+n_val*h_val, n_val)
            diag = np.full(n_val, 2/(h_val**2)) - 2/x_val
            diag_off = np.full(n_val, -1/(h_val**2))
            matrix = spdiags([diag_off, diag, diag_off], [-1, 0, 1], n_val, n_val)
            value, vector = linalg.eigs(matrix, 3, sigma=-1)
            uno[i, j] = np.real(value[0])
            dos[i, j] = np.real(value[1])
            tres[i, j] = np.real(value[2])
            print(f'\ri: {i}, j: {j}', end='')
    print('')
    np.savez(data_loc(rf'1_acc_x0_{x_0:.0e}'), uno=uno, dos=dos, tres=tres, n=n_vals, h=h_vals)