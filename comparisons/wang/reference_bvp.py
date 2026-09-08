import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

F = 96485.33212
R = 8.314462618
T = 298.15
alpha = 0.5
W = 0.005
kappa = 5.9514
sigma = 103.1891
a = 16400.0
j0 = 2.7657
j_app = -400.0

def fun(x, y):
    eta, i = y
    deta = -(j_app - i) / sigma + i / kappa
    di = a * j0 * (np.exp(alpha * F * eta / (R*T))- np.exp(-(1 - alpha) * F * eta / (R*T)))
    #di   = 2.0 * a * j0 * np.sinh( (F*eta) / (2.0*R*T) )
    return np.vstack([deta, di])

def bc(ya, yb):
    etaa, ia =  ya
    etab, ib =  yb

    return np.array([ia - 0.0 , ib - j_app])

x = np.linspace(0, W, 60)
eta_guess = np.linspace(-0.05 if j_app < 0 else 0.05 , 0.0, x.size)
i_guess = j_app  * (x / W)
y_guess = np.vstack([eta_guess, i_guess])

sol = solve_bvp(fun, bc, x, y_guess, tol=1e-10 , max_nodes=100000000.0)

print('success:', sol.success, sol.message)
print('number of used mesh:', len(sol.x))

xx = np.linspace( 0 , W , len(sol.x) )
eta, i = sol.sol(xx)
plt.figure(figsize=(6, 5))
plt.plot(xx, 1000.0 * eta)
plt.ylabel('eta (mV)')
plt.grid(False)
plt.tight_layout()
plt.show()


from pathlib import Path

EXPORT_DIR = Path(__file__).resolve().parent / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

eta_csv = EXPORT_DIR / 'ode_eta_constant_concentration.csv'
np.savetxt(eta_csv, np.column_stack([xx, eta]),
    delimiter=',', header='x_m,eta_V', comments='', fmt='%.10e')

print(f'[export] wrote {eta_csv} with {xx.size} rows.')
