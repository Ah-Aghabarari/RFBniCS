
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / 'exports'

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

############
F = 96485.33212
R = 8.314462618
T = 298.15
alpha = 0.5
z = 1.0
W = 0.0025
kappa = 5.9514
sigma = 3402.1
a = 23000.0
j_app = -0.5
E_formal = -0.36
k0 = 2.5e-07
epsilon = 0.94
cV2_b = 300.0
cV3_b = 300.0

D_V2 = 8.1e-12 * epsilon ** 1.5
D_V3 = 1.65e-11 * epsilon ** 1.5
sigma = sigma * (1 - epsilon) ** 1.5
##########

#  smoothing for kinetics
eps_c = 1e-14

def smooth_pos(c):
    return 0.5 * (c + np.sqrt(c * c + eps_c * eps_c))

def dsmooth_pos(c):
    return 0.5 * (1.0 + c / np.sqrt(c * c + eps_c * eps_c))

# scales

V_scale = R*T/(z*F)
I_scale = abs(j_app)
C_scale = 300.0
j_hat = j_app / I_scale

def fun_scaled(xi, Y):
    
    eta_hat, i_hat, c2_hat, q2, c3_hat, q3 = Y
    eta = V_scale*eta_hat
    i = I_scale*i_hat
    c2 = C_scale*c2_hat
    c3 = C_scale*c3_hat
    p2 = C_scale/W*q2
    p3 = C_scale/W*q3
    c2_eff = smooth_pos(c2)
    c3_eff = smooth_pos(c3)
    dc2_eff_dx = dsmooth_pos(c2) * p2
    dc3_eff_dx = dsmooth_pos(c3) * p3

    dEeq_dx = R * T / F * (dc3_eff_dx / (c3_eff + eps_c) - dc2_eff_dx / (c2_eff + eps_c))
    deta_dx = -(j_app - i) / sigma + i / kappa - dEeq_dx

    f = a * z * F * k0 * c2_eff ** alpha * c3_eff ** (1.0 - alpha) * (
        np.exp(alpha * z * F * eta / (R*T))
        - np.exp(-(1.0 - alpha) * z * F * eta / (R*T)))

    deta_hat_dxi= W / V_scale * deta_dx
    di_hat_dxi= W / I_scale * f
    dc2_hat_dxi= q2
    dq2_dxi= W ** 2 * f / (C_scale * z * F * D_V2)
    dc3_hat_dxi= q3
    dq3_dxi= -W ** 2 * f / (C_scale * z * F * D_V3)

    return np.vstack([deta_hat_dxi, di_hat_dxi, dc2_hat_dxi , dq2_dxi, dc3_hat_dxi , dq3_dxi])

# bc
def bc_scaled(Ya, Yb):

    return np.array([Ya[1] , Yb[1] - j_hat , Ya[3], Ya[5], Yb[2] - cV2_b / C_scale, Yb[4] - cV3_b / C_scale])




xi = np.linspace(0.0, 1.0, 200)
# Initial guess
eta_guess = np.linspace(-0.05 if j_app < 0 else 0.05, 0.0, xi.size)
eta_hat_guess = eta_guess / V_scale
i_hat_guess = j_hat * xi
c2_hat_guess = np.full_like(xi, cV2_b / C_scale)
c3_hat_guess = np.full_like(xi, cV3_b / C_scale)
q2_guess = np.zeros_like(xi)
q3_guess = np.zeros_like(xi)
Y_guess = np.vstack( [eta_hat_guess, i_hat_guess, c2_hat_guess, q2_guess, c3_hat_guess, q3_guess] )

sol = solve_bvp(fun_scaled, bc_scaled, xi, Y_guess,
    tol=1e-10, bc_tol=1e-10, max_nodes=100000, verbose=1)

print('success:', sol.success)
print('message:', sol.message)
print('number of used mesh:', len(sol.x))



xi_ref = np.linspace(0.0, 1.0, 20001)
Y_ref = sol.sol(xi_ref)
x = W * xi_ref
eta = V_scale * Y_ref[0]
i = I_scale * Y_ref[1]
cv2 = C_scale * Y_ref[2]
cv3 = C_scale * Y_ref[4]

#######
plt.figure()
plt.plot(1e6 * x, eta, lw=2)
plt.xlabel('x [micro_m]')
plt.ylabel('eta [V]')
plt.tight_layout()
plt.show()
plt.figure()
plt.plot(1e6 * x, cv2, lw=2, label='c_V2')
plt.plot(1e6 * x, cv3, lw=2, label='c_V3')
plt.xlabel('x [micro_m]')
plt.ylabel('c [mol m-3}]')
plt.legend()
plt.tight_layout()
plt.show()


eta_csv = EXPORT_DIR / 'ode_eta.csv'
np.savetxt(eta_csv, np.column_stack([x, eta]), delimiter=',', header='x_m,eta_V', comments='', fmt='%.16e')


cv2_csv = EXPORT_DIR / 'ode_cv2.csv'
np.savetxt(cv2_csv, np.column_stack([x, cv2]), delimiter=',', header='x_m,cV2_molm3', comments='', fmt='%.16e')



cv3_csv = EXPORT_DIR / 'ode_cv3.csv'
np.savetxt(cv3_csv, np.column_stack([x, cv3]), delimiter=',', header='x_m,cV3_molm3', comments='', fmt='%.16e')
