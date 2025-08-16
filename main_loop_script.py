import tenpy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random
import pickle
import sys
#Used FermiHubbardChain from TeNPy to construct the required Hamiltonian (MFT for 3D lattice of 1D Hubbard chains with interchain hopping)
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.models.model import CouplingMPOModel
from tenpy.models.model import NearestNeighborModel
from tenpy.algorithms import dmrg
from tenpy.linalg import np_conserved as npc

class H_MF_Helper(CouplingMPOModel): 
    def init_sites(self, model_params):
        cons_N_temp = model_params.get('cons_N', None, str) #Particle number is NOT conserved
        cons_Sz = model_params.get('cons_Sz', 'Sz', str)
        site = SpinHalfFermionSite(cons_N=cons_N_temp, cons_Sz=cons_Sz) #S_z is conserved
        return site

    def init_terms(self, model_params):
        t = model_params.get('t', 1., 'real_or_array')
        U = model_params.get('U', 0, 'real_or_array')
        V = model_params.get('V', 0, 'real_or_array')
        L = model_params.get('L', 100, 'real_or_array')
        mu = model_params.get('mu', 0., 'real_or_array')
        r_range = model_params.get('r_range', 1, 'real_or_array')
        alpha = model_params.get('alpha', np.eye(L) * 0.1, 'array')
        beta = model_params.get('beta', np.zeros((2, L, L)), 'array')
        phi_ext = model_params.get('phi_ext', None, 'real')

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-mu, u, 'Ntot')
            self.add_onsite(U, u, 'NuNd')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if phi_ext is None:
                hop = -t
            else:
                hop = self.coupling_strength_add_ext_flux(-t, dx, [0, 2 * np.pi * phi_ext])
            self.add_coupling(hop, u1, 'Cdu', u2, 'Cu', dx, plus_hc=True)
            self.add_coupling(hop, u1, 'Cdd', u2, 'Cd', dx, plus_hc=True)
            self.add_coupling(V, u1, 'Ntot', u2, 'Ntot', dx)
        for i in range(L):
            for k in range(i+1, i+1+r_range): # Using open boundary conditions
                if k<L and alpha[i, k] != 0:
                    self.add_coupling_term(-1*alpha[i, k], i, k, 'Cu', 'Cd', plus_hc=True)
            self.add_onsite_term(-1*alpha[i, i], i, 'Cu Cd', plus_hc=True)
        for i in range(L):
            for r in range(1, r_range+1):
                for sigma in [0, 1]:
                    if i+r<L:
                        if beta[sigma, i, i+r] != 0:
                            if sigma == 0: #0 is down, 1 is up!
                                self.add_coupling_term(beta[0, i, i+r], i, i+r, 'Cdd', 'Cd', plus_hc=True)
                            if sigma == 1:
                                self.add_coupling_term(beta[1, i, i+r], i, i+r, 'Cdu', 'Cu', plus_hc=True)


class H_MF(H_MF_Helper):
    """The :class:`FermiHubbardModel` on a Chain, suitable for TEBD.
    See the :class:`FermiHubbardModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True

def find_density(psi):
    L = psi.L
    particle_number = 0
    for i in range(L):
        particle_number += psi.expectation_value('Ntot', i)
    return particle_number.item()/len(psi.sites)

def run_DMRG(M):
    eng = None
    
    #Construct MPS
    site = SpinHalfFermionSite()
    prod_state = ['up', 'empty', 'down', 'empty']*(M.lat.N_sites//4)
    
    #DMRG Params
    dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'mixer_params': {
        'amplitude': 1.e-5,
        'decay': 1.2,
        'disable_after': 30
    },
    'trunc_params': {
        'chi_max': chi_max,
        'svd_min': 1.e-10,
    },
    'chi_list': chi_list,
    'max_E_err': 1.e-6,
    'max_S_err': 1.e-6,
    'max_sweeps': 150,
    'verbose': 1,
    'combine': True
    }

    psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

    E, psi = eng.run()
    return find_density(psi), E, psi

def run_excited_DMRG(M):
    eng = None
    
    #Construct MPS
    site = SpinHalfFermionSite()
    prod_state = ['up', 'empty', 'down', 'empty']*(M.lat.N_sites//4)
    
    #DMRG Params
    dmrg_params = {
    'mixer': True,  # setting this to True helps to escape local minima
    'mixer_params': {
        'amplitude': 1.e-5,
        'decay': 1.2,
        'disable_after': 30
    },
    'trunc_params': {
        'chi_max': chi_max,
        'svd_min': 1.e-10,
    },
    'chi_list': chi_list,
    'max_E_err': 1.e-6,
    'max_S_err': 1.e-6,
    'max_sweeps': 150,
    'verbose': 1,
    'combine': True,
    }

    psi_0 = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
    psi_1 = psi_0.copy()
    states = dmrg_params['orthogonal_to'] = []
    E_0 = dmrg.run(psi_0, M, dmrg_params)
    dmrg_params['orthogonal_to'] = [psi_0]
    E_1 = dmrg.run(psi_1, M, dmrg_params, orthogonal_to=[psi_0])
    excitation_gap = E_1['E'] - E_0['E']
    return find_density(psi_1), excitation_gap, psi_1

def solve_Ham(mu, model_params, alpha, beta):
    model_params['mu'] = mu
    model_params['alpha'] = alpha
    model_params['beta'] = beta
    M = H_MF(model_params)
    density, E, psi = run_DMRG(M)
    return density, E, psi

#Same tolerances as specified by Phys. Rev. X paper
def close(alpha, alpha_measured, beta, beta_measured, r_range, threshold = 1e-4):
    eps = 1e-12 #divide by zero errors
    for r in range(0, r_range+1):
        if np.any(np.abs(np.diag(alpha_measured, r) - np.diag(alpha, r))/np.abs(np.diag(alpha, r)+eps) > threshold):
            return False
        if np.any(np.abs((np.diag(beta_measured[0, :, :], r)-np.diag(beta[0, :, :], r)))/np.abs(np.diag(beta[0, :, :], r)+eps) > threshold):
            return False
        if np.any(np.abs((np.diag(beta_measured[1, :, :], r)-np.diag(beta[1, :, :], r)))/np.abs(np.diag(beta[1, :, :], r)+eps) > threshold):
            return False
    return True


    
def find_mu_for_target_density(model_params, alpha, beta, mu_init, n_target, tol=1e-5, delta_mu=0.05 , max_iter=100):
    #returns mu, density, E, psi
    mu_0 = mu_init
    n_0, E, psi = solve_Ham(mu_0, model_params, alpha, beta)
    if (np.abs(n_0-n_target)/n_target <= tol):
        print('Same mu:{} and continuing'.format(mu_0))
        return mu_0, n_0, E, psi
    print('Not same mu searching again')
    mu_1 = mu_0 + delta_mu
    n_1, E, psi = solve_Ham(mu_1, model_params, alpha, beta)
    for iteration in range(max_iter):
        if abs(n_1 - n_target)/n_target <= tol:
            print('FOUND MU; mu:{} and n_1:{}'.format(mu_1, n_1))
            return mu_1, n_1, E, psi
        if (n_target >= n_1 and n_target >= n_0):
            mu_0, n_0 = mu_1, n_1
            mu_1 = mu_0 + delta_mu
            n_1, E, psi = solve_Ham(mu_1, model_params, alpha, beta)
            print('n_target > n_1 and n_target > n_0 triggered')
        elif (n_target <= n_1 and n_target <= n_0):               
            mu_0, n_0 = mu_1, n_1
            mu_1 = mu_0 - delta_mu
            n_1, E, psi = solve_Ham(mu_1, model_params, alpha, beta)
            print('n_target < n_1 and n_target < n_0 triggered')
        else:
            for j in range(max_iter):
                #Secant Method Step
                if abs(n_1 - n_0) > 1e-12:
                    print('Trying secant step...')

                    mu_new = mu_1 - (n_1 - n_target) * (mu_1 - mu_0) / (n_1 - n_0)

                    n_new, E, psi = solve_Ham(mu_new, model_params, alpha, beta)

                    if abs(n_new - n_target) / n_target <= tol:
                        print('FOUND MU (secant); mu_new:{}, n_new:{}'.format(mu_new, n_new))
                        return mu_new, n_new, E, psi
                    
                    mu_0, n_0 = mu_1, n_1
                    mu_1, n_1 = mu_new, n_new
                
                #Backup Bisection Method Step
                else:
                    print('Secant unstable, falling back to bisection...')
                    mu_mid = (mu_0 + mu_1) / 2
                    n_mid, E, psi = solve_Ham(mu_mid, model_params, alpha, beta)

                    if abs(n_mid - n_target) / n_target <= tol:
                        print('FOUND MU (bisection); mu_mid:{}, n_mid:{}'.format(mu_mid, n_mid))
                        return mu_mid, n_mid, E, psi
                    
                    if n_1 > n_0:  
                        if n_target > n_mid:
                            mu_0, n_0 = mu_mid, n_mid
                        else:
                            mu_1, n_1 = mu_mid, n_mid
                    else:  
                        if n_target > n_mid:
                            mu_1, n_1 = mu_mid, n_mid
                        else:
                            mu_0, n_0 = mu_mid, n_mid
            
            raise ValueError("Target density not achieved within the maximum number of iterations in the refinement loop")
        print('n_0:{}; n_1:{}; mu_1:{}'.format(n_0, n_1, mu_1))
    print('mu_0:{}, mu_1:{}, n_0:{}; n_1:{}, n_target:{}'.format(mu_0, mu_1, n_0, n_1, n_target))
    raise ValueError("Target density not achieved within the maximum number of iterations")

def calculate_alpha_beta_measured(psi, model_params, r_range, E_p): 
    psi_alpha = psi.copy()
    psi_beta_down = psi.copy()
    psi_beta_up = psi.copy()
    L = model_params['L']
    t_p = model_params['t_p']

    alpha_measured = np.zeros((L, L))
    beta_measured = np.zeros((2, L, L))
                        
    alpha_corr = 2*z_c*t_p**2/E_p*psi_alpha.correlation_function("Cu", "Cd")
    beta_down_corr = 2*z_c*t_p**2/E_p*psi_beta_down.correlation_function("Cdd", "Cd")
    beta_up_corr = 2*z_c*t_p**2/E_p*psi_beta_up.correlation_function("Cdu", "Cu")

    i, j = np.indices((L, L))
    mask = np.abs(i - j) <= r_range
    
    alpha_measured = np.where(mask, alpha_corr, 0.)
    beta_measured[0] = np.where(mask, beta_down_corr, 0.)
    beta_measured[1] = np.where(mask, beta_up_corr, 0.)
    
    return alpha_measured, beta_measured

    
def order_parameter(psi, r):
    temp_corr_arr = np.diagonal(psi.correlation_function("Cu", "Cd"), r)
    return 1/(psi.L-20)*np.sum(temp_corr_arr[10:psi.L-10]) # why -10? It's enough to ignore open/finite boundary effects


#Main loop
def main_loop(model_params, n_target, E_p, max_iter=150):
    alpha = model_params['alpha'] 
    beta = model_params['beta']
    mu = model_params['mu']
    r_range = model_params['r_range']
    
    #Loop which first checks density and then checks alpha and beta convergence
    for iteration in range(max_iter):
        #Check if density is correct and then adjusts mu otherwise
        mu, n_measured, E, psi = find_mu_for_target_density(model_params, alpha, beta, mu, n_target)
        if abs(n_measured - n_target) < 1e-2:
            print(f"Target density achieved with mu: {mu}, density: {n_measured}")
            model_params['mu'] = mu
            model_params['iter'] += 1
        #Checks alpha and beta convergence
        alpha_measured, beta_measured = calculate_alpha_beta_measured(psi, model_params, r_range, E_p)
        print('CHECKING IF INPUT {A, B} AGREES WITH OUTPUT {A, B}_measured')
        if close(alpha, alpha_measured, beta, beta_measured, r_range):
            print("CHECK VALID FOR {A, B} AGREEMENT")
            print(f"Converged alpha and beta. mu: {mu}, density: {n_measured}")
            print('EXITING LOOP')
            return alpha, beta, mu, n_measured, psi, E
        print('CHECK NOT VALID FOR {A, B} AGREEMENT')
        # Update alpha and beta based on measured values if not converged
        alpha = alpha_measured
        beta = beta_measured
        model_params['alpha'] = alpha
        model_params['beta'] = beta
    print("Failed to converge alpha and beta within the maximum number of iterations")
    return alpha, beta, mu, n_measured, psi, E


#Function to run the main loop
def run_loop(L, t, U, t_p, mu_init, n_target, r_range, E_p):
    #Initial alpha, beta values
    alpha = np.eye(L, L)*0.5
    beta = np.zeros((2, L, L))
    
    model_params = {
                'L': L,
                'bc_MPS': 'finite',  # Boundary condition (infinite lattice)
                'cons_N': None,     # Conservation of particle number 
                'cons_Sz': 'Sz',
                't': t,            # Hopping amplitude
                't_p': t_p, 
                'U': U,            # On-site interaction
                'mu': mu_init,
                'verbose': 0,        # Verbosity level (0 for minimal output)
                'conserve': None,
                'r_range': r_range,
                'alpha': alpha,
                'beta': beta,
                'iter': 0
    }
    
    #Go through loop:
    alpha, beta, mu, n_measured, psi, E = main_loop(model_params, n_target, E_p)
    model_params['mu'] = mu
    model_params['alpha'] = alpha
    model_params['beta'] = beta
    M = H_MF(model_params)
    
    _, gap, _ = run_excited_DMRG(M)
    order_param = np.abs(order_parameter(psi, 0))
    temp_dict = {'U': U, 't_p': t_p, 'alpha': alpha, 'beta': beta, 'mu': mu, 'psi': psi, 'order_param': order_param, 'gap': gap, 'E': E}
    return temp_dict

#Run_DMRG given a specific product state
def run_DMRG_prod(model_params, prod_state):
    """Function to run DMRG for given model M, particle number N, and spin S."""
    M = FermiHubbardChain(model_params)
    # DMRG Parameters
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'mixer_params': {
            'amplitude': 1.e-5,
            'decay': 1.2,
            'disable_after': 30
        },
        'trunc_params': {
            'svd_min': 1.e-10,
        },
        #'chi_list': {0: 9, 10: 49, 20: chi_max},
        'chi_list': chi_list,
        'max_E_err': 1.e-6,
        'max_S_err': 1.e-6,
        'max_sweeps': 150,
        'verbose': 1,
    }

    psi = MPS.from_product_state(M.lat.mps_sites(), prod_state, bc=M.lat.bc_MPS)
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

    E, psi = eng.run()
    return E

#Pair binding energy function specifically for n=0.5
def calculate_pair_binding_energy(L, U):
    model_params = {
            'L': L,
            'bc_MPS': 'finite',  
            'cons_N': 'N',     #Conserve N here
            'cons_Sz': 'Sz',
            't': 1,            
            'U': U,            
            'mu': 0,
            'verbose': 0,        
            'conserve': 'N'
    }
    N = L
    prod_state_0 = ['empty'] * L
    for i in range(N):
        if i%4 == 0:
            prod_state_0[i] = 'up'
        elif i%2 == 0:
            prod_state_0[i] = 'down'
    # Run DMRG for N particles
    prod_state = prod_state_0
    model_params['N'] = N
    model_params['S'] = 0
    E_N_0 = run_DMRG_prod(model_params, prod_state)

    # Run DMRG for N+1 particles with S=1/2
    prod_state = prod_state_0
    prod_state[1] = 'up'
    model_params['N'] = N + 1
    model_params['S'] = 0.5
    E_N_plus_1_half = run_DMRG_prod(model_params, prod_state)

    # Run DMRG for N+2 particles with S=0
    prod_state = prod_state_0
    prod_state[1] = 'up'
    prod_state[3] = 'down'
    model_params['N'] = N + 2
    model_params['S'] = 0
    E_N_plus_2 = run_DMRG_prod(model_params, prod_state)

    # Compute pairing energy
    pairing_energy = 2 * E_N_plus_1_half - E_N_0 - E_N_plus_2 
    return pairing_energy


def main():
    if len(sys.argv) != 7:
        print("Usage: python main_loop_script.py <L> <U> <t_p> <chi_max> <E_p>")
        sys.exit(1)

    # Arguments
    # 1: L
    # 2: U
    # 3: t_p
    # 4: chi_max
    # 5: E_p
    # 6: mu_init

    global L, U, t, z_c, chi_max, chi_list, t_p, mu_init, n_target, r_range, E_p

    L = int(sys.argv[1])
    U = float(sys.argv[2])
    t_p = float(sys.argv[3])
    chi_max = int(sys.argv[4])
    chi_list = {0: 20, 10: 50, 30: 100, 40: chi_max}
    if sys.argv[5] is not (None or 0):
        E_p = float(sys.argv[5])
        print(f"Using provided E_p = {E_p}")
    else:
        print("Calculating pair binding energy E_p...")
        E_p = np.abs(calculate_pair_binding_energy(L, U, chi_max, chi_list))
        print(f"Calculated E_p = {E_p}")
    
    mu_init = float(sys.argv[6])

    print(f"Running with t_p={t_p}, U={U}, L={L}, chi_max={chi_max}, E_p={E_p}, mu_init={mu_init}")

    
    t = 1.0
    n_target = 0.5
    r_range = 4
    z_c = 4

    

    result_dict = run_loop(L, t, U, t_p, mu_init, n_target, r_range, E_p)

    outfile_name = f"results_U_{U}_t_p{t_p}.pkl"
    print(f"Calculation finished. Saving results to {outfile_name}")
    with open(outfile_name, 'wb') as f:
        pickle.dump(result_dict, f)


if __name__ == '__main__':
    main()