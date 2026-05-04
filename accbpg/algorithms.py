# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import numpy as np
import time
import warnings
from dataclasses import dataclass


class CountedFunction:
    """Wrap a smooth objective and count calls to f, gradient, and func_grad.

    A call to func_grad(..., flag=2) counts as one oracle call, not two,
    since it is a joint function/gradient query.
    """
    def __init__(self, f):
        self._f = f
        self.oracle_calls = 0

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __call__(self, x):
        self.oracle_calls += 1
        return self._f(x)

    def gradient(self, x):
        self.oracle_calls += 1
        return self._f.gradient(x)

    def func_grad(self, x, flag=2):
        self.oracle_calls += 1
        return self._f.func_grad(x, flag=flag)

    def reset_oracle_calls(self):
        self.oracle_calls = 0


def positive_tk_eta(eta_k, mu, M):
    """Positive root in eta-form; returns t=1 for the limiting eta_k=0 initialization."""
    if M <= 0:
        raise ValueError("M must be positive.")
    if mu < 0:
        raise ValueError("mu must be nonnegative.")
    if eta_k == 0:
        return 1.0
    if eta_k < 0:
        raise ValueError("eta_k must be nonnegative.")

    a = 1.0 + mu * eta_k
    disc = 1.0 + 4.0 * M * eta_k * a
    t = 2.0 * a / (1.0 + math.sqrt(disc))
    return min(1.0, max(0.0, t))


def compute_tau(t_k, mu, current_M, current_L):
    """Return tau = (t - mu / sqrt(M L)) / (1 - mu / sqrt(M L))."""
    if mu == 0.0:
        return t_k
    if current_M <= 0.0:
        raise ValueError("current_M must be positive.")
    if current_L <= 0.0:
        raise ValueError("current_L must be positive.")
    scale = math.sqrt(current_M * current_L)
    delta = mu / scale
    denom = 1.0 - delta
    if denom <= 0.0:
        raise ValueError("Invalid tau: need sqrt(current_M * current_L) > mu.")
    return (t_k - delta) / denom


def backtracking_gradient(y, fy, gy, current_L, f_eval, div_prox_map, divergence,
                          max_backtracks=50):
    """
    Backtracking for the mirror/prox-gradient step.

    Tracks and returns the best fplus seen during backtracking.
    """
    if current_L <= 0:
        raise ValueError("current_L must be positive.")
    if max_backtracks <= 0:
        raise ValueError("max_backtracks must be positive.")

    xplus = None
    fplus = None
    best_xplus = None
    best_fplus = None

    for _ in range(max_backtracks):
        xtrial = div_prox_map(y, gy, current_L)
        step = xtrial - y
        dxy = divergence(xtrial, y)
        ftrial = f_eval(xtrial)

        if best_fplus is None or ftrial < best_fplus:
            best_xplus = xtrial
            best_fplus = ftrial

        if (ftrial - fy) <= np.dot(gy, step) + current_L * dxy:
            return best_xplus, best_fplus, current_L

        xplus = xtrial
        fplus = ftrial
        current_L *= 2.0

    warnings.warn(
        "backtracking_gradient failed to find a valid current_L; "
        "returning the best trial seen.",
        RuntimeWarning,
        stacklevel=2,
    )
    return best_xplus, best_fplus, current_L



@dataclass
class AbraOracles:
    func_grad: object
    f_eval: object
    grad_h: object
    extra_Psi: object
    div_prox_map: object
    divergence: object

    @classmethod
    def from_problem(cls, f, h):
        return cls(
            func_grad=f.func_grad,
            f_eval=f if callable(f) else (lambda x: f.func_grad(x)[0]),
            grad_h=h.gradient,
            extra_Psi=h.extra_Psi,
            div_prox_map=h.div_prox_map,
            divergence=h.divergence,
        )


@dataclass
class AbraState:
    x: np.ndarray
    z: np.ndarray
    lambdak: np.ndarray
    eta: float
    phi: float
    psi_z: float
    x_anchor: np.ndarray
    dx_anchor: np.ndarray
    L_cur: float
    M_cur: float


@dataclass
class AbraStepResult:
    y: np.ndarray
    gy: np.ndarray
    dy: np.ndarray
    xplus: np.ndarray
    zplus: np.ndarray
    lambda_plus: np.ndarray
    philow: float
    phi_plus: float
    dzz: float
    L_cur: float


@dataclass
class AbraHistories:
    F: np.ndarray
    tk: np.ndarray
    tau: np.ndarray
    eta: np.ndarray
    M: np.ndarray
    alpha: np.ndarray
    L: np.ndarray
    rho_eff: np.ndarray
    T: np.ndarray  # oracle calls

    @classmethod
    def allocate(cls, maxitrs):
        return cls(
            F=np.zeros(maxitrs),
            tk=np.zeros(maxitrs),
            tau=np.full(maxitrs, np.nan),
            eta=np.full(maxitrs, np.nan),
            M=np.full(maxitrs, np.nan),
            alpha=np.full(maxitrs, np.nan),
            L=np.full(maxitrs, np.nan),
            rho_eff=np.full(maxitrs, np.nan),
            T=np.zeros(maxitrs),
        )

    def record(self, k, state, t_k, tau_k, mu, elapsed):
        self.F[k] = state.phi
        self.tk[k] = t_k
        self.tau[k] = tau_k
        self.eta[k] = state.eta
        self.M[k] = state.M_cur
        self.alpha[k] = mu if state.eta == 0 else mu + 1.0 / state.eta
        self.L[k] = state.L_cur
        if state.L_cur > 0:
            self.rho_eff[k] = math.sqrt(state.M_cur / state.L_cur) if mu == 0 else (state.M_cur / state.L_cur) ** 0.25
        self.T[k] = elapsed

    def result(self, n):
        return (
            self.F[:n],
            self.tk[:n],
            self.eta[:n],
            self.M[:n],
            self.alpha[:n],
            self.L[:n],
            self.T[:n],
        )

    def diagnostics(self, n):
        return {
            "tau": self.tau[:n],
            "rho_eff": self.rho_eff[:n],
            "M": self.M[:n],
            "alpha": self.alpha[:n],
            "L": self.L[:n],
        }


def acc_init_state(x_start, mu, current_L, current_M, oracles):
    """Initialize/reset one ABRA_GD phase from a fresh anchor."""
    x_anchor = np.copy(x_start)
    dx_anchor = oracles.grad_h(x_anchor)
    phi_x = oracles.f_eval(x_anchor) + oracles.extra_Psi(x_anchor)

    return AbraState(
        x=np.copy(x_anchor),
        z=np.copy(x_anchor),
        lambdak=np.zeros_like(x_anchor),
        eta=0,
        phi=phi_x,
        psi_z=oracles.extra_Psi(x_anchor),
        x_anchor=x_anchor,
        dx_anchor=dx_anchor,
        L_cur=current_L,
        M_cur=current_M,
    )


def BregPDStep(state, eta_plus_inv, t_k, tau_k, mu, oracles,
               max_backtracks=50, local_z=False):
    """
    One generic primal-dual Bregman step.

    If local_z is False, use the original anchor-based prox:
        z_{k+1} = argmin_x <lambda_{k+1}, x>
                  + (mu + eta_{k+1}^{-1}) D_h(x, x_anchor) + Psi(x).

    If local_z is True, use the local form centered at z_k.
    This branch is appropriate only when the local prox formula is valid for the chosen h/Psi.
    """
    y = state.x + tau_k * (state.z - state.x)
    fy, gy = oracles.func_grad(y)
    dy = oracles.grad_h(y)

    if eta_plus_inv <= 0.0:
        raise ValueError("eta_plus_inv <= 0 in BregPDStep. Increase current_M.")
    curvature_new = mu + eta_plus_inv

    dual_target = gy - mu * (dy - state.dx_anchor)
    lambda_plus = state.lambdak + t_k * (dual_target - state.lambdak)

    if local_z:
        dz = oracles.grad_h(state.z)
        local_linear = t_k * (gy - mu * (dy - dz))
        zplus = oracles.div_prox_map(state.z, local_linear, curvature_new)
    else:
        zplus = oracles.div_prox_map(state.x_anchor, lambda_plus, curvature_new)
		
    dzz = oracles.divergence(zplus, state.z)

    xplus, fplus, L_cur = backtracking_gradient(
        y, fy, gy, state.L_cur, oracles.f_eval,
        oracles.div_prox_map, oracles.divergence,
        max_backtracks=max_backtracks,
    )

    aff = np.sum((gy * (state.z - y)).astype(np.longdouble), dtype=np.longdouble)
    dzy = np.longdouble(oracles.divergence(state.z, y))
    philow = float(
        np.longdouble(fy)
        + aff
        + np.longdouble(mu) * dzy
        + np.longdouble(state.psi_z)
    )

    phi_plus = fplus + oracles.extra_Psi(xplus)

    return AbraStepResult(
        y=y,
        gy=gy,
        dy=dy,
        xplus=xplus,
        zplus=zplus,
        lambda_plus=lambda_plus,
        philow=philow,
        phi_plus=phi_plus,
        dzz=dzz,
        L_cur=L_cur,
    )


def _validate_abra_inputs(L, maxitrs, mu, restart_rule):
    if L <= 0:
        raise ValueError("L must be positive.")
    if mu < 0:
        raise ValueError("mu must be nonnegative.")
    if maxitrs <= 0:
        raise ValueError("maxitrs must be positive.")
    if restart_rule not in ('g', 'f'):
        raise ValueError("restart_rule must be either 'g' or 'f'.")


def _restart_state(state, mu, oracles):
    return acc_init_state(
        state.x,
        mu,
        state.L_cur,
        state.M_cur,
        oracles,
    )


def _accept_step(state, step, eta_plus_inv, phi_plus, xplus, oracles):
    state.x = xplus
    state.phi = phi_plus
    state.z = step.zplus
    state.lambdak = step.lambda_plus
    state.eta = 1.0 / eta_plus_inv
    state.psi_z = oracles.extra_Psi(state.z)
    state.L_cur = step.L_cur


def ABRA_GD(f, h, L, x0, maxitrs, mu=0.0, epsilon=0, verbose=True, verbskip=1,
            max_backtracks=50, restart=False, restart_rule='g',
            return_diagnostics=False, Mmin=0.0):
    """
    Adaptive Bregman Accelerated Gradient Descent.

    The ABRA state, oracle bundle, and histories are stored in small containers to
    keep the main loop readable. The numerical update is unchanged from the prior version.
    """
    _validate_abra_inputs(L, maxitrs, mu, restart_rule)

    f = CountedFunction(f)
    oracles = AbraOracles.from_problem(f, h)
    
    if restart is False:
        Mmin = 0.25

    if verbose:
        print("\nABRA_GD method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       eta_k        t_k         L_k         M_k     calls")

    start_time = time.time()
    hist = AbraHistories.allocate(maxitrs)

    state = acc_init_state(
        x0,
        mu,
        float(max(L, mu)),
        1.0,
        oracles,
    )

    for k in range(maxitrs):
        x_prev = np.copy(state.x)
        phi_prev = state.phi

        state.L_cur = max(0.5 * state.L_cur, mu)
        state.M_cur = max(0.5 * state.M_cur, Mmin)
        t_k = 0.0
        tau_k = np.nan

        while True:
            t_k = positive_tk_eta(state.eta, mu, state.M_cur)
            try:
                tau_k = compute_tau(t_k, mu, state.M_cur, state.L_cur)
            except ValueError:
                state.M_cur *= 2.0
                continue

            eta_plus_inv = state.M_cur * t_k * t_k - mu
            if eta_plus_inv <= 0.0:
                state.M_cur *= 2.0
                continue
    
            if np.isnan(state.M_cur) or np.isinf(state.M_cur):
                break

            step = BregPDStep(
                state,
                eta_plus_inv=eta_plus_inv,
                t_k=t_k,
                tau_k=tau_k,
                mu=mu,
                oracles=oracles,
                max_backtracks=max_backtracks,
            )
            state.L_cur = step.L_cur

            if state.phi <= step.phi_plus:
                xplus = state.x
                phi_plus = state.phi
            else:
                xplus = step.xplus
                phi_plus = step.phi_plus

            certificate_rhs = (
                (1.0 - t_k) * state.phi
                + t_k * step.philow
                - (mu + eta_plus_inv) * step.dzz
            )

            if phi_plus <= certificate_rhs:
                restart_now = False
                if restart and k > 0:
                    if restart_rule == 'f':
                        restart_now = step.phi_plus > phi_prev
                    else:
                        restart_now = np.dot(step.gy, step.xplus - x_prev) > 0.0

                if restart_now:
                    state.x = xplus
                    state.phi = phi_plus
                    state = _restart_state(state, mu, oracles)
                    t_k = 0.0
                    tau_k = np.nan
                else:
                    _accept_step(state, step, eta_plus_inv, phi_plus, xplus, oracles)
                break

            state.M_cur *= 2.0

        hist.record(k, state, t_k, tau_k, mu, f.oracle_calls)

        if verbose and k % verbskip == 0:
            print(
                "{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:6.1f}".format(
                    k, hist.F[k], hist.eta[k], hist.tk[k], hist.L[k], hist.M[k], hist.T[k]
                )
            )

        if (state.eta > 0) and (1.0 / state.eta) < epsilon:
            break

    n = k + 1
    result = (state.x,) + hist.result(n)
    if return_diagnostics:
        return result + (hist.diagnostics(n),)
    return result

def BPG(f, h, L, x0, maxitrs, epsilon=0, linesearch=True, ls_ratio=1.2,
        verbose=True, verbskip=1):
    """
    Bregman Proximal Gradient (BGP) method for min_{x in C} f(x) + Psi(x): 
        
    x(k+1) = argmin_{x in C} { Psi(x) + <f'(x(k)), x> + L(k) * D_h(x,x(k))}
 
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        epsilon:  stop if F(x[k])-F(x[k-1]) < epsilon, where F(x)=f(x)+Psi(x)
        linesearch:  whether or not perform line search (True or False)
        ls_ratio: backtracking line search parameter >= 1
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        F:  array storing F(x[k]) for all k
        Ls: array storing local Lipschitz constants obtained by line search
        T:  array storing oracle calls used up to iteration k
    """

    f = CountedFunction(f)

    if verbose:
        print("\nBPG_LS method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)         Lk      calls")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    Ls = np.ones(maxitrs) * L
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        F[k] = fx + h.extra_Psi(x)
        T[k] = f.oracle_calls
        
        if linesearch:
            L = L / ls_ratio
            x1 = h.div_prox_map(x, g, L)
            while f(x1) > fx + np.dot(g, x1-x) + L*h.divergence(x1, x):
                L = L * ls_ratio
                x1 = h.div_prox_map(x, g, L)
            x = x1
        else:
            x = h.div_prox_map(x, g, L)

        # store and display computational progress
        Ls[k] = L
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:6.1f}".format(k, F[k], L, T[k]))
            
        # stopping criteria
        if k > 0 and abs(F[k]-F[k-1]) < epsilon:
            break;

    F = F[0:k+1]
    Ls = Ls[0:k+1]
    T = T[0:k+1]
    return x, F, Ls, T


def solve_theta(theta, gamma, gainratio=1):
    """
    solve theta_k1 from the equation
    (1-theta_k1)/theta_k1^gamma = gainratio * 1/theta_k^gamma
    using Newton's method, starting from theta
    
    """
    ckg = theta**gamma / gainratio
    cta = theta
    eps = 1e-6 * theta
    phi = cta**gamma - ckg*(1-cta)
    while abs(phi) > eps:
        drv = gamma * cta**(gamma-1) + ckg
        cta = cta - phi / drv
        phi = cta**gamma - ckg*(1-cta)
        
    return cta
      

def ABPG(f, h, L, x0, gamma, maxitrs, epsilon=1e-14, theta_eq=False, 
         restart=False, restart_rule='g', verbose=True, verbskip=1):
    """
    Accelerated Bregman Proximal Gradient (ABPG) method for solving 
            minimize_{x in C} f(x) + Psi(x): 

    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        gamma:    triangle scaling exponent (TSE) for Bregman div D_h(x,y)
        maxitrs:  maximum number of iterations
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        theta_eq: calculate theta_k by solving equality using Newton's method
        restart:  restart the algorithm when overshooting (True or False)
        restart_rule: 'f' for function increasing or 'g' for gradient angle
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x: the last iterate of BPG
        F: array storing F(x[k]) for all k
        G: triangle scaling gains D(xk,yk) / D(zk,zk_1) / theta_k^gamma
        T: array storing oracle calls used up to iteration k
    """

    f = CountedFunction(f)

    if verbose:
        print("\nABPG method for minimize_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta" + 
              "        TSG       D(x+,y)     D(z+,z)    calls")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    z = np.copy(x0)
    theta = 1.0      # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        F[k] = fx + h.extra_Psi(x)
        T[k] = f.oracle_calls
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x     # only required for restart mode
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x + theta*z_1
        g = f.gradient(y)
        z = h.div_prox_map(z_1, g, theta**(gamma-1) * L)
        x = (1-theta)*x + theta*z

        # compute triangle scaling quantities
        dxy = h.divergence(x, y)
        dzz = h.divergence(z, z_1)
        Gdr = dxy / dzz / theta**gamma

        # store and display computational progress
        G[k] = Gdr
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:6.1f}".format(
                    k, F[k], theta, Gdr, dxy, dzz, T[k]))

        # restart if gradient predicts objective increase
        kk += 1
        if restart and k > 0:
            #if k > 0 and F[k] > F[k-1]:
            #if np.dot(g, x-x_1) > 0:
            if (restart_rule == 'f' and F[k] > F[k-1]) or (restart_rule == 'g' and np.dot(g, x-x_1) > 0):
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < epsilon:
            break;

    F = F[0:k+1]
    G = G[0:k+1]
    T = T[0:k+1]
    return x, F, G, T


def ABPG_expo(f, h, L, x0, gamma0, maxitrs, epsilon=1e-14, delta=0.2, 
              theta_eq=True, checkdiv=False, Gmargin=10, restart=False, 
              restart_rule='g', verbose=True, verbskip=1):
    """
    Accelerated Bregman Proximal Gradient method with exponent adaption for
            minimize_{x in C} f(x) + Psi(x) 
 
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        gamma0:   initial triangle scaling exponent(TSE) for D_h(x,y) (>2)
        maxitrs:  maximum number of iterations
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        delta:    amount to decrease TSE for exponent adaption
        theta_eq: calculate theta_k by solving equality using Newton's method
        checkdiv: check triangle scaling inequality for adaption (True/False)
        Gmargin:  extra gain margin allowed for checking TSI
        restart:  restart the algorithm when overshooting (True or False)
        restart_rule: 'f' for function increasing or 'g' for gradient angle
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        F:  array storing F(x[k]) for all k
        Gamma: gamma_k obtained at each iteration
        G:  triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma_k
        T:  array storing oracle calls used up to iteration k
    """
    
    f = CountedFunction(f)

    if verbose:
        print("\nABPG_expo method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta       gamma" +
              "        TSG       D(x+,y)     D(z+,z)     time")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    Gamma = np.ones(maxitrs) * gamma0
    T = np.zeros(maxitrs)
    
    gamma = gamma0
    x = np.copy(x0)
    z = np.copy(x0)
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        F[k] = fx + h.extra_Psi(x)
        T[k] = f.oracle_calls
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x_1 + theta*z_1
        #g = f.gradient(y)
        fy, g = f.func_grad(y)
        
        condition = True
        while condition:    # always execute at least once per iteration 
            z = h.div_prox_map(z_1, g, theta**(gamma-1) * L)
            x = (1-theta)*x_1 + theta*z

            # compute triangle scaling quantities
            dxy = h.divergence(x, y)
            dzz = h.divergence(z, z_1)
            Gdr = dxy / dzz / theta**gamma

            if checkdiv:
                condition = (dxy > Gmargin * (theta**gamma) * dzz )
            else:
                condition = (f(x) > fy + np.dot(g, x-y) + theta**gamma*L*dzz)
                
            if condition and gamma > 1:
                gamma = max(gamma - delta, 1)
            else: 
                condition = False
               
        # store and display computational progress
        G[k] = Gdr
        Gamma[k] = gamma
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:10.3e}  {7:6.1f}".format(
                    k, F[k], theta, gamma, Gdr, dxy, dzz, T[k]))

        # restart if gradient predicts objective increase
        kk += 1
        if restart:
            #if k > 0 and F[k] > F[k-1]:
            #if np.dot(g, x-x_1) > 0:
            if (restart_rule == 'f' and F[k] > F[k-1]) or (restart_rule == 'g' and np.dot(g, x-x_1) > 0):
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < epsilon:
            break;

    F = F[0:k+1]
    Gamma = Gamma[0:k+1]
    G = G[0:k+1]
    T = T[0:k+1]
    return x, F, Gamma, G, T


def ABPG_gain(f, h, L, x0, gamma, maxitrs, epsilon=1e-14, G0=1, 
              ls_inc=1.2, ls_dec=1.2, theta_eq=True, checkdiv=False, 
              restart=False, restart_rule='g', verbose=True, verbskip=1):
    """
    Accelerated Bregman Proximal Gradient (ABPG) method with gain adaption for 
            minimize_{x in C} f(x) + Psi(x): 
    
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        gamma:    triangle scaling exponent(TSE) for Bregman distance D_h(x,y)
        G0:       initial value for triangle scaling gain
        maxitrs:  maximum number of iterations
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        ls_inc:   factor of increasing gain (>=1)
        ls_dec:   factor of decreasing gain (>=1)
        theta_eq: calculate theta_k by solving equality using Newton's method
        checkdiv: check triangle scaling inequality for adaption (True/False)
        restart:  restart the algorithm when overshooting (True/False)
        restart_rule: 'f' for function increasing or 'g' for gradient angle
        verbose:  display computational progress (True/False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        F:  array storing F(x[k]) for all k
        Gain: triangle scaling gains G_k obtained by LS at each iteration
        Gdiv: triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma_k
        Gavg: geometric mean of G_k at all steps up to iteration k
        T:  array storing oracle calls used up to iteration k
    """
    f = CountedFunction(f)

    if verbose:
        print("\nABPG_gain method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta         Gk" + 
              "         TSG       D(x+,y)     D(z+,z)      Gavg      calls")

    start_time = time.time()    
    F = np.zeros(maxitrs)
    Gain = np.ones(maxitrs) * G0
    Gdiv = np.zeros(maxitrs)
    Gavg = np.zeros(maxitrs)
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    z = np.copy(x0)
    G = G0
    # logGavg = (gamma*log(G0) + log(G_1) + ... + log(Gk)) / (k+gamma)
    sumlogG = gamma * np.log(G) 
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        F[k] = fx + h.extra_Psi(x)
        T[k] = f.oracle_calls
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        # adaptive option: always try a smaller Gain first before line search
        G_1 = G
        theta_1 = theta
        
        G = G / ls_dec
        
        condition = True
        while condition:
            if kk > 0:
                if theta_eq:
                    theta = solve_theta(theta_1, gamma, G / G_1)
                else:
                    alpha = G / G_1
                    theta = theta_1*((1+alpha*(gamma-1))/(gamma*alpha+theta_1))

            y = (1-theta)*x_1 + theta*z_1
            #g = f.gradient(y)
            fy, g = f.func_grad(y)
        
            z = h.div_prox_map(z_1, g, theta**(gamma-1) * G * L)
            x = (1-theta)*x_1 + theta*z

            # compute triangle scaling quantities
            dxy = h.divergence(x, y)
            dzz = h.divergence(z, z_1)
            if dzz < epsilon:
                break
            
            Gdr = dxy / dzz / theta**gamma

            if checkdiv:
                condition = (Gdr > G )
            else:
                condition = (f(x) > fy + np.dot(g,x-y) + theta**gamma*G*L*dzz)
                
            if condition:
                G = G * ls_inc
               
        # store and display computational progress
        Gain[k] = G
        Gdiv[k] = Gdr
        sumlogG += np.log(G)
        Gavg[k] = np.exp(sumlogG / (gamma + k)) 
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:10.3e}  {7:10.3e}  {8:6.1f}".format(
                    k, F[k], theta, G, Gdr, dxy, dzz, Gavg[k], T[k]))

        # restart if gradient predicts objective increase
        kk += 1
        if restart:
            #if k > 0 and F[k] > F[k-1]:
            #if np.dot(g, x-x_1) > 0:
            if (restart_rule == 'f' and F[k] > F[k-1]) or (restart_rule == 'g' and np.dot(g, x-x_1) > 0):
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < epsilon:
            break;

    F = F[0:k+1]
    Gain = Gain[0:k+1]
    Gdiv = Gdiv[0:k+1]
    Gavg = Gavg[0:k+1]
    T = T[0:k+1]
    return x, F, Gain, Gdiv, Gavg, T


def ABDA(f, h, L, x0, gamma, maxitrs, epsilon=1e-14, theta_eq=True,
           verbose=True, verbskip=1):
    """
    Accelerated Bregman Dual Averaging (ABDA) method for solving
            minimize_{x in C} f(x) + Psi(x) 
    
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        gamma:    triangle scaling exponent (TSE) for Bregman distance D_h(x,y)
        maxitrs:  maximum number of iterations
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        theta_eq: calculate theta_k by solving equality using Newton's method
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x: the last iterate of BPG
        F: array storing F(x[k]) for all k
        G: triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma
        T: array storing oracle calls used up to iteration k
    """
    # Simple restart schemes for dual averaging method do not work!
    restart = False
    
    f = CountedFunction(f)

    if verbose:
        print("\nABDA method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta" + 
              "        TSG       D(x+,y)     D(z+,z)     time")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    G = np.zeros(maxitrs)
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    z = np.copy(x0)
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    gavg = np.zeros(x.size)
    csum = 0
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        F[k] = fx + h.extra_Psi(x)
        T[k] = f.oracle_calls
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x_1 + theta*z_1
        g = f.gradient(y)
        gavg = gavg + theta**(1-gamma) * g
        csum = csum + theta**(1-gamma)
        z = h.prox_map(gavg/csum, L/csum)
        x = (1-theta)*x_1 + theta*z

        # compute triangle scaling quantities
        dxy = h.divergence(x, y)
        dzz = h.divergence(z, z_1)
        Gdr = dxy / dzz / theta**gamma

        # store and display computational progress
        G[k] = Gdr
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:6.1f}".format(
                    k, F[k], theta, Gdr, dxy, dzz, T[k]))

        kk += 1
        # restart does not work for ABDA (restart = False)
        if restart:
            if k > 0 and F[k] > F[k-1]:
            #if np.dot(g, x-x_1) > 0:   # this does not work for dual averaging
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y
                gavg = np.zeros(x.size) # this is why restart does not work
                csum = 0

        # stopping criteria
        if dzz < epsilon:
            break;

    F = F[0:k+1]
    G = G[0:k+1]
    T = T[0:k+1]
    return x, F, G, T
