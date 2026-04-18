# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import numpy as np
import time
import warnings


def positive_tk(alpha_k, mu, L, c):
    """
    Positive root of
        (1 - t) * alpha_k + t * mu >= L * c * t^2.

    Uses a numerically stable formula for the positive root of
        L c t^2 + (alpha_k - mu) t - alpha_k = 0,
    avoiding cancellation when c is large and the root is very small.
    """
    if L <= 0:
        raise ValueError("L must be positive.")
    if c <= 0:
        raise ValueError("c must be positive.")
    if alpha_k < mu:
        raise ValueError("alpha_k must satisfy alpha_k >= mu.")

    if alpha_k == 0.0 and mu == 0.0:
        return 0.0

    Lc = L * c
    delta = alpha_k - mu
    sqrt_disc = math.hypot(delta, 2.0 * math.sqrt(Lc) * math.sqrt(alpha_k))
    t_pos = (2.0 * alpha_k) / (delta + sqrt_disc)
    return min(1.0, max(0.0, t_pos))


def backtracking_gradient(y, fy, gy, current_L, f_eval, div_prox_map, divergence,
                          max_backtracks=50):
    """
    Backtracking for the mirror/prox-gradient step.

    Uses only f(xplus), not func_grad(xplus), because the gradient at xplus
    is not needed here.
    """
    if current_L <= 0:
        raise ValueError("current_L must be positive.")
    if max_backtracks <= 0:
        raise ValueError("max_backtracks must be positive.")

    xplus = None
    fplus = None

    for _ in range(max_backtracks):
        xplus = div_prox_map(y, gy, current_L)
        step = xplus - y
        dxy = divergence(xplus, y)
        fplus = f_eval(xplus)

        if (fplus - fy) <= np.dot(gy, step) + current_L * dxy:
            return xplus, fplus, current_L

        current_L *= 2.0

    warnings.warn(
        "backtracking_gradient failed to find a valid current_L; "
        "returning the last trial as accepted.",
        RuntimeWarning,
        stacklevel=2,
    )
    return xplus, fplus, current_L


def acc_init_state(x_start, mu, current_L, current_c, f_eval, grad_h, extra_Psi):
    """
    Initialize/reset one ABRA_GD phase.

    The current point x_start becomes the new anchor point for the phase.
    The returned state is a sentinel state with eta_k = +inf, meaning the next
    loop iteration must execute acc_first_step from this new anchor.
    """
    x_anchor = np.copy(x_start)
    dx_anchor = grad_h(x_anchor)

    x = np.copy(x_anchor)
    z = np.copy(x_anchor)
    lambdak = np.zeros_like(x_anchor)
    alpha_k = mu
    eta_k = np.inf
    phi_x = f_eval(x)
    phi_x += extra_Psi(x)
    psi_z = extra_Psi(z)

    return (
        x,
        z,
        lambdak,
        alpha_k,
        eta_k,
        phi_x,
        psi_z,
        x_anchor,
        dx_anchor,
        current_L,
        current_c,
    )


def acc_first_step(x_start, mu, x_anchor, dx_anchor, current_L, func_grad, f_eval, grad_h,
                   extra_Psi, div_prox_map, divergence, max_backtracks=50):
    """
    Safe first accelerated step for a fresh ABRA_GD phase.

    The current phase is anchored at x_anchor. Starting from x_start, this
    routine performs one prox-gradient step with backtracking and initializes
    the accelerated state with z = x.
    """
    fx_start, gx_start = func_grad(x_start)
    dx_start = grad_h(x_start)

    x, f_x, current_L = backtracking_gradient(
        x_start, fx_start, gx_start, current_L, f_eval, div_prox_map, divergence,
        max_backtracks=max_backtracks,
    )
    z = np.copy(x)
    lambdak = gx_start - mu * (dx_start - dx_anchor)
    alpha_k = current_L

    phi_x = f_x + extra_Psi(x)
    psi_z = phi_x - f_x
    dzz = divergence(z, x_start)

    return x, z, lambdak, alpha_k, phi_x, psi_z, dzz, current_L


def fallback_gd_step(x, current_L, func_grad, f_eval, extra_Psi,
                     div_prox_map, divergence, max_backtracks=50):
    """
    Fallback non-accelerated prox-gradient step from the current primal point.

    This is used when the acceleration parameter c becomes too large. The
    primal point is updated by one backtracked GD/BPG step, while the dual
    accelerated state (z, lambda, alpha, eta) is left unchanged.
    """
    fx, gx = func_grad(x)
    xplus, fplus, current_L = backtracking_gradient(
        x, fx, gx, current_L, f_eval, div_prox_map, divergence,
        max_backtracks=max_backtracks,
    )
    phi_plus = fplus + extra_Psi(xplus)
    return xplus, phi_plus, current_L


def BregPDStep(x, z, lambdak, alpha_k, t_k, mu, x_anchor, dx_anchor, current_L, psi_z,
               func_grad, f_eval, grad_h, extra_Psi, div_prox_map, divergence):
    """
    One generic primal-dual Bregman step for k >= 1.

    Returns alpha_plus and dzz directly so ABRA_GD does not recompute them.
    """
    omt = 1.0 - t_k

    y = x + t_k * (z - x)
    fy, gy = func_grad(y)
    dy = grad_h(y)

    alpha_plus = omt * alpha_k + t_k * mu
    if alpha_plus <= 0:
        raise ValueError(
            "alpha_plus <= 0 in BregPDStep. "
            "The first step must be handled explicitly."
        )

    lambda_plus = omt * lambdak + t_k * (gy - mu * (dy - dx_anchor))
    zplus = div_prox_map(x_anchor, lambda_plus, alpha_plus)
    dzz = divergence(zplus, z)

    xplus, fplus, current_L = backtracking_gradient(
        y, fy, gy, current_L, f_eval, div_prox_map, divergence
    )
    
    # Gigh precision accumulation
    aff = np.sum((gy * (z - y)).astype(np.longdouble), dtype=np.longdouble)
    dzy = np.longdouble(divergence(z, y))
    philowk = float(np.longdouble(fy) + aff + np.longdouble(mu) * dzy + np.longdouble(psi_z))
    
    # philowk = fy + np.dot(gy, z - y) + mu * divergence(z, y) + psi_z
    psi_xplus = extra_Psi(xplus)
    phi_plus = fplus + psi_xplus

    return y, gy, dy, xplus, zplus, lambda_plus, philowk, phi_plus, alpha_plus, dzz, current_L


def ABRA_GD(f, h, L, x0, maxitrs, mu=0.0, epsilon=1e-14, verbose=True, verbskip=1,
            max_backtracks=50, c_min=1.0, c_max=1.0e12, restart=False, restart_rule='g'):
    """
    Adaptive Bregman Accelerated Gradient Descent.

    Optimizations:
    - carries phi_x instead of recomputing f(x)+Psi(x) every iteration;
    - caches psi_z = Psi(z_k) across the inner loop;
    - uses f(xplus) rather than func_grad(xplus) in backtracking;
    - tracks alpha_k directly, and eta_k explicitly to detect mandatory first steps.
    """
    if L <= 0:
        raise ValueError("L must be positive.")
    if mu < 0:
        raise ValueError("mu must be nonnegative.")
    if maxitrs <= 0:
        raise ValueError("maxitrs must be positive.")
    if c_min <= 0:
        raise ValueError("c_min must be positive.")
    if c_max <= 0:
        raise ValueError("c_max must be positive.")
    if c_max < c_min:
        raise ValueError("c_max must satisfy c_max >= c_min.")
    if restart_rule not in ('g', 'f'):
        raise ValueError("restart_rule must be either 'g' or 'f'.")

    # Local bindings: cheaper in Python loops than repeated attribute lookups.
    func_grad = f.func_grad
    f_eval = f if callable(f) else (lambda x: f.func_grad(x)[0])
    grad_h = h.gradient
    extra_Psi = h.extra_Psi
    div_prox_map = h.div_prox_map
    divergence = h.divergence

    if verbose:
        print("\nABRA_GD method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       eta_k        t_k         L_k         c_k      time")

    start_time = time.time()

    F = np.zeros(maxitrs)
    tk_hist = np.zeros(maxitrs)
    eta_hist = np.full(maxitrs, np.nan)
    ck_hist = np.full(maxitrs, np.nan)
    alpha_hist = np.full(maxitrs, np.nan)
    T = np.zeros(maxitrs)

    current_L = float(max(L, mu))
    current_c = max(1.0, c_min)

    (
        x,
        z,
        lambdak,
        alpha_k,
        eta_k,
        phi_x,
        psi_z,
        x_anchor,
        dx_anchor,
        current_L,
        current_c,
    ) = acc_init_state(x0, mu, current_L, current_c, f_eval, grad_h, extra_Psi)

    trigger_restart = False
    trigger_fallback = False
    for k in range(maxitrs):
        trigger_restart = False
        trigger_fallback = False
        x_prev = np.copy(x)
        phi_prev = phi_x

        if np.isinf(eta_k):
            x, z, lambdak, alpha_k, phi_x, psi_z, dzz, current_L = acc_first_step(
                x,
                mu,
                x_anchor,
                dx_anchor,
                current_L,
                func_grad,
                f_eval,
                grad_h,
                extra_Psi,
                div_prox_map,
                divergence,
                max_backtracks=max_backtracks,
            )
            eta_k = np.inf if alpha_k == mu else 1.0 / (alpha_k - mu)
            t_k = 0.0
            current_c = max(1.0, c_min)
        else:
            # optimistic trial (backtracking LS)
            current_L = max(0.5 * current_L, mu)
            current_c = max(0.5 * current_c, c_min)

            while True:
                if current_c > c_max: # Fallback to simple GD step
                    t_k = 0.0
                    # current_c = max(1.0, c_min)
                else:
                    t_k = positive_tk(alpha_k, mu, current_L, current_c)

                y_cand, gy_cand, dy_cand, x_cand, zplus, lambda_plus, philowk, phi_cand, alpha_plus, dzz, current_L = BregPDStep(
                    x=x,
                    z=z,
                    lambdak=lambdak,
                    alpha_k=alpha_k,
                    t_k=t_k,
                    mu=mu,
                    x_anchor=x_anchor,
                    dx_anchor=dx_anchor,
                    current_L=current_L,
                    psi_z=psi_z,
                    func_grad=func_grad,
                    f_eval=f_eval,
                    grad_h=grad_h,
                    extra_Psi=extra_Psi,
                    div_prox_map=div_prox_map,
                    divergence=divergence,
                )

                # Keep the better primal point before checking the descent inequality.
                if phi_x <= phi_cand:
                    xplus = x
                    phi_plus = phi_x
                else:
                    xplus = x_cand
                    phi_plus = phi_cand

                phi_delta = (phi_plus - phi_x) - t_k * (philowk - phi_x)
                rhs_delta = -alpha_plus * dzz
                c_check_tol = 0.1 * epsilon

                if phi_delta <= rhs_delta + c_check_tol:
                    x = xplus
                    phi_x = phi_plus

                    restart_now = False

                    if restart and k > 0:
                        if restart_rule == 'f':
                            restart_now = (phi_cand > phi_prev)
                        else:
                            restart_now = (np.dot(gy_cand, x_cand - x_prev) > 0.0)

                    if restart_now:
                        print("Restarting Algorithm")
                        trigger_restart = True
                        (
                            x,
                            z,
                            lambdak,
                            alpha_k,
                            eta_k,
                            phi_x,
                            psi_z,
                            x_anchor,
                            dx_anchor,
                            current_L,
                            current_c,
                        ) = acc_init_state(x, mu, current_L, current_c, f_eval, grad_h, extra_Psi)
                        t_k = 0.0
                    else:
                        z = zplus
                        lambdak = lambda_plus
                        alpha_k = alpha_plus
                        eta_k = np.inf if alpha_k == mu else 1.0 / (alpha_k - mu)
                        psi_z = extra_Psi(z)
                    break

                current_c *= 2.0
        if trigger_restart:
            print("current_c: {5:10.3e}", current_c)
            print("current_tk: {5:10.3e}", t_k)
        if trigger_fallback:
            print("Fallback to GD because c exceeded c_max")
            print("current_c: {5:10.3e}", current_c)
            print("current_tk: {5:10.3e}", t_k)

        F[k] = phi_x
        tk_hist[k] = t_k
        eta_hist[k] = eta_k
        ck_hist[k] = current_c
        alpha_hist[k] = alpha_k
        T[k] = time.time() - start_time

        if verbose and k % verbskip == 0:
            print(
                "{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:6.1f}".format(
                    k, F[k], eta_hist[k], tk_hist[k], current_L, current_c, T[k]
                )
            )

        if eta_k < epsilon:
            break

        if eta_k < epsilon:
            break

    return (
        x,
        F[:k + 1],
        tk_hist[:k + 1],
        eta_hist[:k + 1],
        ck_hist[:k + 1],
        alpha_hist[:k + 1],
        T[:k + 1],
    )

def BPG(f, h, L, x0, maxitrs, epsilon=1e-14, linesearch=True, ls_ratio=1.2,
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
        T:  array storing time used up to iteration k
    """

    if verbose:
        print("\nBPG_LS method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)         Lk       time")
    
    start_time = time.time()
    F = np.zeros(maxitrs)
    Ls = np.ones(maxitrs) * L
    T = np.zeros(maxitrs)
    
    x = np.copy(x0)
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        F[k] = fx + h.extra_Psi(x)
        T[k] = time.time() - start_time
        
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
        T: array storing time used up to iteration k
    """

    if verbose:
        print("\nABPG method for minimize_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta" + 
              "        TSG       D(x+,y)     D(z+,z)     time")
    
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
        T[k] = time.time() - start_time
        
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
        T:  array storing time used up to iteration k
    """
    
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
        T[k] = time.time() - start_time
        
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
        T:  array storing time used up to iteration k
    """
    if verbose:
        print("\nABPG_gain method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta         Gk" + 
              "         TSG       D(x+,y)     D(z+,z)      Gavg       time")

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
        T[k] = time.time() - start_time
        
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
        T: array storing time used up to iteration k
    """
    # Simple restart schemes for dual averaging method do not work!
    restart = False
    
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
        T[k] = time.time() - start_time
        
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
