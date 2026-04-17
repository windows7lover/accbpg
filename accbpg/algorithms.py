# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import numpy as np
import time
import warnings

def positive_tk(eta_k, mu, L, c):
    """
    Positive root of
        (1 - t) * alpha_k + t * mu >= L * c * t^2,
    where alpha_k = mu + 1 / eta_k and eta_k = +inf is allowed.

    Returns the positive root clipped to [0, 1].
    """
    if L <= 0:
        raise ValueError("L must be positive.")
    if c <= 0:
        raise ValueError("c must be positive.")

    if np.isinf(eta_k):
        alpha_k = mu
    else:
        if eta_k <= 0:
            raise ValueError("eta_k must be positive or +inf.")
        alpha_k = mu + 1.0 / eta_k

    # Degenerate case: alpha_k = mu = 0 => only t = 0 works.
    if alpha_k == 0.0 and mu == 0.0:
        return 0.0

    disc = (mu - alpha_k) ** 2 + 4.0 * L * c * alpha_k
    t_pos = ((mu - alpha_k) + math.sqrt(disc)) / (2.0 * L * c)
    return min(1.0, max(0.0, t_pos))


def backtracking_gradient(y, f, h, gy, fy, current_L, max_backtracks=50):
    """
    Backtracking for the mirror/prox-gradient step.

    Accept when
        f(xplus) <= fy + <gy, xplus - y> + current_L * D_h(xplus, y).

    If backtracking fails after max_backtracks attempts, emit a warning
    and return the last trial as if it were accepted.
    """
    if current_L <= 0:
        raise ValueError("current_L must be positive.")
    if max_backtracks <= 0:
        raise ValueError("max_backtracks must be positive.")

    xplus = None
    fplus = None

    for _ in range(max_backtracks):
        xplus = h.div_prox_map(y, gy, current_L)
        fplus, _ = f.func_grad(xplus)

        sufficient_decrease = (
            (fplus - fy) <= np.dot(gy, xplus - y) + current_L * h.divergence(xplus, y)
        )
        if sufficient_decrease:
            return xplus, fplus, current_L

        current_L *= 2.0

    warnings.warn(
        "backtracking_gradient failed to find a valid current_L; "
        "returning the last trial as accepted.",
        RuntimeWarning,
        stacklevel=2,
    )
    return xplus, fplus, current_L


def BregPDStep(x, z, lambdak, eta_k, t_k, mu, f, h, x0, dx0, current_L):
    """
    One generic primal-dual Bregman step for k >= 1.
    """
    y = (1.0 - t_k) * x + t_k * z
    fy, gy = f.func_grad(y)
    dy = h.gradient(y)

    # eta_{k+1} = eta_k / (1 - t_k), preserving +inf.
    if np.isinf(eta_k) or t_k >= 1.0:
        eta_plus = np.inf
    else:
        eta_plus = eta_k / (1.0 - t_k)

    alpha_plus = mu if np.isinf(eta_plus) else mu + 1.0 / eta_plus
    if alpha_plus <= 0:
        raise ValueError(
            "alpha_plus <= 0 in BregPDStep. "
            "The first step must be handled explicitly."
        )

    lambda_plus = (1.0 - t_k) * lambdak + t_k * (gy - mu * (dy - dx0))
    zplus = h.div_prox_map(x0, lambda_plus, alpha_plus)

    xplus, fplus, current_L = backtracking_gradient(y, f, h, gy, fy, current_L)

    philowk = fy + np.dot(gy, z - y) + mu * h.divergence(z, y) + h.extra_Psi(z)
    phi_plus = fplus + h.extra_Psi(xplus)

    return xplus, zplus, lambda_plus, philowk, phi_plus, eta_plus, current_L


def ABRA_GD(f, h, L, x0, maxitrs, mu=0.0, epsilon=1e-14, verbose=True, verbskip=1):
    """
    Adaptive Bregman Accelerated Gradient Descent.

    The first iteration is treated as a special initialization step.
    After that, the generic primal-dual update is used for k >= 1.

    Returns
    -------
    x        : last iterate
    F        : objective values at accepted iterates
    tk_hist  : accepted t_k values
    eta_hist : accepted eta_k values
    T        : cumulative times
    """
    if L <= 0:
        raise ValueError("L must be positive.")
    if mu < 0:
        raise ValueError("mu must be nonnegative.")
    if maxitrs <= 0:
        raise ValueError("maxitrs must be positive.")

    if verbose:
        print("\nABRA_GD method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       eta_k        t_k         L_k         c_k      time")

    start_time = time.time()

    F = np.zeros(maxitrs)
    tk_hist = np.zeros(maxitrs)
    eta_hist = np.full(maxitrs, np.nan)
    T = np.zeros(maxitrs)

    x = np.copy(x0)
    z = np.copy(x0)
    lambdak = np.zeros_like(x0)
    eta_k = np.inf
    dx0 = h.gradient(x0)

    current_L = float(max(L, mu))
    current_c = 1.0

    for k in range(maxitrs):
        phi_xk = f(x) + h.extra_Psi(x)

        if k == 0:
            # Explicit first step.
            # We do not use the generic t_k formula here because when
            # eta_0 = +inf and mu = 0, it forces t_0 = 0.
            t_k = 0.0

            y = np.copy(x0)
            fy, gy = f.func_grad(y)
            dy = h.gradient(y)

            xplus, fplus, current_L = backtracking_gradient(y, f, h, gy, fy, current_L)

            # Force z_1 = x_1 at initialization.
            zplus = np.copy(xplus)

            # Choose lambda_1 so that the dual prox update matches z_1 = x_1.
            lambda_plus = gy - mu * (dy - dx0)

            # Enforce alpha_1 = mu + 1 / eta_1 = current_L.
            if current_L < mu:
                raise ValueError("First step requires current_L >= mu.")
            if np.isclose(current_L, mu):
                eta_plus = np.inf
            else:
                eta_plus = 1.0 / (current_L - mu)

            phi_plus = fplus + h.extra_Psi(xplus)

        else:
            # Optimistic trial values for k >= 1.
            current_L = max(0.5 * current_L, mu)
            # current_c = max(0.5 * current_c, 1.0)
            current_c = 0.5 * current_c

            sufficient_decrease = False
            while not sufficient_decrease:
                t_k = positive_tk(eta_k, mu, current_L, current_c)

                xplus, zplus, lambda_plus, philowk, phi_plus, eta_plus, current_L = BregPDStep(
                    x=x,
                    z=z,
                    lambdak=lambdak,
                    eta_k=eta_k,
                    t_k=t_k,
                    mu=mu,
                    f=f,
                    h=h,
                    x0=x0,
                    dx0=dx0,
                    current_L=current_L,
                )

                alpha_plus = mu + 1.0 / eta_plus

                sufficient_decrease = (
                    phi_plus - ((1.0 - t_k) * phi_xk+ t_k * philowk)
                    <=- alpha_plus * h.divergence(zplus, z)
                )

                if not sufficient_decrease:
                    current_c *= 2.0

        dzz = h.divergence(zplus, z)

        x = xplus
        z = zplus
        lambdak = lambda_plus
        eta_k = eta_plus

        F[k] = phi_plus
        tk_hist[k] = t_k
        eta_hist[k] = eta_k
        T[k] = time.time() - start_time

        if verbose and k % verbskip == 0:
            print(
                "{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:6.1f}".format(
                    k, F[k], eta_hist[k], tk_hist[k], current_L, current_c, T[k]
                )
            )

        if dzz < epsilon:
            break

    F = F[:k + 1]
    tk_hist = tk_hist[:k + 1]
    eta_hist = eta_hist[:k + 1]
    T = T[:k + 1]

    return x, F, tk_hist, eta_hist, T
        

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
