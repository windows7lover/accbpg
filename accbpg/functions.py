# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def _require_finite_positive(x, name, strict=True):
    x = _as_float_array(x)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")
    ok = np.all(x > 0) if strict else np.all(x >= 0)
    if not ok:
        op = "> 0" if strict else ">= 0"
        raise ValueError(f"{name} must satisfy {op}.")
    return x


def _xlogx_over_y(x, y):
    """Return x*log(x/y), with the convention 0*log(0/y)=0."""
    x = _as_float_array(x)
    y = _as_float_array(y)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = x[mask] * (np.log(x[mask]) - np.log(y[mask]))
    return out


class RSmoothFunction:
    """
    Relatively-Smooth Function, can query f(x) and gradient
    """
    def __call__(self, x):
        assert 0, "RSmoothFunction: __call__(x) is not defined"
        
    def gradient(self, x):
        assert 0, "RSmoothFunction: gradient(x) is not defined"
 
    def func_grad(self, x, flag):
        """
        flag=0: function, flag=1: gradient, flag=2: function & gradient 
        """
        assert 0, "RSmoothFunction: func_grad(x, flag) is not defined"


class DOptimalObj(RSmoothFunction):
    """
    f(x) = - log(det(H*diag(x)*H')) where H is an m by n matrix, m < n
    """
    def __init__(self, H):
        self.H = H
        self.m = H.shape[0]
        self.n = H.shape[1]
        assert self.m < self.n, "DOptimalObj: need m < n"
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)
        
    def gradient(self, x):
        return self.func_grad(x, flag=1)
        
    def func_grad(self, x, flag=2):
        x = _require_finite_positive(x, "DOptimalObj: x", strict=False)
        if x.size != self.n:
            raise ValueError("DOptimalObj: x.size not equal to n")

        HXHT = (self.H * x) @ self.H.T
        HXHT = 0.5 * (HXHT + HXHT.T)  # remove roundoff asymmetry

        sign, logdet = np.linalg.slogdet(HXHT)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("DOptimalObj: H diag(x) H.T is not positive definite.")

        if flag == 0:
            return -logdet

        # Avoid forming the inverse.  solve(HXHT, H) = HXHT^{-1} H.
        solved = np.linalg.solve(HXHT, self.H)
        g = -np.einsum("ij,ij->j", self.H, solved)

        if flag == 1:
            return g

        return -logdet, g

    def func_grad_slow(self, x, flag=2):
        # Kept for reference.  It now delegates to the stable implementation.
        return self.func_grad(x, flag=flag)


class PoissonRegression(RSmoothFunction):
    """
    f(x) = D_KL(b, Ax) for linear inverse problem A * x = b
    """
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], "A and b sizes not matching"
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)
    
    def func_grad(self, x, flag=2):
        x = _as_float_array(x)
        if x.size != self.n:
            raise ValueError("PoissonRegression: x.size not equal to n.")
        Ax = self.A @ x
        if np.any(Ax <= 0) or not np.all(np.isfinite(Ax)):
            raise ValueError("PoissonRegression: Ax must be finite and strictly positive.")

        # D_KL(b, Ax) = sum_i b_i log(b_i/Ax_i) + Ax_i - b_i,
        # with 0 log 0 convention handled by _xlogx_over_y.
        if flag == 0:
            return float(np.sum(_xlogx_over_y(self.b, Ax) + Ax - self.b))

        g = self.A.T @ (1.0 - self.b / Ax)
        if flag == 1:
            return g

        fx = float(np.sum(_xlogx_over_y(self.b, Ax) + Ax - self.b))
        return fx, g


class KLdivRegression(RSmoothFunction):
    """
    f(x) = D_KL(Ax, b) for linear inverse problem A * x = b
    """
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], "A and b size not matching"
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)
    
    def func_grad(self, x, flag=2):
        x = _as_float_array(x)
        if x.size != self.n:
            raise ValueError("KLdivRegression: x.size not equal to n.")
        Ax = self.A @ x
        if np.any(Ax <= 0) or not np.all(np.isfinite(Ax)):
            raise ValueError("KLdivRegression: Ax must be finite and strictly positive for gradient evaluation.")
        if np.any(self.b <= 0):
            raise ValueError("KLdivRegression: b must be strictly positive.")

        if flag == 0:
            return float(np.sum(Ax * (np.log(Ax) - np.log(self.b)) - Ax + self.b))

        g = self.A.T @ (np.log(Ax) - np.log(self.b))
        if flag == 1:
            return g

        fx = float(np.sum(Ax * (np.log(Ax) - np.log(self.b)) - Ax + self.b))
        return fx, g

           
           
#######################################################################


class LegendreFunction:
    """
    Function of Legendre type, used as the kernel of Bregman divergence for
    composite optimization 
         minimize_{x in C} f(x) + Psi(x) 
    where f is L-smooth relative to a Legendre function h(x),
          Psi(x) is an additional simple convex function.
    """
    def __call__(self, x):
        assert 0, "LegendreFunction: __call__(x) is not defined."
        
    def extra_Psi(self, x):
        return 0
        
    def gradient(self, x):
        assert 0, "LegendreFunction: gradient(x) is not defined."

    def divergence(self, x, y):
        """
        Return D(x,y) = h(x) - h(y) - <h'(y), x-y>
        """
        assert 0, "LegendreFunction: divergence(x,y) is not defined."

    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * h(x) }
        """
        assert 0, "LegendreFunction: prox_map(x, L) is not defined."

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  } 
        default implementation by calling prox_map(g - L*g(y), L)
        """
        assert y.shape == g.shape, "Vectors y and g should have same size." 
        assert L > 0, "Relative smoothness constant L should be positive."
        return self.prox_map(g - L*self.gradient(y), L)


class BurgEntropy(LegendreFunction):
    """
    h(x) = - sum_{i=1}^n log(x[i]) for x > 0
    """
    def __call__(self, x):
        x = _require_finite_positive(x, "BurgEntropy: x")
        return float(-np.sum(np.log(x)))
    
    def gradient(self, x):
        x = _require_finite_positive(x, "BurgEntropy: x")
        return -1.0 / x
    
    def divergence(self, x, y):
        x = _require_finite_positive(x, "BurgEntropy: x")
        y = _require_finite_positive(y, "BurgEntropy: y")
        if x.shape != y.shape:
            raise ValueError("BurgEntropy: x and y have different shapes.")
        r = x / y - 1.0
        # r - log(1+r) is accurate near r=0.
        return float(np.sum(r - np.log1p(r)))        

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { <g, x> + L * h(x) } 
        This function needs to be replaced with inheritance
        """
        if L <= 0:
            raise ValueError("BurgEntropy prox_map only takes positive L value.")
        g = _require_finite_positive(g, "BurgEntropy prox_map: g")
        return float(L) / g
           
    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x > C} { <g, x> + L * D(x,y) }
        This is a general function that works for all derived classes
        """
        if y.shape != g.shape:
            raise ValueError("BurgEntropy: y and g have different shapes.") 
        if L <= 0:
            raise ValueError("BurgEntropy: L must be positive.")
        _require_finite_positive(y, "BurgEntropy: y")
        return self.prox_map(g - L * self.gradient(y), L)


class BurgEntropyL1(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i]) used in context of solving the problem 
            min_{x > 0} f(x) + lamda * ||x||_1 
    """
    def __init__(self, lamda=0, x_max=1e4):
        assert lamda >= 0, "BurgEntropyL1: lambda should be nonnegative."
        self.lamda = lamda
        self.x_max = x_max

    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * x.sum()

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { lambda * ||x||_1 + <g, x> + L h(x) }
        !!! This proximal mapping may have unbounded solution x->infty
        """
        if L <= 0:
            raise ValueError("BurgEntropyL1: prox_map only takes positive L.")
        g = _as_float_array(g)
        denom = self.lamda + g
        lower = 1.0 / float(self.x_max)
        if np.any(denom <= 0):
            # Avoid unbounded/invalid iterates instead of returning negative values.
            denom = np.maximum(denom, lower)
        return float(L) / denom

       
class BurgEntropyL2(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i]) used in context of solving the problem 
            min_{x > 0} f(x) + (lambda/2) ||x||_2^2 
    """
    def __init__(self, lamda=0):
        assert lamda >= 0, "BurgEntropyL2: lamda should be nonnegative."
        self.lamda = lamda

    def extra_Psi(self, x):
        """
        return (lamda/2) * ||x||_2^2
        """
        return (self.lamda / 2) * np.dot(x, x)

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { (lamda/2) * ||x||_2^2 + <g, x> + L * h(x) }
        """
        if L <= 0:
            raise ValueError("BurgEntropyL2: prox_map only takes positive L value.")
        g = _as_float_array(g)
        if self.lamda == 0:
            return BurgEntropy.prox_map(self, g, L)
        # Solve lamda*x^2 + g*x - L = 0 using a cancellation-safe formula.
        disc = np.sqrt(g * g + 4.0 * self.lamda * L)
        return (2.0 * L) / (disc + g)

       
class BurgEntropySimplex(BurgEntropy):
    """
    h(x) = - sum_i log(x_i), on the standard simplex
        C = {x >= 0, sum_i x_i = 1}.

    The prox solves
        min_{x in C} <g, x> + L h(x),
    whose solution is
        x_i = 1 / (g_i / L + c),
    with c chosen so that sum_i x_i = 1.
    """

    def __init__(self, eps=1e-12, max_iters=200):
        assert eps > 0, "BurgEntropySimplex: eps should be positive."
        assert max_iters > 0, "max_iters must be positive."
        self.eps = float(eps)
        self.max_iters = int(max_iters)

    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { <g, x> + L h(x) } where C is the unit simplex.
        """
        assert L > 0, "BurgEntropySimplex prox_map only takes positive L."

        gg = _as_float_array(g) / float(L)
        if not np.all(np.isfinite(gg)):
            raise ValueError("BurgEntropySimplex prox_map: non-finite g/L.")

        # Domain: c > -min(gg).
        c_lo = np.nextafter(-np.min(gg), np.inf)

        def root_fun(c):
            return np.sum(1.0 / (gg + c)) - 1.0

        c_hi = max(c_lo + 1.0, 1.0)
        while root_fun(c_hi) > 0.0:
            c_hi *= 2.0

        for _ in range(self.max_iters):
            c = 0.5 * (c_lo + c_hi)
            fc = root_fun(c)
            if abs(fc) <= self.eps:
                break
            if fc > 0.0:
                c_lo = c
            else:
                c_hi = c
            if c_hi - c_lo <= self.eps * max(1.0, abs(c_hi), abs(c_lo)):
                break

        c = 0.5 * (c_lo + c_hi)
        x = 1.0 / (gg + c)
        # Normalize to remove residual bisection error.
        return x / np.sum(x)
       

class ShannonEntropy(LegendreFunction):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    """
    def __init__(self, delta=1e-20):
        self.delta = delta
        
    def __call__(self, x):
        x = _require_finite_positive(x, "ShannonEntropy: x", strict=False)
        return float(np.sum(_xlogx_over_y(x, np.ones_like(x))))

    def gradient(self, x):         
        x = _require_finite_positive(x, "ShannonEntropy: x", strict=False)
        xx = np.maximum(x, self.delta)
        return 1.0 + np.log(xx)

    def divergence(self, x, y):
        x = _require_finite_positive(x, "ShannonEntropy: x", strict=False)
        y = _require_finite_positive(y, "ShannonEntropy: y", strict=True)
        if x.shape != y.shape:
            raise ValueError("ShannonEntropy: x and y have different shapes.")
        return float(np.sum(_xlogx_over_y(x, y) + y - x))        
        
    def prox_map(self, g, L):
        """
        Return argmin_{x >= 0} { <g, x> + L * h(x) }
        """
        if L <= 0:
            raise ValueError("ShannonEntropy prox_map require L > 0.")
        return np.exp(-_as_float_array(g) / float(L) - 1.0)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x >= 0} { <g, x> + L * D(x,y) }
        """
        if y.shape != g.shape:
            raise ValueError("ShannonEntropy: y and g have different shapes.") 
        if L <= 0:
            raise ValueError("ShannonEntropy: L must be positive.")
        _require_finite_positive(y, "ShannonEntropy: y", strict=False)
        return y * np.exp(-_as_float_array(g) / float(L))
   

class ShannonEntropyL1(ShannonEntropy):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    used in the context of  min_{x >=0 } f(x) + lamda * ||x||_1
    """
    def __init__(self, lamda=0, delta=1e-20): 
        ShannonEntropy.__init__(self, delta)
        self.lamda = lamda
        
    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * x.sum()
       
    def prox_map(self, g, L):
        """
        Return argmin_{x >= 0} { lamda * ||x||_1 + <g, x> + L * h(x) }
        """
        return ShannonEntropy.prox_map(self, self.lamda + g, L)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x >= 0} { lamda * ||x||_1 + <g, x> + L * D(x,y) }
        """
        return ShannonEntropy.div_prox_map(self, y, self.lamda + g, L)
   
       
class ShannonEntropySimplex(ShannonEntropy):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    used in the context of  min_{x in C } f(x) where C is standard simplex 
    """
    
    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { <g, x> + L * h(x) } where C is unit simplex
        """
        if L <= 0:
            raise ValueError("ShannonEntropy prox_map require L > 0.")
        a = -_as_float_array(g) / float(L)
        a -= np.max(a)
        x = np.exp(a)
        return x / np.sum(x)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { <g, x> + L*d(x,y) } where C is unit simplex
        """
        if y.shape != g.shape:
            raise ValueError("ShannonEntropySimplex: y and g have different shapes.")
        if L <= 0:
            raise ValueError("ShannonEntropySimplex: L must be positive.")
        y = _require_finite_positive(y, "ShannonEntropySimplex: y")
        a = np.log(y) - _as_float_array(g) / float(L)
        a -= np.max(a)
        x = np.exp(a)
        return x / np.sum(x)
   

class SumOf2nd4thPowers(LegendreFunction):
    """
    h(x) = (1/2)||x||_2^2 + (M/4)||x||_2^4
    """       
    def __init__(self, M):
        self.M = M
    
    def __call__(self, x):
        normsq = np.dot(x, x)
        return 0.5 * normsq + (self.M / 4) * normsq**2

    def gradient(self, x):
        normsq = np.dot(x, x)         
        return (1 + self.M * normsq) * x

    def divergence(self, x, y):
        assert x.shape == y.shape, "Bregman div: x and y not same shape."
        return self.__call__(x) - (self.__call__(y) 
                                   + np.dot(self.gradient(y), x-y))

class SquaredL2Norm(LegendreFunction):
    """
    h(x) = (1/2)||x||_2^2
    """       
    def __call__(self, x):
        return float(0.5*np.dot(x, x))

    def gradient(self, x):         
        return x

    def divergence(self, x, y):
        assert x.shape == y.shape, "SquaredL2Norm: x and y not same shape."
        xy = x - y
        return float(0.5*np.dot(xy, xy))

    def prox_map(self, g, L):
        assert L > 0, "SquaredL2Norm: L should be positive."
        return -(1/L)*g
        
    def div_prox_map(self, y, g, L):
        assert y.shape == g.shape and L > 0, "Vectors y and g not same shape."
        return y - (1/L)*g


class PowerNeg1(LegendreFunction):
    """
    h(x) = 1/x  for x>0
    """       
    def __call__(self, x):
        x = _require_finite_positive(x, "PowerNeg1: x")
        return float(np.sum(1.0 / x))

    def gradient(self, x):         
        x = _require_finite_positive(x, "PowerNeg1: x")
        return -1.0 / (x * x)

    def divergence(self, x, y):
        x = _require_finite_positive(x, "PowerNeg1: x")
        y = _require_finite_positive(y, "PowerNeg1: y")
        if x.shape != y.shape:
            raise ValueError("PowerNeg1: x and y not same shape.")
        xy = x - y
        return float(np.sum(xy * xy / (x * y * y)))

    def prox_map(self, g, L):
        if L <= 0:
            raise ValueError("PowerNeg1: L should be positive.")
        g = _require_finite_positive(g, "PowerNeg1 prox_map: g")
        return np.sqrt(float(L) / g)
        

class L2L1Linf(LegendreFunction):
    """
    usng h(x) = (1/2)||x||_2^2 in solving problems of the form
    
        minimize    f(x) + lamda * ||x||_1
        subject to  ||x||_inf <= B
        
    """       
    def __init__(self, lamda=0, B=1): 
        self.lamda = lamda
        self.B = B
        
    def __call__(self, x):
        return 0.5*np.dot(x, x)

    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * np.sum(abs(x))

    def gradient(self, x):         
        """
        gradient of h(x) = (1/2)||x||_2^2
        """
        return x

    def divergence(self, x, y):
        """
        Bregman divergence D(x, y) = (1/2)||x-y||_2^2
        """
        assert x.shape == y.shape, "L2L1Linf: x and y not same shape."
        xy = x - y
        return 0.5*np.dot(xy, xy)

    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * h(x) }
        """
        assert L > 0, "L2L1Linf: L should be positive."
        x = -(1.0 / L) * np.array(g, dtype=float, copy=True)
        threshold = self.lamda / L
        x[np.abs(x) <= threshold] = 0.0
        x[x > threshold] -= threshold
        x[x < -threshold] += threshold
        np.clip(x, -self.B, self.B, out=x)
        return x
        
    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  } 
        """
        assert y.shape == g.shape and L > 0, "Vectors y and g not same shape."
        return self.prox_map(g - L*y, L)
        