# demo to compute Madelung constant for NaCl (cubic lattice) to 15 digits, via
# periodic-BC solve approach instead of summation over lattice.
# Barnett 11/17/19

# Note on runtimes:
# If run cold from python, takes 5s CPU time due to numba jit compilation.
# If rerun it once njit done (eg within ipython): 0.2s precomp + 0.003s comp.

from numpy import eye,array,random,prod,pi
import perilap3d as l3p

L = eye(3)                            # unit cube (each row is a lattice vec)
p = l3p.lap3d3p(L)                    # make a periodizer object
p.precomp(tol=1e-12,verb=0)           # do expensive precomputation (0.2 sec)

# now we can periodize any arrangement of (zero-sum) charges & dipoles fast...
y = array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]])               # the eight NaCl atoms in unit cell side-length 4
c = prod(y,1)                         # their charges alternate (-1)^{i+j+k}
y = y/4                               # rescale for side length 1
pot,grad = p.eval(y,None,None,c)      # charges (no dipoles), self only  (3 ms)

# extract Madelung const (note overall pot const is arb for periodic solve)...
v = (pot[1]-pot[0])/2                 # half the pot diff btw Na and Cl locs
v = v*4*pi                            # remove 1/4pi fac from 1/r fund sol
v = v/2                               # rescale as if NaCl on unit lattice
vexact = 1.74756459463318219          # cf Bailey et al 2006, Amer Math Monthly
print("Madelung const = %.15g (error from known: %.3g)"%(v,abs(v-vexact)))

# Note on series summation:
# The infinite sum does not converge if taken using an expanding sphere, but
# does for an expanding cube. This illustrates the subtlety of lattice sums,
# and motivates our periodic BVP approach.
# Due to some nice work in python's mpmath package by F. Johansson, nsum can
# amazingly apply series convergence acceleration methods and get 16 digits
# for the constant, in arbitrary precision arithmetic, in 10s CPU time, via:
#   from mpmath import nsum,inf
#   nsum(lambda i,j,k: (-1)**(i+j+k)/(i**2+j**2+k**2)**0.5,
#        ...     [-inf,inf], [-inf,inf], [-inf,inf], ignore=True)
# Out[13]: mpf('-1.7475645946331821')
