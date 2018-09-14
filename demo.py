# demo code to evaluate self-interaction field of set of triply-periodic dipoles
# Barnett 9/14/18

from numpy import array,random
import perilap3d as l3p

L = array([[1,0,0],[-.3,1.1,0],[.2,-.4,.9]])    # each row is a lattice vec 
p = l3p.lap3d3p(L)    # make a periodizing object for this lattice
p.precomp(tol=1e-6)   # do expensive precomputation (0.6 sec)

ns = 300              # how many sources
y = (random.rand(ns,3)-1/2).dot(L)    # randomly in unit cell
d = random.randn(ns,3)                # dipole strength vectors
pot,grad = p.eval(y,d)                # grad contains negatives of E field

# note that the first time eval is run, numba jit compilation may take a
# few seconds. After this, run time for the above eval call is 11 ms on i7.
