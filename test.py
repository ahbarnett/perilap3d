# self-test for perilap3d; includes plots. Takes 10-20 sec to complete.
# Barnett 9/13/18

import lap3dkernels as l3k
import perilap3d as l3p
import matplotlib.pyplot as pl

# test free-space kernels
l3k.warmup()
l3k.test()

# test triply periodic evaluator
pl.ion()                           # so figs don't interrupt the code
l3p.show_perislice()
l3p.test_lap3d3p(tol=1e-3,verb=2)  # include pics
l3p.test_lap3d3p(tol=1e-6)
l3p.test_lap3d3p(tol=1e-9)
l3p.test_lap3d3p(tol=1e-12)

# convergence test for grad at sources
l3p.test_conv_lap3d3p()
