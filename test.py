import lap3dkernels as l3k
import perilap3d as l3p
import matplotlib.pyplot as pl

# test free-space kernels
l3k.warmup()
l3k.test()

# test triply periodic evaluator
pl.ion()                           # so figs don't interrupt the code
l3p.test_lap3d3p(tol=1e-3,verb=2)  # include pics
l3p.test_lap3d3p(tol=1e-6,verb=1)
l3p.test_lap3d3p(tol=1e-9,verb=1)
pl.ioff()
l3p.show_perislice()
