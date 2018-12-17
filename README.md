# perilap3d: triply periodic electrostatic kernels for general unit cell

version 0.6,  9/14/18

Author: Alex H Barnett

![perilap3d demo for self-interaction of 1000 sources (light blue points) in a skew unit cell.
Triply-periodic potential is shown on a slice.
Red points show the auxiliary source points outside the unit cell,
dark blue the surface collocation points, and the six faces are named
(with normals in light blue).](perilap3d2cut.png)

This python/numba library computes the potential and fields at a set
of targets inside a given general unit cell, due to a
triply-periodized set of dipoles in the unit cell, to a requested
accuracy tolerance.
The applications include molecular dynamics and density functional theory in crystals.
The parallelepiped unit cell is general (described by three
lattice vectors), although currently it may not have high aspect
ratio.  Potentials and fields are also available at the sources
themselves, where the self-interaction (_j_=_i_) is excluded from the
sum.  The scheme uses direct near-image sums plus solution of the
empty "discrepancy" BVP via an auxiliary proxy point (method of fundamental
solutions or MFS) representation.  It is compatible with the fast multipole
method (FMM), although currently only direct summation is used.

For _N_ sources and targets, where the nonperiodic direct evaluation
cost is _N_<sup>2</sup>,
the cost of the periodic evaluation is about
_c_<sup>3</sup>_N_<sup>2</sup> + O(_p_<sup>2</sup>_N_),
where _c_ is a small
constant around 2, and _p_ scales linearly with the number of digits
required. Typically _p_=8 to 16, and the second term is around 
10<sup>3</sup>_N_.
For _N_=1000, the periodic evaluation is 10x slower than
the non-periodic,at 3 digits of accuracy, and 20x slower at 9 digits.
(If the FMM were used, both of the _N_<sup>2</sup> in the above would
be replaced by _O_(_N_), and there would be ways to replace _c_ by
close to 1.) There is also a precomputation phase with cost
O(_p_<sup>6</sup>), which need be
done only once for a given unit cell shape, independent of the source
or target locations.

### Dependencies

This code has been tested in `python3`. It requires `numpy`, `scipy`, `numba`,
and, if plotting is needed, `matplotlib`.

We recommend the
[Intel Distribution for Python](https://software.intel.com/en-us/distribution-for-python),
in which all of our tests were done.

### Testing and usage

Run `test.py` for a complete test of the library.
It will produce output similar to the file `test.out`, which has timings
for an i7 laptop.

Here is a simple example (see `demo.py`):
```
from numpy import array,random
import perilap3d as l3p

L = array([[1,0,0],[-.3,1.1,0],[.2,-.4,.9]])    # each row is a lattice vec 
p = l3p.lap3d3p(L)    # make a periodizing object for this lattice
p.precomp(tol=1e-6)   # do expensive precomputation (0.6 sec)

ns = 300                              # how many sources
y = (random.rand(ns,3)-1/2).dot(L)    # randomly in unit cell
d = random.randn(ns,3)                # dipole strength vectors
pot,grad = p.eval(y,d)                # grad contains negatives of E field
```
The first call to `eval` after importing will require a few seconds to jit-compile the numba code. With this done, the above `eval` call takes 11 ms on an i7.

See `perilap3d.py` test codes for more examples.

### To Do

* add charge sources

* spherical harmonics (or better) for aux rep for rapid empty BVP solve

* fix fat Q case with QR solve? (only needed if m<P)

* pass-fail accuracy test, wider range of unit cells?

* way to save the Q factors for later use for that unit cell, for >8 digit acc


### References

* For a reference on a similar scheme in 2D due to the author, combining with the FMM, see:

_A unified integral equation scheme for doubly-periodic Laplace and Stokes boundary value problems in two dimensions_,
Alex H. Barnett, Gary Marple, Shravan Veerapaneni, Lin Zhao,
_in press, Comm. Pure Appl. Math._ (2018).
`http://arxiv.org/abs/1611.08038`

* For a recent use of this idea in 3D cubic unit cells combining with the FMM
(and showing that the extra near-neighbor cost can essentially be removed),
see:

_Flexibly imposing periodicity in kernel independent FMM: A
Multipole-To-Local operator approach_,
Wen Yan and Michael Shelley,
_J. Comput. Phys._ *355*, 214-232 (2018).
`http://arxiv.org/abs/1705.02043`
