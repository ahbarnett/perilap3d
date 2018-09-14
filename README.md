# perilap3d: triply periodic electrostatic kernels for general unit cell

version 0.5,  9/13/18

Author: Alex H Barnett

This python/numba library computes the potential and fields at a set
of targets inside a given general unit cell, due to a
triply-periodized set of dipoles in the unit cell, to a requested
accuracy tolerance. The unit cell is general (described by three
lattice vectors), although currently it may not have high aspect
ratio.  Potentials and fields are also available at the sources
themselves, where the self-interaction (_j_=_i_) is excluded from the
sum.  The scheme uses direct near-image sums plus solution of the
empty "discrepancy" BVP via an auxiliary proxy point (fundamental
solution) representation.  It is compatible with the fast multipole
method (FMM), although currently only direct summation is used.

For _N_ sources and targets, where the nonperiodic direct evaluation
cost is _N_<sup>2</sup>, the cost of the periodic evaluation is about
_c_<sup>3</sup>_N_<sup>2</sup> + _CN_, where _c_ is a small constant
around 2, and _C_ is a larger constant around 10<sup>3</sup>, whose
size scales like the square of the number of requested digits of
accuracy.  For _N_=1000, the periodic evaluation is 10x slower than
the non-periodic, at 3 digits of accuracy, and 20x slower at 9 digits.
(If the FMM were used, both of the _N_<sup>2</sup> in the above would
be replaced by _O_(_N_), and there would be ways to replace _c_ by
close to 1.) There is also a precomputation phase with cost growing
like the 6th power of the number of requested digits, which need be
done only once for a given unit cell shape, independent of the source
locations.

### Dependencies

This code has been tested in `python3`. It requires `numpy`, `scipy`, `numba`,
and, if plotting is needed, `matplotlib`.

We recommend the
[Intel Distribution for Python](https://software.intel.com/en-us/distribution-for-python),
in which all of our tests were done.

### Testing and usage

Run `test.py` for a complete test of the library.

### To Do

* add outputs at sources

* fix fat Q case with QR solve

* add charge sources

* pass-fail accuracy test, wider range of unit cells?

* spherical harmonics for aux rep instead?

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
