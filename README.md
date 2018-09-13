# perilap3d: triply periodic electrostatic kernels with general unit cell

**Alex H Barnett**

9/13/18

This python library computes the potential and fields at a set of
targets inside a given general unit cell, due to a triply-periodized
set of dipoles in the unit cell. This means the 3D Laplace kernel with
periodic boundary conditions is evaluated.  The unit cell is general
(described by three lattice vectors), although currently it may not
have high aspect ratio.  Potentials and fields are also available at
the sources themselves, where the self-interaction (i=j) is excluded from
the sum.  The scheme is compatible with the fast multipole method,
although currently only direct summation is used.

For _N_ sources and targets, where the direct summation cost is _N_<sup>2</sup>




