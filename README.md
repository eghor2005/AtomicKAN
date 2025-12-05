## This is a VERY COARSE version and absolutely NOT FULLY TESTED! it's only intended for experiementing! Any discussion and criticism are welcome! Check the issues for more information!

# AtomicKAN
Kolmogorov-Arnold Networks (KAN) using Atomic functions instead of B-splines.

This is inspired by Kolmogorov-Arnold Networks <https://arxiv.org/abs/2404.19756v2>, which uses B-splines to approximate functions. B-splines are poor in performance and not very intuitive to use. I'm trying to replace B-splines with Atomic functions.

The atomic function is solutions with a compact support of the linear functional differential equations with
constant coefficients and linear transforms of the argument. The atomic function theory was created in the 70's
of the 20th century due to the necessity to solve different applied problems, in particular, boundary value problems. One of the reasons for the appearance of atomic functions and some other classes of functions was the
inability to apply such classic approximation tools as algebraic and trigonometric polynomials. V.A. Rvachev
up-function is the most famous and widely applies atomic function.

# Usage
Just copy `AtomicKANLayer.py` to your project and import it.
```python
from AtomicKANLayer import AtomicKANLayer
```

Yo can see the examples of usage in the .ipynb files.