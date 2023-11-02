# Get rotated

### About
Python 3.11 code to study the inertial (torque free) motion of spinning tops.

### How to use
1. Clone the repository and open its folder from the CLI.
1. Run the command `pip install -r requirements.txt` to install dependencies.
1. Run the command `python main.py` (or `python3 main.py` if both Python 2 and Python 3 are installed on your computer).
1. You will be prompted to input information regarding the simulation. Press enter after answering each prompt.
1. Wait while the animation loads. The programme will open an interactive Matplotlib window.

I opted not to save animations due to how slow Matplotlib's `animation.save()` is. However, the examples folder does have some premade gifs you can check out.

### Preview
![alt text](preview.gif)

### Theory
The inertial rotation of rigid bodies is typically studied in a body frame of principal axes of inertia by solving Euler's equations:

$$I_i \dot{\omega}_i - (I_j - I_k)\omega_j\omega_k = 0$$

with $(i,j,k)$ a cyclic permutation of $(1,2,3)$.

Solutions to these equations for the case of spherically symmetric, axially symmetric and asymmetric bodies are readily available in many classical mechanics textbooks. What usually isn't discussed is the transformation from the body frame to the lab frame.

The position $\vec{r}_i(t)$ of a point $i$ in the lab frame, in terms of the position of the center of mass in the lab frame $\vec{R}(t)$ and the position of that same point in the body frame $\tilde{\vec{r}}_i $, is:

$$\vec{r}_i(t)=\vec{R}(t)+A^\top(t)\tilde{\vec{r}}_i$$

where $A^\top(t)$ (the attitude matrix) transforms vectors from the body frame to the lab frame. 

It then follows that:

$$\vec{v}_i = \vec{V} + \vec{\omega}\times(\vec{r}_i-\vec{R})$$

where $\vec{v}_i = \dot{\vec{r}}_i$, $\vec{V} = \dot{\vec{R}}$ and $\vec{\omega}$ is the body's angular velocity as measured in the lab frame. If we consider the antisymmetric matrix $W(\vec{\omega})$ associated to the linear operator $\vec{\omega}\times$, we then find:

$$\dot{A}^\top A = W(\vec{\omega})$$

We want to obtain an ODE for $A$ in terms of the angular velocity as measured in the body frame, $\tilde{\vec{\omega}}$, which we know how to solve for.

$$\begin{align*}
    A \dot{A}^\top A A^\top &= A W(\vec{\omega}) A^\top \\
    A \dot{A}^\top &= W(\tilde{\vec{\omega}}) \\
    -W(\tilde{\vec{\omega}}) A &= \dot{A}
\end{align*}$$

This matrix ODE could easily be solved numerically, however it turns out that the problem has an analytic solution!! I've implemented the algorithm outlined in van Zon & Schofield, 2007 in order to calculate the $A$ matrix at any time $t$.

### References
[van Zon, Ramses & Schofield, Jeremy. (2007). Numerical implementation of the exact dynamics of free rigid bodies. Journal of Computational Physics. 225. 145-164. 10.1016/j.jcp.2006.11.019.](https://www.researchgate.net/publication/222535012_Numerical_implementation_of_the_exact_dynamics_of_free_rigid_bodies)

### Acknowledgements
Many thanks to my Classical Mechanics professor [Artemio González López](http://jacobi.fis.ucm.es/artemio/UCM/English.html) for helping me understand the problem and directing me to the article linked above.