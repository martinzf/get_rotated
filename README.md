# Get rotated
Simple Python 3.11 code to study the inertial motion of spinning tops. WIP

### How to use

The rotational motion is periodic, so long simulations aren't necessary. Solving the equations of motion is quick, but loading the animations takes time. I would therefore recommend short simulation times, on the order of 5s.

### Preview
![alt text]()

### Theory
The inertial rotation of rigid bodies is typically studied in a body frame of principal axes of inertia by solving Euler's equations:

$$I_i \dot{\omega}_i - (I_j - I_k)\omega_j\omega_k = 0$$

with $(i,j,k)$ a cyclic permutation of $(1,2,3)$.

Solutions to these equations for the case of spherically symmetric, axially symmetric and asymmetric bodies are readily available in many classical mechanics textbooks. What usually isn't discussed is the transformation from the body frame to the lab frame.

The position $\vec{r}_i(t) $ of a point $i$ in the lab frame, in terms of the position of the center of mass in the lab frame $\vec{R}(t)$ and the position of that same point in the body frame $\tilde{\vec{r}}_i $, is:

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

This could easily be solved numerically, however it turns out that the problem has an analytic solution!! I've implemented the algorithm outlined in van Zon et al., 2007 in order to calculate the attitude matrix at any time $t$.

### References
[van Zon, Ramses & Schofield, Jeremy. (2007). Numerical implementation of the exact dynamics of free rigid bodies. Journal of Computational Physics. 225. 145-164. 10.1016/j.jcp.2006.11.019.](https://www.researchgate.net/publication/222535012_Numerical_implementation_of_the_exact_dynamics_of_free_rigid_bodies)

### Acknowledgements
Many thanks to my Classical Mechanics teacher [Artemio López](http://jacobi.fis.ucm.es/artemio/UCM/English.html) for helping me understand the problem and showing me the article linked above.