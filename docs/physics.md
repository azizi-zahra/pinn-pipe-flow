# Physics of Hagen-Poiseuille Pipe Flow

This document provides an accessible yet rigorous guide to the physical principles, governing equations, and mathematical formulation underlying the `pinn-pipe-flow` project. It is written for readers with a foundation in basic calculus who may not have a formal background in fluid mechanics.

---

## 1. Problem Description

The physical system modeled in this project is **steady, laminar flow of an incompressible Newtonian fluid through a straight, horizontal circular pipe** with rigid walls—classically known as **Hagen-Poiseuille flow**.

```
            r = R  (Pipe Wall: No-slip u = 0)
    +-----------------------------------------------+
    |  ->                                       ->  |
    |  --->                                   --->  |
    |  ----->                               ----->  |
r=0 |  ------->   Velocity Profile u(r)   ------->  |  Flow Axis (z) --->
    |  ----->                               ----->  |  dp/dz < 0 (Driving Pressure)
    |  --->                                   --->  |
    |  ->                                       ->  |
    +-----------------------------------------------+
            r = -R (Pipe Wall: No-slip u = 0)
```

The pipe has an inner radius $R$. Fluid enters one end and is driven along the axial coordinate $z$ by a favorable pressure gradient (higher pressure upstream, lower pressure downstream). Because of friction between the fluid and the pipe wall, the flow does not move as a uniform solid plug. Instead, fluid in the middle moves fastest, while fluid adjacent to the wall remains stationary.

---

## 2. Physical Intuition

To understand why the fluid develops a curved velocity profile, consider three key physical concepts: **shear**, **viscosity**, and the **no-slip condition**.

### 2.1 Velocity Variation Across the Pipe

Imagine the fluid in the pipe divided into an infinite number of concentric, cylindrical shells sliding past one another along the pipe axis:

- The outermost shell is in direct contact with the stationary solid pipe wall.
- The inner shells move forward at progressively higher speeds.
- The innermost shell (along the centerline axis $r = 0$) travels the fastest.

Because each shell travels at a slightly different velocity than its neighbors, there is a **velocity gradient** in the radial direction, denoted mathematically as $\frac{du}{dr}$.

### 2.2 The No-Slip Condition

At ordinary macroscopic scales, fluid molecules directly contacting a solid surface adhere to that surface due to intermolecular forces (adhesion). Consequently, there is zero relative velocity between the solid wall and the adjacent fluid layer:

$$u(R) = 0$$

This fundamental empirical law of fluid mechanics is known as the **no-slip condition**. Because the wall does not move, the fluid at $r = R$ cannot move either.

### 2.3 The Role of Viscosity

**Dynamic viscosity** ($\mu$) represents the fluid's internal resistance to deformation or shearing. When two adjacent fluid layers slide past each other at different velocities, viscosity generates an internal friction force between them. 

- A faster-moving fluid layer exerts a forward drag on an adjacent slower layer.
- Conversely, the slower layer exerts an equal and opposite retarding drag on the faster layer.

According to **Newton's Law of Viscosity**, the shear stress $\tau$ (force per unit area parallel to the flow direction) is directly proportional to the radial velocity gradient:

$$\tau = \mu \frac{du}{dr}$$

- Near the wall, where velocity drops sharply to zero, the velocity gradient $|\frac{du}{dr}|$ is largest, producing maximum shear stress.
- At the pipe centerline ($r = 0$), the profile peaks and flattens, so $\frac{du}{dr} = 0$, meaning the shear stress drops to zero.

---

## 3. Forces Acting on the Fluid

Consider a cylindrical fluid element of radius $r$ and length $\Delta z$ centered along the pipe axis. In steady flow, this element is subjected to two competing forces:

```
        Pressure Force (Inlet)                Pressure Force (Outlet)
        P(z) * pi * r^2  ======> [ Fluid ] ======> P(z + Dz) * pi * r^2
                                 [ Element]
                         <====== [        ]
                         Viscous Shear Force
                         tau * (2 * pi * r * Dz)
```

### 3.1 The Pressure Force (Driving Force)

Fluid flows because there is a pressure difference between the pipe entrance and exit:

- At position $z$, pressure is $p(z)$.
- At position $z + \Delta z$, pressure is $p(z + \Delta z) < p(z)$.

The pressure gradient $\frac{dp}{dz}$ is negative, representing a continuous pressure drop along the pipe. The net pressure force acting on the cross-sectional area ($\pi r^2$) of the cylinder is:

$$F_{\text{pressure}} = p(z)\pi r^2 - p(z + \Delta z)\pi r^2 \approx -\frac{dp}{dz} \Delta z (\pi r^2)$$

Because $\frac{dp}{dz} < 0$, this net pressure force is positive and pushes the fluid forward.

### 3.2 The Viscous Force (Retarding Force)

The cylindrical surface of area $A = 2\pi r \Delta z$ experiences a backward drag from the slower fluid surrounding it at radius $r + dr$. This viscous shear force opposes forward motion:

$$F_{\text{viscous}} = \tau (2\pi r \Delta z) = \mu \frac{du}{dr} (2\pi r \Delta z)$$

Because velocity decreases as $r$ increases ($\frac{du}{dr} \le 0$), this force naturally acts in the negative $z$-direction, resisting flow.

### 3.3 Steady-State Force Balance

In **steady flow**, the fluid velocity at any given location does not change with time. Because fluid particles are neither accelerating nor decelerating along the axial direction ($a = 0$), **Newton's Second Law** states that the net force on the fluid element must be identically zero:

$$\sum F_z = F_{\text{pressure}} + F_{\text{viscous}} = 0$$

Equating the driving pressure force to the retarding viscous force yields:

$$-\frac{dp}{dz} \Delta z (\pi r^2) + \mu \frac{du}{dr} (2\pi r \Delta z) = 0$$

Simplifying this algebraic balance reveals that the shear stress must increase linearly with radius:

$$\tau(r) = \mu \frac{du}{dr} = \frac{r}{2} \frac{dp}{dz}$$

---

## 4. The Governing Equation in Cylindrical Coordinates

### 4.1 Derivation from the Navier-Stokes Equations

The general motion of an incompressible Newtonian fluid is governed by the **Navier-Stokes momentum equations**. In cylindrical coordinates $(r, \theta, z)$, where $u_r$, $u_\theta$, and $u_z = u$ represent radial, azimuthal, and axial velocity components, the axial momentum equation is:

$$\rho \left( \frac{\partial u}{\partial t} + u_r \frac{\partial u}{\partial r} + \frac{u_\theta}{r}\frac{\partial u}{\partial \theta} + u\frac{\partial u}{\partial z} \right) = -\frac{\partial p}{\partial z} + \mu \left[ \frac{1}{r}\frac{\partial}{\partial r}\left(r \frac{\partial u}{\partial r}\right) + \frac{1}{r^2}\frac{\partial^2 u}{\partial \theta^2} + \frac{\partial^2 u}{\partial z^2} \right] + \rho g_z$$

Under our physical assumptions (detailed in Section 8):
1. **Steady flow**: $\frac{\partial u}{\partial t} = 0$.
2. **Unidirectional flow**: $u_r = 0$, $u_\theta = 0$.
3. **Axisymmetric**: All derivatives with respect to $\theta$ vanish ($\frac{\partial}{\partial \theta} = 0$).
4. **Fully developed flow**: Velocity does not vary along the length of the pipe ($\frac{\partial u}{\partial z} = 0$).
5. **Horizontal pipe**: Gravity has no component in the axial direction ($g_z = 0$).

All convective acceleration terms on the left-hand side drop out, leaving an exact balance between pressure gradient and viscous diffusion:

$$\frac{dp}{dz} = \frac{\mu}{r}\frac{d}{dr}\left(r \frac{du}{dr}\right)$$

### 4.2 Differential Form and Expansion

Expanding the product rule on the right-hand side gives:

$$\frac{1}{r}\frac{d}{dr}\left(r \frac{du}{dr}\right) = \frac{1}{r}\left(\frac{du}{dr} + r \frac{d^2 u}{dr^2}\right) = \frac{d^2 u}{dr^2} + \frac{1}{r}\frac{du}{dr}$$

Multiplying by dynamic viscosity $\mu$, the governing ordinary differential equation (ODE) is:

$$\mu \left( \frac{d^2 u}{dr^2} + \frac{1}{r}\frac{du}{dr} \right) = \frac{dp}{dz}$$

### 4.3 Physical Meaning of the Terms

| Term | Mathematical Expression | Physical Interpretation |
| :--- | :---: | :--- |
| **Viscous Diffusion** | $\mu \frac{d^2 u}{dr^2}$ | Classical shear curvature; transfers momentum across fluid layers. |
| **Cylindrical Metric Correction** | $\frac{\mu}{r}\frac{du}{dr}$ | Arises because cylindrical shell surface area increases with radius ($2\pi r \Delta z$), requiring more force at larger radii to maintain momentum balance. |
| **Driving Pressure Gradient** | $\frac{dp}{dz}$ | Axial force driving fluid through the pipe (constant throughout the cross-section). |

### 4.4 The PINN Residual Formulation

In a Physics-Informed Neural Network (PINN), the neural network $\hat{u}(r)$ acts as a candidate solution. The differential equation is rearranged into a **PDE residual** that must evaluate to zero everywhere in the pipe interior:

$$\mathcal{R}_{\text{pde}}(r) = \mu \left( \frac{\partial^2 \hat{u}}{\partial r^2} + \frac{1}{r}\frac{\partial \hat{u}}{\partial r} \right) - \frac{dp}{dz}$$

The network's derivatives $\frac{\partial \hat{u}}{\partial r}$ and $\frac{\partial^2 \hat{u}}{\partial r^2}$ are computed automatically using PyTorch autograd (`torch.autograd.grad`).

---

## 5. Boundary Conditions

An ordinary differential equation of second order requires **two boundary conditions** to determine a unique physical solution.

```
                    Centerline (r = 0)                   Wall (r = R)
                         |                                    |
Velocity:          Peak: u(0) = u_max                   Zero: u(R) = 0
Slope (du/dr):     Flat: du/dr = 0                      Steepest slope
Condition Name:    Symmetry Boundary Condition          No-slip Boundary Condition
```

### 5.1 No-Slip at the Wall ($r = R$)

Fluid in direct contact with the pipe wall is stationary:

$$u(R) = 0$$

In the PINN loss function, this boundary condition is enforced by penalizing non-zero predicted velocities at $r = R$:

$$\mathcal{L}_{\text{wall}} = \frac{1}{N_{\text{bc}}} \sum_{i=1}^{N_{\text{bc}}} \left( \hat{u}(R) \right)^2$$

### 5.2 Symmetry at the Center ($r = 0$)

The pipe geometry and flow are completely axisymmetric. Therefore, the velocity profile must reach a smooth extremum (maximum) at the center:

$$\left. \frac{du}{dr} \right|_{r=0} = 0$$

Mathematically, this condition is also necessary to prevent the term $\frac{1}{r}\frac{du}{dr}$ in the governing equation from blowing up as $r \to 0$. From L'Hôpital's rule:

$$\lim_{r \to 0} \frac{1}{r}\frac{du}{dr} = \left. \frac{d^2 u}{dr^2} \right|_{r=0}$$

which is finite only when $\left. \frac{du}{dr} \right|_{r=0} = 0$. In the PINN loss function, this is enforced via:

$$\mathcal{L}_{\text{symmetry}} = \frac{1}{N_{\text{bc}}} \sum_{i=1}^{N_{\text{bc}}} \left( \left. \frac{\partial \hat{u}}{\partial r} \right|_{r=0} \right)^2$$

---

## 6. Analytical Solution: The Parabolic Velocity Profile

We can solve the governing ODE analytically by direct integration. Starting from:

$$\frac{d}{dr}\left(r \frac{du}{dr}\right) = \frac{r}{\mu} \frac{dp}{dz}$$

Integrate once with respect to $r$:

$$r \frac{du}{dr} = \frac{r^2}{2\mu} \frac{dp}{dz} + C_1$$

Divide by $r$:

$$\frac{du}{dr} = \frac{r}{2\mu} \frac{dp}{dz} + \frac{C_1}{r}$$

Applying the symmetry condition $\left. \frac{du}{dr} \right|_{r=0} = 0$ forces $C_1 = 0$ (otherwise $\frac{C_1}{r} \to \infty$ at the center).

Integrate a second time:

$$u(r) = \frac{r^2}{4\mu} \frac{dp}{dz} + C_2$$

Now apply the wall no-slip condition $u(R) = 0$:

$$0 = \frac{R^2}{4\mu} \frac{dp}{dz} + C_2 \implies C_2 = -\frac{R^2}{4\mu} \frac{dp}{dz}$$

Substituting $C_2$ back into the equation:

$$u(r) = -\frac{1}{4\mu} \frac{dp}{dz} \left( R^2 - r^2 \right) = -\frac{R^2}{4\mu} \frac{dp}{dz} \left( 1 - \frac{r^2}{R^2} \right)$$

### 6.1 Relation to Centerline Velocity ($u_{\max}$)

Evaluating $u(r)$ at the pipe center ($r = 0$) defines the maximum velocity $u_{\max}$:

$$u_{\max} = u(0) = -\frac{R^2}{4\mu} \frac{dp}{dz}$$

*(Note: Because $\frac{dp}{dz} < 0$, $u_{\max} > 0$.)*

Substituting $u_{\max}$ yields the celebrated **Hagen-Poiseuille parabolic velocity profile**:

$$u(r) = u_{\max} \left( 1 - \frac{r^2}{R^2} \right)$$

This exact closed-form solution serves as the analytical benchmark for validating our neural network's accuracy during evaluation.

---

## 7. Why $u_{\max}$ is a Variable Input

In classical numerical simulations (e.g., standard finite difference or finite element solvers), a model is solved for a single specific pressure gradient or flow rate. If $u_{\max}$ changes, the entire simulation must be re-run from scratch.

In this PINN implementation, the neural network takes **two inputs**:

$$\hat{u} = \text{Model}(r, u_{\max})$$

```
Input: [r, u_max] ----> [ PINN Neural Network ] ----> Output: [u_pred]
```

### Why parameterize by $u_{\max}$?

1. **Surrogate Modeling (Meta-Learning)**: Rather than learning a single solution curve $u(r)$, the PINN learns an entire continuous family of solutions parameterized across operating regimes.
2. **Generalization Across Flow Rates**: During training, $u_{\max}$ is uniformly sampled from a specified operating interval (e.g., $u_{\max} \in [0.5, 2.0]$). The network simultaneously learns to satisfy physics across laminar flow conditions.
3. **Instant Inference**: Once trained, the single network evaluates velocity at any coordinate $r$ for any arbitrary flow rate in the trained range in microseconds without re-training.

---

## 8. Summary of Modeling Assumptions

The mathematical formulation holds under the following physical assumptions:

| Assumption | Physical Meaning | Justification / Applicability |
| :--- | :--- | :--- |
| **Steady Flow** | $\frac{\partial}{\partial t} = 0$ | The driving pressure gradient and flow rate do not fluctuate over time. |
| **Laminar Flow** | Reynolds number $\text{Re} = \frac{\rho \bar{u} (2R)}{\mu} < 2300$ | Fluid moves smoothly in parallel concentric layers without turbulent eddies or chaotic mixing. |
| **Incompressible** | Density $\rho = \text{constant}$ | Liquids (like water or oil) and gases at low Mach numbers ($\text{Ma} < 0.3$) have negligible density variations. |
| **Newtonian Fluid** | Shear stress is linear: $\tau = \mu \frac{du}{dr}$ | Viscosity $\mu$ depends only on temperature and pressure, not on shear rate or agitation (valid for air, water, oils). |
| **Axisymmetric Flow** | Derivatives $\frac{\partial}{\partial \theta} = 0$, $u_\theta = 0$ | The pipe is perfectly circular and boundary conditions are radially symmetric. |
| **Fully Developed** | $\frac{\partial u}{\partial z} = 0$ | The observation region is located sufficiently far downstream from the pipe inlet, so entrance effects have decayed. |
| **Rigid, Non-Porous Wall** | Wall at $r=R$ is fixed and impermeable | Fluid cannot penetrate the wall ($u_r(R) = 0$), and the wall does not deform. |
| **Horizontal Pipe** | Body force $g_z = 0$ | Gravity acts perpendicular to the flow direction and is balanced by hydrostatic pressure; it does not accelerate axial flow. |
