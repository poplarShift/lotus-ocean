# Other relevant literature:

taylor2008stratification: Stratified bottom Ekman layer

# Equations

General scheme outlined in Ch. 7.

The leap-frog scheme uses the first and third column as the old and new time steps for the mean quantities u, S, and T, respectively.
The middle column is the one from which K (eddy diffusivity/viscosity) is updated. In Ch. 7.1., this K is assumed constant and therefore it has no time index. The equations cannot dynamically take into account that K, too, will change during the time step, because they are governed by empirical formulae instead of a conservation law scheme. In other words, K is not part of the "half-step" (middle column) of the leap-frog scheme that applies to the time increments, but instead will always refer to the previous time step $j-1$.

## Time-dependent Ekman equation

The following repeats Ch. 7.1. for the time-dependent Ekman equations (Eq. 7.2.)

## Boundary conditions

---
bullshit! rewrite this

The terms $D_i$, $E_i$ are therefore ultimately determined by the vertical diffusivity or viscosity. They relate the mean values (u, v, T, S) in different vertical grid cells $z_i$ to each other
by the recursion relations $\theta_i = D_i + E_i\theta_{i+1}$. This recursion relation can be started either at the bottom $z_1$ or the surface boundary ($z_{nz}) of the domain by fixing a value at either boundary.

How are D, E determined? From recursion relations involving A, B, C, which in turn are determined by the vertical diffusivity or viscosity of the respective tracer. To fix a boundary condition of these recursion relations, we need a flux condition, again either at the surface or bottom boundary.

In sum, we need one boundary condition at the surface and one at the bottom.

--- nope
Most convenient are fixing either the flux of $\theta$ at the surface (Ch. 7.2.1) or the value of $\theta$ at the bottom (Ch. 7.2.5.) because they lead to simple conditions. Fixing the bottom flux of $\theta$ usually has little practical value because we are interested in the dynamic response of the water column to a surface forcing. Fixing surface $\theta$ is very well possible (Ch. 7.2.2.) but necessitates jumping through a few extra hoops.
