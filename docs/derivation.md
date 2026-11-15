# How drag shrinks an orbit

This is the physics behind [`src/orbit_decay.py`](../src/orbit_decay.py). It starts from the drag force and ends with the equations the code integrates.

**Symbols**

| Symbol | Meaning | Units |
|---|---|---|
| μ | Earth's gravitational parameter, 3.986 × 10¹⁴ | m³/s² |
| R | Earth's equatorial radius, 6,378 km | m |
| a, e | semi-major axis and eccentricity of the orbit | m, — |
| r, v | distance from Earth's center and speed | m, m/s |
| h | altitude above Earth, h = r − R | m |
| ρ(h) | air density at altitude h | kg/m³ |
| B | drag parameter, B = C_D·A/m (drag coefficient × area / mass) | m²/kg |

## 1. The drag force

An object moving at speed v through air of density ρ feels a drag force opposite to its motion:

$$F_D = \tfrac{1}{2}\,\rho\,v^2\,C_D\,A$$

Dividing by the mass gives the drag acceleration:

$$a_D = \tfrac{1}{2}\,B\,\rho\,v^2, \qquad B = \frac{C_D A}{m}$$

B bundles together everything about the object itself. A light, wide object (large A/m, such as a spent solar panel) has a large B and falls quickly; a dense, compact one has a small B. C_D is about 2.2 for most satellites in thin air. Nobody knows A/m for a piece of debris, so the model estimates B from how fast each orbit is actually sinking (section 6).

## 2. Circular orbits: from energy loss to shrinking

The orbital energy per unit mass depends only on the semi-major axis:

$$E = -\frac{\mu}{2a}$$

Drag does negative work at rate force × velocity, so the energy drops at

$$\frac{dE}{dt} = -a_D\,v = -\tfrac{1}{2}\,B\,\rho\,v^3$$

Differentiating E with respect to time, dE/dt = (μ / 2a²)·da/dt. Setting the two equal:

$$\frac{da}{dt} = -\frac{B\,\rho\,v^3\,a^2}{\mu}$$

In a circular orbit v² = μ/a, so v³a²/μ = √(μa) and

$$\boxed{\;\frac{da}{dt} = -B\,\rho(h)\,\sqrt{\mu a}\;}$$

Two things follow. First, **drag makes the satellite speed up**: a smaller orbit has v = √(μ/a) larger, so losing energy raises the speed. That's why `MEAN_MOTION_DOT` (the orbit getting faster) is the model's most useful input. Second, the rate is proportional to ρ, which grows roughly exponentially as the orbit drops, so decay keeps accelerating. That's why the "current rate continues" rule of thumb overestimates lifetimes by a factor of 4–5.
