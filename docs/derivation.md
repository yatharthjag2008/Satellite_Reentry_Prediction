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

## 3. An exact lifetime for an exponential atmosphere

Real air density depends on altitude, solar activity, time of day and more. To have something exact to test the code against, take the simplest model, density falling off exponentially with scale height H:

$$\rho(h) = \rho_0\, e^{-(h-h_0)/H}$$

With h = a − R and a₀ = R + h₀, the circular decay equation becomes

$$\frac{da}{dt} = -B\rho_0\sqrt{\mu}\; a^{1/2}\, e^{-(a-a_0)/H}$$

This separates. Moving everything with a to one side and integrating from the starting radius a_i down to the final radius a_f:

$$T = \frac{1}{B\rho_0\sqrt{\mu}} \int_{a_f}^{a_i} a^{-1/2}\, e^{(a-a_0)/H}\, da$$

Substitute a = H u², so da = 2H u du and a^{−1/2} = 1/(u√H):

$$T = \frac{2\sqrt{H}\,e^{-a_0/H}}{B\rho_0\sqrt{\mu}} \int_{u_f}^{u_i} e^{u^2}\, du, \qquad u = \sqrt{a/H}$$

The integral of e^{u²} has no elementary form, but it equals e^{u²}·D(u), where D is the [Dawson function](https://en.wikipedia.org/wiki/Dawson_function) (`scipy.special.dawsn`). So the lifetime is exactly

$$\boxed{\;T = \frac{2\sqrt{H}}{B\rho_0\sqrt{\mu}} \Big[ e^{(a_i-a_0)/H} D\big(\sqrt{a_i/H}\big) - e^{(a_f-a_0)/H} D\big(\sqrt{a_f/H}\big) \Big]\;}$$

Writing it with D keeps every exponent small; the equivalent form with erfi overflows.

**Why the rule of thumb fails.** For large u, D(u) ≈ 1/(2u). Keeping only the first term,

$$T \approx \frac{H}{B\,\rho(h_i)\sqrt{\mu a_i}} = \frac{H}{|da/dt|_i}$$

The lifetime is about **one scale height divided by today's sinking rate**, not the remaining altitude divided by the rate. The rule of thumb uses (h − 120 km)/rate. At 400 km with H ≈ 60 km, that's 280/60 ≈ 5 times too long, which matches its measured 430% error (experiment 001).

## 4. Stretched (eccentric) orbits

Most debris is in slightly stretched orbits, and some rocket bodies are in very stretched ones (perigee 200 km, apogee 30,000 km). There, drag isn't constant around the orbit: it's strongest near perigee, where the air is densest. We need how both a and e change.

**Semi-major axis.** The energy argument of section 2 didn't assume a circle, so it still holds at every point of the orbit:

$$\frac{da}{dt} = -\frac{B\,\rho\,v^3\,a^2}{\mu}, \qquad v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)$$

**Eccentricity.** Use the angular momentum per unit mass, h_ang = √(μa(1−e²)), which gives e² = 1 − h_ang²/(μa). Drag acts along the velocity. Its torque is r × F, and only the part of the velocity perpendicular to r contributes, so

$$\frac{dh_{ang}}{dt} = r\cos\gamma \cdot (-a_D) = -\frac{h_{ang}}{v}\,a_D$$

where γ is the flight-path angle and r cos γ = h_ang/v. Differentiating e² = 1 − h_ang²/(μa):

$$2e\,\frac{de}{dt} = -\frac{2h_{ang}}{\mu a}\frac{dh_{ang}}{dt} + \frac{h_{ang}^2}{\mu a^2}\frac{da}{dt}$$

Substituting both rates, using h_ang²/(μa) = 1 − e² and av²/μ − 1 = 2(a − r)/r, then r = a(1−e²)/(1 + e cos ν) with true anomaly ν, everything simplifies to

$$\boxed{\;\frac{de}{dt} = -B\,\rho\,v\,(e + \cos\nu)\;}$$

**Why the apogee drops first.** Take drag acting only at perigee (ν = 0), where v_p² = μ(1+e)/(a(1−e)). The perigee radius is r_p = a(1−e), so

$$\frac{dr_p}{dt} = (1-e)\frac{da}{dt} - a\frac{de}{dt} = -B\rho\left[\frac{(1-e)\,a^2 v_p^3}{\mu} - a\,v_p(1+e)\right]$$

The first term inside the brackets equals a v_p (1+e) after substituting v_p², so **dr_p/dt = 0**. Drag at perigee lowers the apogee but leaves the perigee where it is. The orbit circularizes first, and only then do both ends sink together. This is why the rule of thumb uses the average of perigee and apogee, and why `apogee_rate_7d` is a useful feature.

## 5. Averaging over one orbit

An orbit takes about 90 minutes, but it shrinks noticeably only over days. So instead of following the satellite around each orbit, the code integrates the **orbit-averaged** rates

$$\left\langle \frac{da}{dt} \right\rangle = \frac{1}{2\pi}\int_0^{2\pi} \frac{da}{dt}\, dM$$

over the mean anomaly M, which advances uniformly in time. It's easiest to integrate over the eccentric anomaly E instead, using Kepler's equation M = E − e sin E, so dM = (1 − e cos E) dE, together with

$$r = a(1 - e\cos E), \qquad \cos\nu = \frac{\cos E - e}{1 - e\cos E}$$

The code evaluates the integrand at evenly spaced values of E. For a smooth periodic function, this simple rule converges very fast.

**Left out on purpose.** The atmosphere rotates with Earth, which changes the air speed seen by the satellite by a few percent, and Earth's equatorial bulge means a fixed distance r is a different altitude at different latitudes. Both effects change the overall drag by a roughly constant factor for each object, so they are absorbed into the fitted value of B.

## 6. Estimating B from the observed decay

B is unknown for almost every object, and even for satellites with a known size and mass it changes as they tumble. But B can be measured from the orbit history. Every rate in sections 2–5 is proportional to B, so write

$$\frac{da}{dt} = B\,g(a, e, \text{space weather}), \qquad g = \text{the orbit-averaged rate with } B = 1$$

Along the *observed* orbits, integrating from the start of a window t₀ gives

$$a(t) - a(t_0) = B \int_{t_0}^{t} g\big(a_{obs}(s), e_{obs}(s), \text{weather}(s)\big)\, ds \equiv B\,G(t)$$

G(t) is computed from the observed orbits and the recorded space weather alone, so no simulation is needed, and the observed a(t) should be a straight line in G with slope B. A least-squares fit over the window

$$B = \frac{\operatorname{cov}(a, G)}{\operatorname{var}(G)}$$

is then a single pass of running sums for every object and day. The semi-major axis comes from the mean motion n in each orbit record through Kepler's third law, a = (μ/n²)^{1/3}.

A negative B means the orbit grew over the window, which drag can't do. That signals a maneuver (or a bad orbit record), and the physics prediction is skipped for that row.

## 7. Predicting forward

From the latest orbit, with the fitted B, the code integrates the averaged equations forward until the apogee sinks to 120 km (section 5). Below that, the rest of the fall takes hours, which is less than the one-day precision of the recorded reentry dates.

The future air density depends on future solar activity, which isn't known at prediction time. The baseline holds today's 81-day solar flux and recent geomagnetic activity fixed, and uses the year-averaged density. Section 2 showed that the time to fall scales as 1/B for a fixed atmosphere, so the uncertainty in B and in the weather forecast are the two main sources of error, and the experiment log measures each.
