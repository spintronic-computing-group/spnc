# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Superparamagnets Network - Control of magnetization through anistropy

# %% [markdown]
# ## The Stoner-Wolfarth model
#
# The goal is to influence $m_{eq}$ by applying a strain on the magnet, which will change the anisotropy of the system. If the strain induces an anistropy $K_\sigma$ forming an angle $\phi$ with the easy axis, the energy will read:
#
# $$E(\theta) = KV\sin^2{\theta} + K_\sigma V\sin^2{(\theta-\phi)} - \mu_0M_SVH\cos{(\theta-\theta_H)}$$
#
# Which can be wrote:
#
# $$E(\theta) = \tilde{K}V\sin^2{(\theta-\psi)} - \mu_0M_SVH\cos{(\theta-\theta_H)}$$
#
# where
#
# $$\tilde{K} = \sqrt{\left(K+K_\sigma\cos{(2\phi)}\right)^2 +\left(K_\sigma\sin{(2\phi)}\right)^2}$$
#
# $$\psi = \frac{1}{2}\arctan{\left(\frac{K_\sigma\sin{(2\phi)}}{K+K_\sigma\cos{(2\phi)}}\right)}$$
#
# The control paramter is  $K_\sigma$, which influences the angle $\psi$ and $\tilde{K}$.
#
# ## Arrhenius equation
#
# We are facing a two-states system with two barriers ($E^-_{b}$ and $E^+_{b}$) of energy. For the moment, we will write de transition rate like this:
#
# $$\omega = \frac{1}{\tau} = f^-_{0}\exp{\left(\frac{-E^-_{b}}{K_BT}\right)}+f^+_{0}\exp{\left(\frac{-E^+_{b}}{K_BT}\right)}$$
#
# We will fix $f_{0,-}=f_{0,+}=1$. We will call $T_{max}$ the temperature verifying $KV/(k_BT) = 3$. Above $T_{max}$, the Arrhenius law cannot be used anymore and our simulation is incorrect.
#
# ## Dimensionless equations
#
# Let $e = \frac{E}{KV}$, $H_K = \frac{2K}{\mu_0M_S}$, $h=\frac{H}{H_K}$, $k_\sigma=\frac{K_\sigma}{K}$, $\omega'=\frac{\omega}{f_0}$ and $\beta'=\frac{KV}{k_BT}$. The energy of the system now reads:
#
# $$e(\theta) = \tilde{k}\sin^2{(\theta-\psi)} - 2h\cos{(\theta-\theta_H)}$$
#
# where
#
# $$\tilde{k} = \sqrt{\left(1+k_\sigma\cos{(2\phi)}\right)^2 +\left(k_\sigma\sin{(2\phi)}\right)^2}$$
#
# $$\psi = \frac{1}{2}\arctan{\left(\frac{k_\sigma\sin{(2\phi)}}{1+k_\sigma\cos{(2\phi)}}\right)}$$
#
# The Arrhenius law becomes:
#
# $$\omega' = \exp{\left(-e^-_{b}\beta'\right)}+\exp{\left(-e^+_{b}\beta'\right)}$$

# %%
