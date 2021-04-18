# Rate with anisotropy (single dot)

This code is for testing how the rate changes when a second anisotropy term is added in

Based on original parameters from "single_nanomagnet_no_input_6.mx3" from Axel-initial-sims.

- "single_nanomagnet_0-kprime.mx3" : the base case. Intrinsic anisotropy defined by a custom field, additional anisotropy defined as 0*Kmain along 45 degrees. Should be effectively the same as Axel's original sim, but defined with custom anisotropy field.
- "single_nanomagnet_x-kprime.mx3" : a series of applied kprimes "x"
- "single_nanomagnet_truer-alpha_0-kprime" : A simulation with no applied anisotropy but a more accurate value of alpha (0.005). Looking to see if this only effects the base rate.
- "single_nanomagnet_92nm_x-kprime.mx3" : a series of applied kprimes with the dot diameter raised from 40 nm to 92 nm (cell size kept constant).
