# Resolution Testing
This code is for testing how fine the resolution needs to be to stop any issues with edge roughness.

Comparisons should be made to "single_nanomagnet_no_input_6.mx3" from Axel-initial-sims. This is the base line.

## Code files
- "single-nanomagnet_edge-smooth.mx3" : Test out edge smoothing on base code
- "single-nanomagnet_20pix.mx3" : Test out higher res version (no smoothing) 20x20 pixels rather than 10x10
- "singe_nanomagnet_macrospin.mx3" : Testing out single spin (macrospin) version where spin is one pixel and demag is off
