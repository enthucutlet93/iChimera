# inspiral evolved in Mino time and using Julia's base least squares to fit the multipole moments to their Fourier series expansion in order to estimate their high order time derivatives. If use_FDM=true, finite differences are used to compute
# the (lower order) time derivatives of the multipole moments required for waveform generation. If use_FDM=false, these time derivatives are also estimated using Fourier fits.
include("params.jl");
@time Chimera.compute_inspiral(emri; JIT = true);
@time Chimera.compute_inspiral(emri; JIT = false);

# load solution
Chimera.load_trajectory(emri);
Chimera.load_constants_of_motion(emri);
Chimera.load_fluxes(emri);

# compute in both frames
emri.frame = "SSB";
@time Chimera.compute_waveform(emri);
Chimera.load_waveform(emri);

emri.frame = "Source";
@time Chimera.compute_waveform(emri);
Chimera.load_waveform(emri);