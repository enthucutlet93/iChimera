# inspiral evolved in Mino time and using Julia's base least squares to fit the multipole moments to their Fourier series expansion in order to estimate their high order time derivatives. If use_FDM=true, finite differences are used to compute
# the (lower order) time derivatives of the multipole moments required for waveform generation. If use_FDM=false, these time derivatives are also estimated using Fourier fits.
@time ChimeraInspiral.FourierFit.MinoTime.compute_inspiral(a, p, e, θmin, sign_Lz, q, psi0, chi0, phi0, nPointsFit, nHarm, fit_time_range_factor, compute_fluxes, t_max_M, use_FDM, fit_type, reltol, abstol;
        h=h, data_path=data_path, JIT=true)
@time ChimeraInspiral.FourierFit.MinoTime.compute_inspiral(a, p, e, θmin, sign_Lz, q, psi0, chi0, phi0, nPointsFit, nHarm, fit_time_range_factor, compute_fluxes, t_max_M, use_FDM, fit_type, reltol, abstol;
        h=h, data_path=data_path, JIT=false)