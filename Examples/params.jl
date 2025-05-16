include("../main.jl");

# constants
c = 2.99792458 * 1e8; Grav_Newton = 6.67430 * 1e-11; Msol = (1.988) * 1e30; year = 365 * 24 * 60 * 60;

# (initial) orbital parameters
a = 0.98; # spin of the (massive) black hole
e = 0.6; # eccentricity
mass_ratio = 1e-5; # mass ratio of the small compact object to the central massive black hole
p = 7.0; # semi-latus rectum

inclination = 57.39; # inclination angle (in degrees)
sign_Lz = inclination < 90.0 ? 1 : -1; # sign of z-component of angular momentum: +1 for prograde, -1 for retrograde
inclination_type = "iota"; # type of inclination. "iota": cos(ι) = Lz / sqrt(Lz^2 + C) (Eq. 25 of arXiv:1109.0572v2) or "theta_inc": θ_inc = π/2 - sign(Lz) * θmin (Eq. 1.2 of arXiv:2401.09577v2)

lmax_mass = 4 # maximum mass-type multipole moment l mode to include in the flux and waveform computation with 2 ≤ lmax ≤ 4
lmax_current = 3 # maximum current-type multipole moment l mode to include in the flux and waveform computation with 1 ≤ lmax ≤ 3 (lmax = 1 excludes any current-type moment and only up to l=3 included at this time)

t_max_secs = (10^-3) * year / 3.; # maximum orbit evolution time in seconds
Mass_MBH = 1e6 * Msol; # mass of the massive black hole

dt_save = 5.0; # time interval in seconds between saving data points (e.g., waveform, trajectory, etc.)
save_every = 1000; # save solution to file after every save_every steps

# initial angle variables
psi0 = 0.1; # (initial condition) intial radial angle variable
chi0 = 0.2; # (initial condition) initial polar angle variable
phi0 = 0.3; # (initial condition) initial azimuthal angle variable

# waveform parameters
obs_distance = 1.;  # radial distance to observer
frame = "SSB"; # frame in which the waveform is computed: "SSB" for solar system barycenter or "Source" for source frame
ThetaS = 10.0; # (waveform — SSB) EMRI system polar orientation in solar system barycenter (SSB) frame (degrees)
PhiS = 5.0; # (waveform — SSB) EMRI system azimuthal orientation in SSB frame (degrees)
ThetaK = 6.0; # (waveform — SSB) MBH spin polar orientation in SSB frame (degrees)
PhiK = 8.0; # (waveform — SSB) MBH spin azimuthal orientation in SSB frame (degrees)
ThetaObs = 50.0; # (waveform — source) observer polar orientation in source frame (degrees)
PhiObs = 20.0; # (waveform — source) observer azimuthal orientation in source frame (degrees)

save_traj = true; # save BL trajectory
save_SF = true; # save BL self-force
save_constants = true; # save constants of motion
save_fluxes = true; # save fluxes
save_gamma = true; # save gamma factor

# file paths
results_path = "../Results";
data_path=results_path * "/Data/"; # path for saving output files
mkpath(data_path);

# compute multiplicative factor to convert GW strain into SI units
pc = 3.08568025e16; # parsec in meters
MtoSecs = Mass_MBH * Chimera.Grav_Newton / Chimera.c^3
length_conversion_factor = Grav_Newton * Mass_MBH / c / c; 
Gpc_M_units = 1.0e9 * pc / length_conversion_factor;
strain_to_SI = mass_ratio / obs_distance / Gpc_M_units;

############################################# DO NOT CHANGE WITHOUT GOOD REASON #############################################
inspiral_type = "Analytic"; # type of inspiral: "Fitted" will use Fourier fits and finite difference methods (if use_FDM = true), "Analytic" will not use any Fourier fits or finite difference methods

reltol =  1e-14; # relative tolerance for the geodesic solver
abstol =  1e-14; # absolute tolerance for the geodesic solver
compute_SF_frac = 0.01 # determines how often the self-force is to be computed as a fraction of the maximum time period: Δt_SF = compute_SF_frac * (2π / min(ω)), where ω are the fundamental frequencies

### relevant parameters if inspiral_type = "Fitted" ###
use_FDM = true; # use finite differences to compute derivatives of the multipole moments required for the waveform
fit_type = "Julia";  # fitting method to use: "Julia" uses julia's base backslash operator to solve a linear system while "GSL" used GSL's least squares solver
nPointsFit = 101; # number of points to use in the fits. Must be odd
nHarm = 2; # number of harmonics to use in the fits (controls number of terms in the Fourier expansion)
fit_time_range_factor = 0.5; # determines the "length", Δλ, of the time series data used to perform the fits in Mino time: Δλ = fit_time_range_factor * (2π / min(ω)), where ω are the fundamental frequencies
h=0.001; # time step between points in geodesic solver — fixed due to use of finite differences to compute waveform mulitpole moment derivatives

emri = Chimera.EMRI(a, p, e, inclination, inclination_type, sign_Lz, mass_ratio, lmax_mass, lmax_current, psi0, chi0, phi0, frame, ThetaS,
        PhiS, ThetaK, PhiK, ThetaObs, PhiObs, dt_save, data_path, t_max_secs, Mass_MBH, obs_distance, use_FDM, reltol, abstol, fit_type, nPointsFit, nHarm,
        fit_time_range_factor, h, compute_SF_frac, inspiral_type, save_every, save_traj, save_constants, save_fluxes, save_gamma);