include("../main.jl");

# constants
c = 2.99792458 * 1e8; Grav_Newton = 6.67430 * 1e-11; Msol = (1.988) * 1e30; year = 365 * 24 * 60 * 60;

# (initial) orbital parameters
a = 0.98;                                               # spin
e = 0.6;                                                # eccentricity
q = 1e-5;                                               # mass ratio q
p = 7.0;                                                # semi-latus rectum

# specify your favourite inclination angle (in degrees)
inclination = 57.39;
sign_Lz = inclination < 90.0 ? 1 : -1;                  # prograde versus retrograde orbit
iota = false; I = !iota;                                # use iota: cos(ι) = Lz / sqrt(Lz^2 + C) (Eq. 25 of arXiv:1109.0572v2) or I = π/2 - sign(Lz) * θmin (Eq. 1.2 of arXiv:2401.09577v2)

if iota
    if inclination == 0.0
        θmin = π/2
    else
        θmin = InclinationMappings.iota_to_theta_min(a, p, e, inclination)
    end
    # println("ι = $(inclination) degrees => θmin = $(θmin) radians")
else
    if inclination == 0.0
        θmin = π/2
    else
        θmin = InclinationMappings.I_to_theta_min(inclination, sign_Lz)
    end
    # println("I = $(inclination) degrees => θmin = $(θmin) radians ")
end

# EMRI duration (seconds)
t_max_secs = (10^-3) * year / 3.                        # seconds
Mass_MBH = 1e6 * Msol;                                  # mass of the MBH — sets evolution time scale 

# initial angle variables
psi0 = 0.1;                                        # intial radial angle variable
chi0 = 0.2;                                             # initial polar angle variable
phi0 = 0.3;                                             # initial azimuthal angle variable

# waveform parameters
obs_distance = 1.;
ThetaSource = 0.1;                                      # EMRI system polar orientation in solar system barycenter (SSB) frame
PhiSource = 0.2;                                        # EMRI system azimuthal orientation in SSB frame
ThetaKerr = 0.3;                                        # MBH spin polar orientation in SSB frame
PhiKerr = 0.4;                                          # MBH spin azimuthal orientation in SSB frame

# file paths
results_path = "../Results";
data_path=results_path * "/Data/";
plot_path=results_path * "/Plots/";
mkpath(data_path);
mkpath(plot_path);

# compute multiplicative factor to convert GW strain into SI units
pc = 3.08568025e16; # parsec in meters
length_conversion_factor = Grav_Newton * Mass_MBH / c / c; 
Gpc_M_units = 1.0e9 * pc / length_conversion_factor;
strain_to_SI = q / obs_distance / Gpc_M_units;

############################################# DO NOT CHANGE WITHOUT GOOD REASON #############################################
use_FDM = true                                          # use finite differences to compute the time derivatives of the multipole moments (as opposed to doing additional Fourier fits)
### geodesic solver parameters ###
reltol =  1e-14;                                        # relative tolerance for the geodesic solver
abstol =  1e-14;                                        # absolute tolerance for the geodesic solver

### evolution time ###
MtoSecs = Mass_MBH * Grav_Newton / c^3;                 # conversion from t(M) -> t(s)
t_max_M = t_max_secs / MtoSecs;                         # units of M

### fourier fit parameters ###
fit_type = "Julia"
nPointsFit = 101;                                       # number of points in the multipole moment fit (must be odd)
nHarm = 2;                                              # number of harmonics in the fourier series expansion
fit_time_range_factor = 0.5;                            # determines the "length", Δλ, of the time series data used to perform the fits in Mino time: Δλ = fit_time_range_factor * (2π / min(ω)), where ω are the fundamental frequencies

### Time spacing between points in geodesic solver — fixed due to use of finite differences to compute waveform mulitpole moment derivatives ###
h=0.001;

### frequency of self-force computation ###
compute_SF_frac = 0.01                                  # determines how often the self-force is to be computed as a fraction of the maximum time period: Δt_SF = compute_SF_frac * (2π / min(ω)), where ω are the fundamental frequencies

EE, LL, QQ, CC = ConstantsOfMotion.compute_ELC(a, p, e, θmin, sign_Lz);
rplus = Kerr.KerrMetric.rplus(a); rminus = Kerr.KerrMetric.rminus(a);

# Mino time frequencies
ω = ConstantsOfMotion.KerrFreqs(a, p, e, θmin, EE, LL, QQ, CC, rplus, rminus);

# BL time frequencies
# Ω = ω[1:3]/ω[4]; Ωr, Ωθ, Ωϕ = Ω;

# interval between successive self-force computations

# generic case
if e != 0 && inclination !=0
    compute_fluxes = compute_SF_frac * minimum(@. 2π /ω[1:3])
# eccentric equatorial
elseif e != 0
    compute_fluxes = compute_SF_frac * minimum(@. 2π /[ω[1], ω[3]])
# circular inclined
elseif inclination != 0
    compute_fluxes = compute_SF_frac * minimum(@. 2π /ω[2:3])
end