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
iota = true; I = !iota;                                 # use iota: cos(ι) = Lz / sqrt(Lz^2 + C) (Eq. 25 of arXiv:1109.0572v2) or I = π/2 - sign(Lz) * θmin (Eq. 1.2 of arXiv:2401.09577v2)

if iota
    θmin = InclinationMappings.iota_to_theta_min(a, p, e, inclination)
    # println("ι = $(inclination) degrees => θmin = $(θmin) radians")
else
    θmin = InclinationMappings.I_to_theta_min(inclination, sign_Lz)
    # println("I = $(inclination) degrees => θmin = $(θmin) radians ")
end

# EMRI duration (seconds)
t_max_secs = (10^-3) * year / 3.                        # seconds
Mass_MBH = 1e6 * Msol;                                  # mass of the MBH — sets evolution time scale 

# initial angle variables
psi0 = float(π);                                        # intial radial angle variable
chi0 = 0.0;                                             # initial polar angle variable
phi0 = 0.0;                                             # initial azimuthal angle variable

# waveform parameters
obs_distance = 1.;
ThetaSource = 0.0;                                      # EMRI system polar orientation in solar system barycenter (SSB) frame
PhiSource = 1.5;                                        # EMRI system azimuthal orientation in SSB frame
ThetaKerr = 0.0;                                        # MBH spin polar orientation in SSB frame
PhiKerr = 1.4;                                          # MBH spin azimuthal orientation in SSB frame

# file paths
results_path = "../Results";
data_path=results_path * "/Data/";
plot_path=results_path * "/Plots/";
mkpath(data_path);
mkpath(plot_path);

# we don't recommend changing the parameters below this line until further testing has been done
############################################# DO NOT CHANGE #############################################

### geodesic solver parameters ###
nPointsGeodesic = 500;
reltol =  1e-14;                                        # relative tolerance for the geodesic solver
abstol =  1e-14;                                        # absolute tolerance for the geodesic solver

### evolution time ###
MtoSecs = Mass_MBH * Grav_Newton / c^3;                 # conversion from t(M) -> t(s)
t_max_M = t_max_secs / MtoSecs;                         # units of M

### fourier fit parameters ###
gsl_fit = "GSL";
nPointsFitGSL = 501;                                    # number of points in each least-squares fit (must be odd)
nHarmGSL = 2;                                           # number of harmonics in the fourier series expansion

julia_fit = "Julia"
nPointsFitJulia = 101;                                  # number of points in the julia fit (must be odd)
nHarmJulia = 3;                                         # number of harmonics in the fourier series expansion

### Boyer-Lindquist fourier fit params ###
t_range_factor_BL = 0.5;                                # determines the "length", Δt, of the time series data used to perform the fits in BL time: Δt = t_range_factor_BL * (2π / min(Ω)), where Ω are the fundamental frequencies

### Mino time fourier fit params ###
t_range_factor_Mino_FF = 0.05;                          # determines the "length", Δλ, of the time series data used to perform the fits in Mino time: Δλ = t_range_factor_Mino_FF * (2π / min(ω)), where ω are the fundamental frequencies

### Mino time finite differences params ###
h=0.001;

### frequency of self-force computation ###
compute_SF_frac = 0.05                                  # determines how often the self-force is to be computed as a fraction of the maximum time period (in BL time and Mino time): Δt_SF = compute_SF_frac * (2π / min(Ω)), where Ω are the fundamental frequencies

EE, LL, QQ, CC = ConstantsOfMotion.compute_ELC(a, p, e, θmin, sign_Lz);
rplus = Kerr.KerrMetric.rplus(a); rminus = Kerr.KerrMetric.rminus(a);

# Mino time frequencies
ω = ConstantsOfMotion.KerrFreqs(a, p, e, θmin, EE, LL, QQ, CC, rplus, rminus);

# BL time frequencies
Ω = ω[1:3]/ω[4]; Ωr, Ωθ, Ωϕ = Ω;

# generic case
if e != 0 && inclination !=0
    compute_fluxes_BL = compute_SF_frac * minimum(@. 2π /Ω);         # in units of M (not seconds)
    compute_fluxes_Mino = compute_SF_frac * minimum(@. 2π /ω[1:3])   # in units of M (not seconds)
# eccentric equatorial
elseif e != 0
    compute_fluxes_BL = compute_SF_frac * minimum(@. 2π /[Ω[1], Ω[3]])
    compute_fluxes_Mino = compute_SF_frac * minimum(@. 2π /[ω[1], ω[3]])
# circular inclined
elseif inclination != 0
    compute_fluxes_BL = compute_SF_frac * minimum(@. 2π /Ω[2:3])
    compute_fluxes_Mino = compute_SF_frac * minimum(@. 2π /ω[2:3])
end

### number of points per geodesic in the FDM approach ###
nPointsFDM = Int(floor(compute_fluxes_Mino / h));