#=

    In this file we write a master composite type and associated methods for inspirals using this package.

=#

module Chimera
using ..InclinationMappings
using ..ConstantsOfMotion
using ..Kerr
using ..FittedInspiral
using ..AnalyticInspiral
c::Float64 = 2.99792458 * 1e8 # speed of light
Grav_Newton::Float64 = 6.67430 * 1e-11 # newton's gravitational constant
Msol::Float64 = (1.988) * 1e30 # one solar mass in kg
year::Float64 = 365.0 * 24.0 * 60.0 * 60.0 # seconds in a year

mutable struct EMRI
    a::Float64 # spin of the (massive) black hole
    p::Float64 # semi-latus rectum
    e::Float64 # eccentricity
    inclination::Float64 # inclination angle in DEGREES
    inclination_type::String # type of inclination. "iota": cos(ι) = Lz / sqrt(Lz^2 + C) (Eq. 25 of arXiv:1109.0572v2) or "theta_inc": θ_inc = π/2 - sign(Lz) * θmin (Eq. 1.2 of arXiv:2401.09577v2)
    θmin::Float64 # minimum polar angle (in radians)
    sign_Lz::Int64 # sign of Lz. 1 for prograde, -1 for retrograde
    mass_ratio::Float64 # mass ratio of the small compact object to the central massive black hole
    lmax_mass::Int64 # maximum mass-type multipole moment l mode to include in the flux and waveform computation with 2 ≤ lmax ≤ 4
    lmax_current::Int64 # maximum current-type multipole moment l mode to include in the flux and waveform computation with 1 ≤ lmax ≤ 3 (lmax = 1 excludes any current-type moment and only up to l=3 included at this time)
    psi0::Float64 # (initial condition) intial radial angle variable
    chi0::Float64 # (initial condition) initial polar angle variable
    phi0::Float64 # (initial condition) initial azimuthal angle variable
    frame::String # waveform frame: "SSB" for solar system barycenter or "Source" for source frame
    ThetaS::Float64 # (waveform — SSB frame) EMRI system polar orientation in solar system barycenter (SSB) frame
    PhiS::Float64 # (waveform — SSB frame) EMRI system azimuthal orientation in SSB frame
    ThetaK::Float64 # (waveform — SSB frame) MBH spin polar orientation in SSB frame
    PhiK::Float64 # (waveform — SSB frame) MBH spin azimuthal orientation in SSB frame
    ThetaObs::Float64 # (waveform — source frame) observer polar orientation
    PhiObs::Float64 # (waveform — source frame) observer azimuthal orientation
    path::String # path for saving output files
    T_secs::Float64 # maximum orbit evolution time in seconds
    M::Float64 # mass of the massive black hole
    obs_dist::Float64 # radial distance to observer
    use_FDM::Bool # use finite differences to compute derivatives of the multipole moments required for the waveform
    reltol::Float64 # relative tolerance for the geodesic ODE solver
    abstol::Float64 # absolute tolerance for the geodesic ODE solver
    fit_type::String # fitting method to use: "Julia" uses julia's base backslash operator to solve a linear system while "GSL" used GSL's least squares solver
    nPointsFit::Int64 # number of points to use in the fits. Must be odd
    nHarmonics::Int64 # number of harmonics to use in the fits (controls number of terms in the Fourier expansion)
    fit_time_range_factor::Float64 # determines the "length", Δλ, of the time series data used to perform the fits in Mino time: Δλ = fit_time_range_factor * (2π / min(ω)), where ω are the fundamental frequencies
    h::Float64 # time step between points in geodesic solver — fixed due to use of finite differences to compute waveform mulitpole moment derivatives
    compute_SF_frac::Float64 # determines how often the self-force is to be computed as a fraction of the maximum time period: Δt_SF = compute_SF_frac * (2π / min(ω)), where ω are the fundamental frequencies
    inspiral_type::String # type of inspiral: "Fitted" will use Fourier fits and finite difference methods (if use_FDM = true), "Analytic" will not use any Fourier fits or finite difference methods
    save_traj::Bool # save BL trajectory
    save_SF::Bool # save BL self-force
    save_constants::Bool # save constants of motion
    save_fluxes::Bool # save fluxes
    save_gamma::Bool # save gamma factor
    # checks to make sure that the parameters are valid
    EMRI(a, p, e, inclination, inclination_type, θmin, sign_Lz, mass_ratio, lmax_mass, lmax_current, psi0, chi0, phi0, frame, ThetaS,
        PhiS, ThetaK, PhiK, ThetaObs, PhiObs, path, T_secs, M, obs_dist, use_FDM, reltol, abstol, fit_type, nPointsFit, nHarmonics,
        fit_time_range_factor, h, compute_SF_frac, inspiral_type, save_traj, save_SF, save_constants, save_fluxes, save_gamma) = begin
    if a < 0.0 || a > 1.0
        error("Spin parameter 'a' must be between 0 and 1.")
    elseif p <= 0.0
        error("Semi-latus rectum 'p' must be positive.")
    elseif e < 0.0 || e >= 1.0
        error("Eccentricity 'e' must be between 0 and 1.")
    elseif inclination < 0.0 || inclination > 180.0
        error("Inclination 'inclination' must be between 0 and 180 degrees.")
    elseif inclination_type != "iota" && inclination_type != "theta_inc"
        error("Inclination type 'inclination_type' must be either 'iota' or 'theta_inc'.")
    elseif θmin < 0.0 || θmin > π
        error("Minimum polar angle 'θmin' must be between 0 and π radians.")
    elseif sign_Lz != 1 && sign_Lz != -1
        error("Sign of Lz 'sign_Lz' must be either 1 (prograde) or -1 (retrograde).")
    elseif mass_ratio <= 0.0
        error("Mass ratio 'mass_ratio' must be positive.")
    elseif lmax_mass < 2 || lmax_mass > 4
        error("Maximum mode 'lmax_mass' must be between 2 and 4.")
    elseif lmax_current < 1 || lmax_current > 3
        error("Maximum mode 'lmax_current' must be between 1 and 3.")
    elseif frame != "SSB" && frame != "Source"
        error("Waveform frame 'frame' must be either 'SSB' or 'Source'.")
    elseif ThetaS < 0.0 || ThetaS > 180.0
        error("ThetaS 'ThetaS' must be between 0 and 180 degrees.")
    elseif PhiS < 0.0 || PhiS > 360.0
        error("PhiS 'PhiS' must be between 0 and 360 degrees.")
    elseif ThetaK < 0.0 || ThetaK > 180.0
        error("ThetaK 'ThetaK' must be between 0 and 180 degrees.")
    elseif PhiK < 0.0 || PhiK > 360.0
        error("PhiK 'PhiK' must be between 0 and 360 degrees.")
    elseif ThetaObs < 0.0 || ThetaObs > 180.0
        error("ThetaObs 'ThetaObs' must be between 0 and 180 degrees.")
    elseif PhiObs < 0.0 || PhiObs > 360.0
        error("PhiObs 'PhiObs' must be between 0 and 360 degrees.")
    elseif T_secs <= 0.0
        error("Maximum orbit evolution time 'T_secs' must be positive.")
    elseif M <= 0.0
        error("Mass of the massive black hole 'M' must be positive.")
    elseif obs_dist <= 0.0
        error("Radial distance to observer 'obs_dist' must be positive.")
    elseif reltol <= 0.0
        error("Relative tolerance 'reltol' must be positive.")
    elseif abstol <= 0.0
        error("Absolute tolerance 'abstol' must be positive.")
    elseif fit_type != "Julia" && fit_type != "GSL"
        error("Fit type 'fit_type' must be either 'Julia' or 'GSL'.")
    elseif nPointsFit <= 1 || iseven(nPointsFit)
        error("Number of points for fit 'nPointsFit' must be odd and greater than 1.")
    elseif nHarmonics <= 0
        error("Number of harmonics 'nHarmonics' must be positive.")
    elseif fit_time_range_factor <= 0.0
        error("Fit time range factor 'fit_time_range_factor' must be positive.")
    elseif h <= 0.0
        error("Step size 'h' must be positive.")
    elseif compute_SF_frac <= 0.0
        error("Compute SF fraction 'compute_SF_frac' must be positive.")
    elseif compute_SF_frac <= h
        error("Compute SF fraction 'compute_SF_frac' must be greater than step size 'h', i.e., geodesic time length must be greater than step size h.")
    elseif inspiral_type != "Fitted" && inspiral_type != "Analytic"
        error("Inspiral type 'inspiral_type' must be either 'Fitted' or 'Analytic'.")
    else
        new(a, p, e, inclination, inclination_type, θmin, 
        sign_Lz, mass_ratio, lmax_mass, lmax_current, psi0, chi0, phi0, frame, ThetaS, PhiS, ThetaK, PhiK, ThetaObs, PhiObs, path, T_secs, M, obs_dist,
        use_FDM, reltol, abstol, fit_type, nPointsFit, nHarmonics, fit_time_range_factor, h, compute_SF_frac, inspiral_type,
        save_traj, save_SF, save_constants, save_fluxes, save_gamma)
    end
    end
end

function compute_theta_min(a::Float64, p::Float64, e::Float64, inclination::Float64, inclination_type::String, sign_Lz::Int64)
    if inclination == 0.0
        return π/2
    elseif inclination_type == "iota"
        return InclinationMappings.iota_to_theta_min(a, p, e, inclination)
    elseif inclination_type == "theta_inc"
        return InclinationMappings.theta_inc_to_theta_min(inclination, sign_Lz)
    else
        error("Invalid inclination type. Use 'iota' or 'theta_inc'.")
    end
end

function EMRI(
    a::Float64,
    p::Float64,
    e::Float64,
    inclination::Float64,
    inclination_type::String,
    sign_Lz::Int64,
    mass_ratio::Float64,
    lmax_mass::Int64,
    lmax_current::Int64,
    psi0::Float64,
    chi0::Float64,
    phi0::Float64,
    frame::String,
    ThetaS::Float64,
    PhiS::Float64,
    ThetaK::Float64,
    PhiK::Float64,
    ThetaObs::Float64,
    PhiObs::Float64,
    path::String,
    T_secs::Float64,
    M::Float64,
    obs_dist::Float64,
    use_FDM::Bool,
    reltol::Float64,
    abstol::Float64,
    fit_type::String,
    nPointsFit::Int64,
    nHarmonics::Int64,
    fit_time_range_factor::Float64,
    h::Float64,
    compute_SF_frac::Float64,
    inspiral_type::String,
    save_traj::Bool,
    save_SF::Bool,
    save_constants::Bool,
    save_fluxes::Bool,
    save_gamma::Bool)
    # checks to make sure that the parameters are valid
    if a < 0.0 || a > 1.0
        error("Spin parameter 'a' must be between 0 and 1.")
    elseif p <= 0.0
        error("Semi-latus rectum 'p' must be positive.")
    elseif e < 0.0 || e >= 1.0
        error("Eccentricity 'e' must be between 0 and 1.")
    elseif inclination < 0.0 || inclination > 180.0
        error("Inclination 'inclination' must be between 0 and 180 degrees.")
    elseif inclination_type != "iota" && inclination_type != "theta_inc"
        error("Inclination type 'inclination_type' must be either 'iota' or 'theta_inc'.")
    elseif sign_Lz != 1 && sign_Lz != -1
        error("Sign of Lz 'sign_Lz' must be either 1 (prograde) or -1 (retrograde).")
    elseif mass_ratio <= 0.0
        error("Mass ratio 'mass_ratio' must be positive.")
    elseif lmax_mass < 2 || lmax_mass > 4
        error("Maximum mode 'lmax_mass' must be between 2 and 4.")
    elseif lmax_current < 1 || lmax_current > 3
        error("Maximum mode 'lmax_current' must be between 1 and 3.")
    elseif frame != "SSB" && frame != "Source"
        error("Waveform frame 'frame' must be either 'SSB' or 'Source'.")
    elseif ThetaS < 0.0 || ThetaS > 180.0
        error("ThetaS 'ThetaS' must be between 0 and 180 degrees.")
    elseif PhiS < 0.0 || PhiS > 360.0
        error("PhiS 'PhiS' must be between 0 and 360 degrees.")
    elseif ThetaK < 0.0 || ThetaK > 180.0
        error("ThetaK 'ThetaK' must be between 0 and 180 degrees.")
    elseif PhiK < 0.0 || PhiK > 360.0
        error("PhiK 'PhiK' must be between 0 and 360 degrees.")
    elseif ThetaObs < 0.0 || ThetaObs > 180.0
        error("ThetaObs 'ThetaObs' must be between 0 and 180 degrees.")
    elseif PhiObs < 0.0 || PhiObs > 360.0
        error("PhiObs 'PhiObs' must be between 0 and 360 degrees.")
    elseif T_secs <= 0.0
        error("Maximum orbit evolution time 'T_secs' must be positive.")
    elseif M <= 0.0
        error("Mass of the massive black hole 'M' must be positive.")
    elseif obs_dist <= 0.0
        error("Radial distance to observer 'obs_dist' must be positive.")
    elseif reltol <= 0.0
        error("Relative tolerance 'reltol' must be positive.")
    elseif abstol <= 0.0
        error("Absolute tolerance 'abstol' must be positive.")
    elseif fit_type != "Julia" && fit_type != "GSL"
        error("Fit type 'fit_type' must be either 'Julia' or 'GSL'.")
    elseif nPointsFit <= 1 || iseven(nPointsFit)
        error("Number of points for fit 'nPointsFit' must be odd and greater than 1.")
    elseif nHarmonics <= 0
        error("Number of harmonics 'nHarmonics' must be positive.")
    elseif fit_time_range_factor <= 0.0
        error("Fit time range factor 'fit_time_range_factor' must be positive.")
    elseif h <= 0.0
        error("Step size 'h' must be positive.")
    elseif compute_SF_frac <= 0.0
        error("Compute SF fraction 'compute_SF_frac' must be positive.")
    elseif compute_SF_frac <= h
        error("Compute SF fraction 'compute_SF_frac' must be greater than step size 'h', i.e., geodesic time length must be greater than step size h.")
    elseif inspiral_type != "Fitted" && inspiral_type != "Analytic"
        error("Inspiral type 'inspiral_type' must be either 'Fitted' or 'Analytic'.")
    end
    theta_min = compute_theta_min(a, p, e, inclination, inclination_type, sign_Lz)
    return EMRI(a, p, e, inclination, inclination_type, theta_min, sign_Lz, mass_ratio, lmax_mass, lmax_current, psi0, chi0, phi0, frame, ThetaS,
        PhiS, ThetaK, PhiK, ThetaObs, PhiObs, path, T_secs, M, obs_dist, use_FDM, reltol, abstol, fit_type, nPointsFit, nHarmonics,
        fit_time_range_factor, h, compute_SF_frac, inspiral_type, save_traj, save_SF, save_constants, save_fluxes, save_gamma)
end

function compute_inspiral(emri; JIT::Bool = false)
    a = emri.a
    p = emri.p
    e = emri.e
    θmin = emri.θmin
    sign_Lz = emri.sign_Lz
    inclination = emri.inclination
    Mass_MBH = emri.M
    t_max_secs = emri.T_secs
    compute_SF_frac = emri.compute_SF_frac


    EE, LL, QQ, CC = ConstantsOfMotion.compute_ELC(a, p, e, θmin, sign_Lz);
    rplus = Kerr.KerrMetric.rplus(a); rminus = Kerr.KerrMetric.rminus(a);

    # Mino time frequencies
    ω = ConstantsOfMotion.KerrFreqs(a, p, e, θmin, EE, LL, QQ, CC, rplus, rminus);

    # BL time frequencies
    Ω = ω[1:3]/ω[4]; Ωr, Ωθ, Ωϕ = Ω;

    ### evolution time ###
    MtoSecs = Mass_MBH * Grav_Newton / c^3; # conversion from t(M) -> t(s)
    t_max_M = t_max_secs / MtoSecs; # units of M

    if emri.inspiral_type == "Fitted" # Mino time
        if e != 0.0 && inclination != 0.0
            compute_fluxes = compute_SF_frac * minimum(@. 2π /ω[1:3])
        # eccentric equatorial
        elseif e != 0.0
            compute_fluxes = compute_SF_frac * minimum(@. 2π /[ω[1], ω[3]])
        # circular inclined
        elseif inclination != 0.0
            compute_fluxes = compute_SF_frac * minimum(@. 2π /ω[2:3])
        end
        
        FittedInspiral.compute_inspiral(emri.a, emri.p, emri.e, emri.θmin, emri.sign_Lz, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.nPointsFit, emri.nHarmonics, emri.fit_time_range_factor, compute_fluxes, t_max_M, emri.use_FDM, emri.fit_type, emri.reltol, emri.abstol; h=emri.h, data_path=emri.path, JIT=JIT, lmax_mass=emri.lmax_mass, lmax_current=emri.lmax_current, save_traj=emri.save_traj, save_SF=emri.save_SF, save_constants=emri.save_constants, save_fluxes=emri.save_fluxes, save_gamma=emri.save_gamma)
    elseif emri.inspiral_type == "Analytic" # BL time
        if e != 0.0 && inclination != 0.0
            compute_fluxes = compute_SF_frac * minimum(@. 2π /Ω[1:3])
        # eccentric equatorial
        elseif e != 0.0
            compute_fluxes = compute_SF_frac * minimum(@. 2π /[Ω[1], Ω[3]])
        # circular inclined
        elseif inclination != 0.0
            compute_fluxes = compute_SF_frac * minimum(@. 2π /Ω[2:3])
        end

        nPointsGeodesic = 50; # number of points in geodesic solver
        AnalyticInspiral.BLTime.compute_inspiral(emri.a, emri.p, emri.e, emri.θmin, emri.sign_Lz, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, nPointsGeodesic, compute_fluxes, t_max_M, emri.reltol, emri.abstol; data_path=emri.path, JIT=JIT, lmax_mass=emri.lmax_mass, lmax_current=emri.lmax_current, save_traj=emri.save_traj, save_SF=emri.save_SF, save_constants=emri.save_constants, save_fluxes=emri.save_fluxes, save_gamma=emri.save_gamma)
    end
end

function compute_waveform(emri)
    if emri.inspiral_type == "Fitted" # Mino time
        if emri.frame == "SSB"
            FittedInspiral.compute_waveform(emri.obs_dist, emri.ThetaS, emri.PhiS, emri.ThetaK, emri.PhiK, emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.nHarmonics, emri.fit_time_range_factor, emri.fit_type, emri.lmax_mass, emri.lmax_current, emri.path);
        elseif emri.frame == "Source"
            FittedInspiral.compute_waveform(emri.obs_dist, emri.ThetaObs, emri.PhiObs, emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.nHarmonics, emri.fit_time_range_factor, emri.fit_type, emri.lmax_mass, emri.lmax_current, emri.path);
        end
    elseif emri.inspiral_type == "Analytic" # BL time
        if emri.frame == "SSB"
            AnalyticInspiral.BLTime.compute_waveform(emri.obs_dist, emri.ThetaS, emri.PhiS, emri.ThetaK, emri.PhiK, emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.lmax_mass, emri.lmax_current, emri.path);
        elseif emri.frame == "Source"
            AnalyticInspiral.BLTime.compute_waveform(emri.obs_dist, emri.ThetaObs, emri.PhiObs, emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.lmax_mass, emri.lmax_current, emri.path);
        end
    end
end


function load_trajectory(emri)
    if emri.inspiral_type == "Fitted"
        fname = FittedInspiral.solution_fname(emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.nHarmonics, emri.fit_time_range_factor, emri.fit_type, emri.lmax_mass, emri.lmax_current, emri.path)
        return FittedInspiral.load_trajectory(fname)
    elseif emri.inspiral_type == "Analytic"
        fname = AnalyticInspiral.BLTime.solution_fname(emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.lmax_mass, emri.lmax_current, emri.path)
        return AnalyticInspiral.load_trajectory(fname)
    end
end

function load_constants_of_motion(emri)
    if emri.inspiral_type == "Fitted"
        fname = FittedInspiral.solution_fname(emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.nHarmonics, emri.fit_time_range_factor, emri.fit_type, emri.lmax_mass, emri.lmax_current, emri.path)
        return FittedInspiral.load_constants_of_motion(fname)
    elseif emri.inspiral_type == "Analytic"
        fname = AnalyticInspiral.BLTime.solution_fname(emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.lmax_mass, emri.lmax_current, emri.path)
        return AnalyticInspiral.load_constants_of_motion(fname)
    end
end

function load_fluxes(emri)
    if emri.inspiral_type == "Fitted"
        fname = FittedInspiral.solution_fname(emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.nHarmonics, emri.fit_time_range_factor, emri.fit_type, emri.lmax_mass, emri.lmax_current, emri.path)
        return FittedInspiral.load_fluxes(fname)
    elseif emri.inspiral_type == "Analytic"
        fname = AnalyticInspiral.BLTime.solution_fname(emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.lmax_mass, emri.lmax_current, emri.path)
        return AnalyticInspiral.load_fluxes(fname)
    end
end

function load_waveform(emri)
    if emri.inspiral_type == "Fitted" # Mino time
        if emri.frame == "SSB"
            return FittedInspiral.load_waveform(emri.obs_dist, emri.ThetaS, emri.PhiS, emri.ThetaK, emri.PhiK, emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.nHarmonics, emri.fit_time_range_factor, emri.fit_type, emri.lmax_mass, emri.lmax_current, emri.path);
        elseif emri.frame == "Source"
            return FittedInspiral.load_waveform(emri.obs_dist, emri.ThetaObs, emri.PhiObs, emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.nHarmonics, emri.fit_time_range_factor, emri.fit_type, emri.lmax_mass, emri.lmax_current, emri.path);
        end
    elseif emri.inspiral_type == "Analytic" # BL time
        if emri.frame == "SSB"
            return AnalyticInspiral.BLTime.load_waveform(emri.obs_dist, emri.ThetaS, emri.PhiS, emri.ThetaK, emri.PhiK, emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.lmax_mass, emri.lmax_current, emri.path);
        elseif emri.frame == "Source"
            return AnalyticInspiral.BLTime.load_waveform(emri.obs_dist, emri.ThetaObs, emri.PhiObs, emri.a, emri.p, emri.e, emri.θmin, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, emri.lmax_mass, emri.lmax_current, emri.path);
        end
    end
end

end