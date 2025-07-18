#=
    In this module we write the master functions for computing EMRIs based on the Chimera kludge scheme presented in arXiv:1109.0572v2 (hereafter Ref. [1]), which introduced a local treatment of the self-force approximated using post-Newtonian, post-Minkowskian and
    black hole perturbation theoretic methods, hence the kludge nature of the scheme. In particular, Ref. [1] employs PN and PM expansions to obtain expressions for the regularized metric perturbations in terms of time-asymmetric radiation reaction potentials which are
    functions of the EMRI trajectory. These metric perturbations are then substituted into MiSaTaQuWa equation from black hole perturbation theory to obtain an approximation of the self-force, which is then used to compute radiative fluxes of the constants of motion. 
    
    Schematically, our implementation of the Chimera is as follows (noting that Eq. XX refers to Ref. [1]):

    (1) Numerically evolve the geodesic equation with the initial constants of motion and initial conditions for a time ΔT. (Note that we do not yet consider conservative effects.)
    (2) Compute the multipole moment derivatives required for waveform waveform generation (Eqs. 48-49, 85-86). These are only computed at user specified intervals of coordinate time. They are not used to evolve the inspiral, but are saved for waveform computation outside
        these master functions.

    (3) Compute the multipole moment derivatives required for the self-force computation (Eqs. 48-49). These are computed at the end point of each piecewise geodesic.
    
    (4) Compute the self-acceleration and use this to update the constants of motion.
    
    (5) Evolve the next geodesic using the final point on the previous geodesic as the intial point for the next geodesic, but with the updated constants of motion. Repeat this process until the inspiral has been evolved for the desired amount of time.

    Steps (1-5) are schematic, and, in practice, we adopt two approaches to carry out steps (2-3).

    (i) This approach uses a combination of "Fourier fits" and finite difference formulae to estimate the multipole moment derivatives.
    
        Fourier fits: Yhe multipole moments are orbital functionals and thus possess a Fourier series expansion in terms of the fundamental frequencies of motion (e.g., see Ref. [1] and
        arXiv:astro-ph/0308479). We can compute time series data (from analytic expressions) of the first and second derivatives of the mass and current multipole moments, fit these to their fourier series expansion for the coefficients, and take analytic time derivatives
        of the (truncated) Fourier series to approximate the high order derivatives (see Eqs. 98-99).
        
        For the waveform moments, this is done by taking the physical trajectory from a given piecewise geodesic and using the corresponding time series data in the fit. This physical trajectory is in contrast to a "fictitous" trajectory we use to compute the multipole
        moments for the self-force computation. We wish to compute the self force at the endpoint of each piecewise geodesic. We could do this by fitting to the physical trajectory, but this would place the point of interest at the final
        point in the time series, which is where the fit performs the worst (i.e., at the edges of the time series). So, to get a better fit, we artifically place the point of interest at the center of a fictitous geodesic trajectory and then perform the fit using this time series data.
        This ficticious trajectory is constructed by taking the point of interest and evolving the geodesic equation into the past and future of this point to obtain a time series with the point of interest at its center (this approach was taken by the authors of Ref. [1] in their
        numerical implementation). This fictitious trajectory is only used to compute the self-force and is discarded thereafter. Once these steps have been carried out to approximate the relevant multipole derivatives at the point of interest, one can then compute the self-force
        there, update the constants of motion and continue in the inspiral evolution.

        We provide two options for carrying out this fitting procedure. The first uses GSL's multilinear fitting algorithm to fit the data to its fourier series expansion. This method is slow, and gets increasingly slow as one includes more harmonic
        frequencies in the fit (which doesn't even necessarily improve the fit). For this option, we recommend using N=2 harmonics, for which we have found the sixth time derivative of test orbital functionals (e.g., r cos(θ) sin (θ)) to be accurate to 1 part in 
        10^{5} (which is consistent with that found in [1]). The second method for implementing these fits useσ Julia's base least squares algorithm, which we have generally found to be faster, and which can be tuned to give more accurate fits. For test orbital
        functionals, we found an accuracy in the sixth time derivative of 1 part in 10^{7} for N=3 harmonics and N=200 points in the time series.

        Finite differences: Finite differences as implemented in coordinate time leads to catastrophic cancellations in the high order derivative estimation (this was also reported in Ref. [1]).
        However, when applying finite difference formulae in Mino time, we have not observed the same numerical instabilities in the testing we have done so far.
        
        Our first numerical algorithm consists of the following: estimate only the multipole moment derivatives required for the self-force computation using a least-squares fitting algorithm, and estimate those required for the waveform using finite differences. To compute the radiation reaction fluxes, we must
        numerically take up to six more time derivatives of time series data of the multipole moments. However, to generate the waveform, we need only take up to two more time derivatives of the multipole moments. Thus, we can reliably use finite differences
        to estimate these derivatives, while using the more robust fourier fitting to compute the fluxes. This leads to a significant speed up because, in total, there are 22 independent components of the multipole moments which must be computed for the self-force,
        while there are 41 components which must be computed for the waveform. Thus, by using finite differences for the waveform moments, we can reduce the number of fits we must perform by a factor of ~2/3 (since we compute the fits for the waveform moments
        and the self-force moments separately).

    (ii) Our second approach is to compute all the derivatives analytically by differentiating the first-order geodesic equations of motions in order to find expressions for high-order derivatives of the Boyer-Lindquist coordinates. We then convert these to derivatives of the harmonic coordinates by differentiating the transformations equations. With these, we can then evaluate the various high-order derivatives of the multipole moments.

    Approach (i) is implemented in this module and approach (ii) is implemented in the "AnalyticInspiral.jl" file.
=#

module FittedInspiral
using LinearAlgebra
using Combinatorics
using StaticArrays
using HDF5
using DifferentialEquations
using JLD2
using Printf
using SciMLBase
import ..HDF5Helper: create_file_group!, create_dataset!, append_data!
using ..SymmetricTensors
using ..Kerr
using ..ConstantsOfMotion
using ..MinoTimeGeodesics
using ..FourierFitGSL
using ..CircularNonEquatorial
import ..HarmonicCoords: g_tt_H, g_tr_H, g_rr_H, g_μν_H, gTT_H, gTR_H, gRR_H, gμν_H
using ..HarmonicCoords
using ..SelfAcceleration
using ..EstimateMultipoleDerivs
using ..EvolveConstants
using ..Waveform
using ..MultipoleFitting

Z_1(a::Float64) = 1 + (1 - a^2 / 1.0)^(1/3) * ((1 + a)^(1/3) + (1 - a)^(1/3))
Z_2(a::Float64) = sqrt(3 * a^2 + Z_1(a)^2)
LSO_r(a::Float64) = (3 + Z_2(a) - sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # retrograde LSO
LSO_p(a::Float64) = (3 + Z_2(a) + sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # prograde LSO

function initialize_solution_file!(file::HDF5.File, chunk_size::Int64, lmax_mass::Int64, lmax_current::Int64, save_traj::Bool, save_constants::Bool, save_fluxes::Bool, save_gamma::Bool)
    create_dataset!(file, "", "t", Float64, chunk_size);
    create_dataset!(file, "", "lambda", Float64, chunk_size);

    traj_group_name = "Trajectory"
    create_file_group!(file, traj_group_name);
    
    if save_traj
        create_dataset!(file, traj_group_name, "r", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "theta", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "phi", Float64, chunk_size);
    end

    if save_gamma
        create_dataset!(file, traj_group_name, "Gamma", Float64, chunk_size);
    end

    constants_group_name = "ConstantsOfMotion"
    create_file_group!(file, constants_group_name);
    if save_constants || save_fluxes
        create_dataset!(file, constants_group_name, "t", Float64, chunk_size);
    end

    if save_constants
        create_dataset!(file, constants_group_name, "Energy", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "AngularMomentum", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "CarterConstant", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "AltCarterConstant", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "p", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "eccentricity", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "theta_min", Float64, chunk_size);
    end

    if save_fluxes
        create_dataset!(file, constants_group_name, "Edot", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "Ldot", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "Qdot", Float64, chunk_size);
        create_dataset!(file, constants_group_name, "Cdot", Float64, chunk_size);
    end


    # SF_group_name = "SelfForce"
    # create_file_group!(file, SF_group_name);
    # if save_SF
    #     create_dataset!(file, SF_group_name, "self_acc_BL_t", Float64, 1);
    #     create_dataset!(file, SF_group_name, "self_acc_BL_r", Float64, 1);
    #     create_dataset!(file, SF_group_name, "self_acc_BL_θ", Float64, 1);
    #     create_dataset!(file, SF_group_name, "self_acc_BL_ϕ", Float64, 1);
    #     create_dataset!(file, SF_group_name, "self_acc_Harm_t", Float64, 1);
    #     create_dataset!(file, SF_group_name, "self_acc_Harm_x", Float64, 1);
    #     create_dataset!(file, SF_group_name, "self_acc_Harm_y", Float64, 1);
    #     create_dataset!(file, SF_group_name, "self_acc_Harm_z", Float64, 1);
    # end

    ## INDEPENDENT WAVEFORM MOMENT COMPONENTS ##
    wave_group_name = "WaveformMoments"
    create_file_group!(file, wave_group_name);

    # mass and current quadrupole second time derivs
    if lmax_mass >= 2
        create_dataset!(file, wave_group_name, "Mij11_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mij12_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Mij13_2", Float64, chunk_size);
        create_dataset!(file, wave_group_name, "Mij22_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mij23_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Mij33_2", Float64, chunk_size);
    end

    if lmax_current >= 2
        create_dataset!(file, wave_group_name, "Sij11_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sij12_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Sij13_2", Float64, chunk_size);
        create_dataset!(file, wave_group_name, "Sij22_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sij23_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Sij33_2", Float64, chunk_size);
    end

    # mass and current octupole third time derivs
    if lmax_mass >= 3
        create_dataset!(file, wave_group_name, "Mijk111_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk112_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk122_3", Float64, chunk_size); 
        create_dataset!(file, wave_group_name, "Mijk113_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk133_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk123_3", Float64, chunk_size); 
        create_dataset!(file, wave_group_name, "Mijk222_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk223_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk233_3", Float64, chunk_size); 
        create_dataset!(file, wave_group_name, "Mijk333_3", Float64, chunk_size);
    end

    if lmax_current == 3
        create_dataset!(file, wave_group_name, "Sijk111_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk112_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk122_3", Float64, chunk_size); 
        create_dataset!(file, wave_group_name, "Sijk113_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk133_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk123_3", Float64, chunk_size); 
        create_dataset!(file, wave_group_name, "Sijk222_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk223_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk233_3", Float64, chunk_size); 
        create_dataset!(file, wave_group_name, "Sijk333_3", Float64, chunk_size);
    end

    # mass hexadecapole fourth time deriv
    if lmax_mass == 4
        create_dataset!(file, wave_group_name, "Mijkl1111_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1112_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1122_4", Float64, chunk_size);
        create_dataset!(file, wave_group_name, "Mijkl1222_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1113_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1133_4", Float64, chunk_size);
        create_dataset!(file, wave_group_name, "Mijkl1333_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1123_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1223_4", Float64, chunk_size);
        create_dataset!(file, wave_group_name, "Mijkl1233_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl2222_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl2223_4", Float64, chunk_size);
        create_dataset!(file, wave_group_name, "Mijkl2233_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl2333_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl3333_4", Float64, chunk_size);
    end
    return file
end

@views function save_traj!(file::HDF5.File, chunk_size::Int64, t::Vector{Float64}, λ::Vector{Float64}, r::Vector{Float64}, θ::Vector{Float64}, ϕ::Vector{Float64}, dt_dτ::Vector{Float64}, save_traj::Bool, save_gamma::Bool)
    append_data!(file, "", "t", t[1:chunk_size], chunk_size);
    append_data!(file, "", "lambda", λ[1:chunk_size], chunk_size);

    traj_group_name = "Trajectory"

    if save_traj
        append_data!(file, traj_group_name, "r", r[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "theta", θ[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "phi", ϕ[1:chunk_size], chunk_size);
    end

    if save_gamma
        append_data!(file, traj_group_name, "Gamma", dt_dτ[1:chunk_size], chunk_size);
    end
end

@views function save_moments!(file::HDF5.File, chunk_size::Int64, Mij2::AbstractArray, Sij2::AbstractArray, Mijk3::AbstractArray, Sijk3::AbstractArray, Mijkl4::AbstractArray, lmax_mass::Int64, lmax_current::Int64)

    ## INDEPENDENT WAVEFORM MOMENT COMPONENTS ##
    wave_group_name = "WaveformMoments"
    # mass and current quadrupole second time derivs
    if lmax_mass >= 2
        append_data!(file, wave_group_name, "Mij11_2", Mij2[1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mij12_2", Mij2[1,2][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Mij13_2", Mij2[1,3][1:chunk_size], chunk_size);
        append_data!(file, wave_group_name, "Mij22_2", Mij2[2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mij23_2", Mij2[2,3][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Mij33_2", Mij2[3,3][1:chunk_size], chunk_size);
    end

    if lmax_current >= 2
        append_data!(file, wave_group_name, "Sij11_2", Sij2[1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sij12_2", Sij2[1,2][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Sij13_2", Sij2[1,3][1:chunk_size], chunk_size);
        append_data!(file, wave_group_name, "Sij22_2", Sij2[2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sij23_2", Sij2[2,3][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Sij33_2", Sij2[3,3][1:chunk_size], chunk_size);
    end

    # mass and current octupole third time derivs
    if lmax_mass >= 3
        append_data!(file, wave_group_name, "Mijk111_3", Mijk3[1,1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk112_3", Mijk3[1,1,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk122_3", Mijk3[1,2,2][1:chunk_size], chunk_size); 
        append_data!(file, wave_group_name, "Mijk113_3", Mijk3[1,1,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk133_3", Mijk3[1,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk123_3", Mijk3[1,2,3][1:chunk_size], chunk_size); 
        append_data!(file, wave_group_name, "Mijk222_3", Mijk3[2,2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk223_3", Mijk3[2,2,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk233_3", Mijk3[2,3,3][1:chunk_size], chunk_size); 
        append_data!(file, wave_group_name, "Mijk333_3", Mijk3[3,3,3][1:chunk_size], chunk_size);
    end

    if lmax_current == 3
        append_data!(file, wave_group_name, "Sijk111_3", Sijk3[1,1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk112_3", Sijk3[1,1,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk122_3", Sijk3[1,2,2][1:chunk_size], chunk_size); 
        append_data!(file, wave_group_name, "Sijk113_3", Sijk3[1,1,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk133_3", Sijk3[1,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk123_3", Sijk3[1,2,3][1:chunk_size], chunk_size); 
        append_data!(file, wave_group_name, "Sijk222_3", Sijk3[2,2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk223_3", Sijk3[2,2,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk233_3", Sijk3[2,3,3][1:chunk_size], chunk_size); 
        append_data!(file, wave_group_name, "Sijk333_3", Sijk3[3,3,3][1:chunk_size], chunk_size);
    end

    # mass hexadecapole fourth time deriv
    if lmax_mass == 4
        append_data!(file, wave_group_name, "Mijkl1111_4", Mijkl4[1,1,1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1112_4", Mijkl4[1,1,1,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1122_4", Mijkl4[1,1,2,2][1:chunk_size], chunk_size);
        append_data!(file, wave_group_name, "Mijkl1222_4", Mijkl4[1,2,2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1113_4", Mijkl4[1,1,1,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1133_4", Mijkl4[1,1,3,3][1:chunk_size], chunk_size);
        append_data!(file, wave_group_name, "Mijkl1333_4", Mijkl4[1,3,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1123_4", Mijkl4[1,1,2,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1223_4", Mijkl4[1,2,2,3][1:chunk_size], chunk_size);
        append_data!(file, wave_group_name, "Mijkl1233_4", Mijkl4[1,2,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl2222_4", Mijkl4[2,2,2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl2223_4", Mijkl4[2,2,2,3][1:chunk_size], chunk_size);
        append_data!(file, wave_group_name, "Mijkl2233_4", Mijkl4[2,2,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl2333_4", Mijkl4[2,3,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl3333_4", Mijkl4[3,3,3,3][1:chunk_size], chunk_size);
    end
end

@views function save_constants!(file::HDF5.File, chunk_size::Int64, t::Vector{Float64}, E::Vector{Float64}, dE_dt::Vector{Float64}, L::Vector{Float64}, dL_dt::Vector{Float64}, Q::Vector{Float64}, dQ_dt::Vector{Float64}, C::Vector{Float64}, dC_dt::Vector{Float64}, p::Vector{Float64}, e::Vector{Float64}, θmin::Vector{Float64}, save_constants::Bool, save_fluxes::Bool)
    constants_group_name = "ConstantsOfMotion"
    if save_constants || save_fluxes
        append_data!(file, constants_group_name, "t", t[1:chunk_size], chunk_size);
    end

    if save_constants
        append_data!(file, constants_group_name, "Energy", E[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "AngularMomentum", L[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "CarterConstant", C[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "AltCarterConstant", Q[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "p", p[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "eccentricity", e[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "theta_min", θmin[1:chunk_size], chunk_size);
    end

    if save_fluxes
        append_data!(file, constants_group_name, "Edot", dE_dt[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "Ldot", dL_dt[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "Qdot", dQ_dt[1:chunk_size], chunk_size);
        append_data!(file, constants_group_name, "Cdot", dC_dt[1:chunk_size], chunk_size);
    end
end

function save_self_acceleration!(file::HDF5.File, acc_BL::Vector{Float64}, acc_Harm::Vector{Float64})
    SF_group_name = "SelfForce"
    append_data!(file, SF_group_name, "self_acc_BL_t", acc_BL[1]);
    append_data!(file, SF_group_name, "self_acc_BL_r", acc_BL[2]);
    append_data!(file, SF_group_name, "self_acc_BL_θ", acc_BL[3]);
    append_data!(file, SF_group_name, "self_acc_BL_ϕ", acc_BL[4]);
    append_data!(file, SF_group_name, "self_acc_Harm_t", acc_Harm[1]);
    append_data!(file, SF_group_name, "self_acc_Harm_x", acc_Harm[2]);
    append_data!(file, SF_group_name, "self_acc_Harm_y", acc_Harm[3]);
    append_data!(file, SF_group_name, "self_acc_Harm_z", acc_Harm[4]);
end

function load_waveform_moments(sol_filename::String, lmax_mass::Int64, lmax_current::Int64)
    Mij2 = [Float64[] for i=1:3, j=1:3]
    Sij2 = [Float64[] for i=1:3, j=1:3]
    Mijk3 = [Float64[] for i=1:3, j=1:3, k=1:3]
    Sijk3 = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mijkl4 = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
   
    h5f = h5open(sol_filename, "r")

    # mass and current quadrupole second time derivs
    if lmax_mass >= 2
        Mij2[1,1] = h5f["WaveformMoments/Mij11_2"][:];
        Mij2[1,2] = h5f["WaveformMoments/Mij12_2"][:];
        Mij2[1,3] = h5f["WaveformMoments/Mij13_2"][:];
        Mij2[2,2] = h5f["WaveformMoments/Mij22_2"][:];
        Mij2[2,3] = h5f["WaveformMoments/Mij23_2"][:];
        Mij2[3,3] = h5f["WaveformMoments/Mij33_2"][:];
    end

    if lmax_current >= 2
        Sij2[1,1] = h5f["WaveformMoments/Sij11_2"][:];
        Sij2[1,2] = h5f["WaveformMoments/Sij12_2"][:];
        Sij2[1,3] = h5f["WaveformMoments/Sij13_2"][:];
        Sij2[2,2] = h5f["WaveformMoments/Sij22_2"][:];
        Sij2[2,3] = h5f["WaveformMoments/Sij23_2"][:];
        Sij2[3,3] = h5f["WaveformMoments/Sij33_2"][:];
    else
        Sij2[1,1] = zeros(length(Mij2[1,1]));
        Sij2[1,2] = zeros(length(Mij2[1,2]));
        Sij2[1,3] = zeros(length(Mij2[1,3]));
        Sij2[2,2] = zeros(length(Mij2[2,2]));
        Sij2[2,3] = zeros(length(Mij2[2,3]));
        Sij2[3,3] = zeros(length(Mij2[3,3]));
    end


    # mass and current octupole third time derivs
    if lmax_mass >= 3
        Mijk3[1,1,1] = h5f["WaveformMoments/Mijk111_3"][:];
        Mijk3[1,1,2] = h5f["WaveformMoments/Mijk112_3"][:];
        Mijk3[1,2,2] = h5f["WaveformMoments/Mijk122_3"][:];
        Mijk3[1,1,3] = h5f["WaveformMoments/Mijk113_3"][:];
        Mijk3[1,3,3] = h5f["WaveformMoments/Mijk133_3"][:];
        Mijk3[1,2,3] = h5f["WaveformMoments/Mijk123_3"][:];
        Mijk3[2,2,2] = h5f["WaveformMoments/Mijk222_3"][:];
        Mijk3[2,2,3] = h5f["WaveformMoments/Mijk223_3"][:];
        Mijk3[2,3,3] = h5f["WaveformMoments/Mijk233_3"][:];
        Mijk3[3,3,3] = h5f["WaveformMoments/Mijk333_3"][:];
    else
        Mijk3[1,1,1] = zeros(length(Mij2[1,1]));
        Mijk3[1,1,2] = zeros(length(Mij2[1,2]));
        Mijk3[1,2,2] = zeros(length(Mij2[2,2]));
        Mijk3[1,1,3] = zeros(length(Mij2[1,3]));
        Mijk3[1,3,3] = zeros(length(Mij2[3,3]));
        Mijk3[1,2,3] = zeros(length(Mij2[2,3]));
        Mijk3[2,2,2] = zeros(length(Mij2[2,2]));
        Mijk3[2,2,3] = zeros(length(Mij2[2,3]));
        Mijk3[2,3,3] = zeros(length(Mij2[3,3]));
        Mijk3[3,3,3] = zeros(length(Mij2[3,3]));
    end

    if lmax_current == 3
        Sijk3[1,1,1] = h5f["WaveformMoments/Sijk111_3"][:];
        Sijk3[1,1,2] = h5f["WaveformMoments/Sijk112_3"][:];
        Sijk3[1,2,2] = h5f["WaveformMoments/Sijk122_3"][:];
        Sijk3[1,1,3] = h5f["WaveformMoments/Sijk113_3"][:];
        Sijk3[1,3,3] = h5f["WaveformMoments/Sijk133_3"][:];
        Sijk3[1,2,3] = h5f["WaveformMoments/Sijk123_3"][:];
        Sijk3[2,2,2] = h5f["WaveformMoments/Sijk222_3"][:];
        Sijk3[2,2,3] = h5f["WaveformMoments/Sijk223_3"][:];
        Sijk3[2,3,3] = h5f["WaveformMoments/Sijk233_3"][:];
        Sijk3[3,3,3] = h5f["WaveformMoments/Sijk333_3"][:];
    else
        Sijk3[1,1,1] = zeros(length(Mij2[1,1]));
        Sijk3[1,1,2] = zeros(length(Mij2[1,2]));
        Sijk3[1,2,2] = zeros(length(Mij2[2,2]));
        Sijk3[1,1,3] = zeros(length(Mij2[1,3]));
        Sijk3[1,3,3] = zeros(length(Mij2[3,3]));
        Sijk3[1,2,3] = zeros(length(Mij2[2,3]));
        Sijk3[2,2,2] = zeros(length(Mij2[2,2]));
        Sijk3[2,2,3] = zeros(length(Mij2[2,3]));
        Sijk3[2,3,3] = zeros(length(Mij2[3,3]));
        Sijk3[3,3,3] = zeros(length(Mij2[3,3]));
    end


    # mass hexadecapole fourth time deriv
    if lmax_mass == 4
        Mijkl4[1,1,1,1] = h5f["WaveformMoments/Mijkl1111_4"][:];
        Mijkl4[1,1,1,2] = h5f["WaveformMoments/Mijkl1112_4"][:];
        Mijkl4[1,1,2,2] = h5f["WaveformMoments/Mijkl1122_4"][:];
        Mijkl4[1,2,2,2] = h5f["WaveformMoments/Mijkl1222_4"][:];
        Mijkl4[1,1,1,3] = h5f["WaveformMoments/Mijkl1113_4"][:];
        Mijkl4[1,1,3,3] = h5f["WaveformMoments/Mijkl1133_4"][:];
        Mijkl4[1,3,3,3] = h5f["WaveformMoments/Mijkl1333_4"][:];
        Mijkl4[1,1,2,3] = h5f["WaveformMoments/Mijkl1123_4"][:];
        Mijkl4[1,2,2,3] = h5f["WaveformMoments/Mijkl1223_4"][:];
        Mijkl4[1,2,3,3] = h5f["WaveformMoments/Mijkl1233_4"][:];
        Mijkl4[2,2,2,2] = h5f["WaveformMoments/Mijkl2222_4"][:];
        Mijkl4[2,2,2,3] = h5f["WaveformMoments/Mijkl2223_4"][:];
        Mijkl4[2,2,3,3] = h5f["WaveformMoments/Mijkl2233_4"][:];
        Mijkl4[2,3,3,3] = h5f["WaveformMoments/Mijkl2333_4"][:];
        Mijkl4[3,3,3,3] = h5f["WaveformMoments/Mijkl3333_4"][:];
    else
        Mijkl4[1,1,1,1] = zeros(length(Mij2[1,1]));
        Mijkl4[1,1,1,2] = zeros(length(Mij2[1,2]));
        Mijkl4[1,1,2,2] = zeros(length(Mij2[2,2]));
        Mijkl4[1,2,2,2] = zeros(length(Mij2[2,2]));
        Mijkl4[1,1,1,3] = zeros(length(Mij2[1,3]));
        Mijkl4[1,1,3,3] = zeros(length(Mij2[3,3]));
        Mijkl4[1,3,3,3] = zeros(length(Mij2[3,3]));
        Mijkl4[1,1,2,3] = zeros(length(Mij2[2,3]));
        Mijkl4[1,2,2,3] = zeros(length(Mij2[2,3]));
        Mijkl4[1,2,3,3] = zeros(length(Mij2[3,3]));
        Mijkl4[2,2,2,2] = zeros(length(Mij2[2,2]));
        Mijkl4[2,2,2,3] = zeros(length(Mij2[2,3]));
        Mijkl4[2,2,3,3] = zeros(length(Mij2[3,3]));
        Mijkl4[2,3,3,3] = zeros(length(Mij2[3,3]));
        Mijkl4[3,3,3,3] = zeros(length(Mij2[3,3]));
    end

    t = h5f["t"][:];
    close(h5f)

    # symmetrize 
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij2);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3);
    SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);
    
    return t, Mij2, Mijk3, Mijkl4, Sij2, Sijk3
end

function load_trajectory(sol_filename::String)
    h5f = h5open(sol_filename, "r")
    t = h5f["t"][:]
    λ = h5f["lambda"][:]
    r = h5f["Trajectory/r"][:]
    θ = h5f["Trajectory/theta"][:]
    ϕ = h5f["Trajectory/phi"][:]
    # dr_dt = h5f["Trajectory/r_dot"][:]
    # dθ_dt = h5f["Trajectory/theta_dot"][:]
    # dϕ_dt = h5f["Trajectory/phi_dot"][:]
    # d2r_dt2 = h5f["Trajectory/r_ddot"][:]
    # d2θ_dt2 = h5f["Trajectory/theta_ddot"][:]
    # d2ϕ_dt2 = h5f["Trajectory/phi_ddot"][:]
    # dt_dτ = h5f["Trajectory/Gamma"][:]
    # dt_dλ = h5f["Trajectory/dt_dlambda"][:]
    close(h5f)
    return λ, t, r, θ, ϕ
end

function load_constants_of_motion(sol_filename::String)
    h5f = h5open(sol_filename, "r")
    t = h5f["ConstantsOfMotion/t"][:]
    EE = h5f["ConstantsOfMotion/Energy"][:]
    LL = h5f["ConstantsOfMotion/AngularMomentum"][:]
    CC = h5f["ConstantsOfMotion/CarterConstant"][:]
    QQ = h5f["ConstantsOfMotion/AltCarterConstant"][:]
    pArray = h5f["ConstantsOfMotion/p"][:]
    ecc = h5f["ConstantsOfMotion/eccentricity"][:]
    θminArray = h5f["ConstantsOfMotion/theta_min"][:]
    close(h5f)
    return t, EE, LL, QQ, CC, pArray, ecc, θminArray
end

function load_fluxes(sol_filename::String)
    h5f = h5open(sol_filename, "r")
    t = h5f["ConstantsOfMotion/t"][:]
    Edot = h5f["ConstantsOfMotion/Edot"][:]
    Ldot = h5f["ConstantsOfMotion/Ldot"][:]
    Qdot = h5f["ConstantsOfMotion/Qdot"][:]
    Cdot = h5f["ConstantsOfMotion/Cdot"][:]
    close(h5f)
    return t, Edot, Ldot, Qdot, Cdot
end

"""
    compute_inspiral(args...)

Evolve inspiral with with Mino time parameterization and estimating the high order multipole derivatives using Fourier fits with respect to λ.

- `tInspiral::Float64`: total coordinate time to evolve the inspiral.
- `compute_SF::Float64`: Mino time interval between self-force computations.
- `fit_time_range_factor::Float64`: time range over which to perform the fourier fits as a fraction of the minimum time period associated with the Mino time fundamental frequencies.
- `nPointsFit::Int64`: number of points in each fit.
- `q::Float64`: mass ratio.
- `a::Float64`: black hole spin 0 < a < 1.
- `p::Float64`: initial semi-latus rectum.
- `e::Float64`: initial eccentricity.
- `θmin::Float64`: initial inclination angle.
- `sign_Lz::Int64`: sign of the z-component of the angular momentum (+1 for prograde, -1 for retrograde).
- `psi_0::Float64`: initial radial angle variable.
- `chi_0::Float64`: initial polar angle variable.
- `phi_0::Float64`: initial azimuthal angle.
- `fit::String`: type of fit to perform. Options are "GSL" or "Julia" to use Julia's GSL wrapper for a multilinear fit, or to use Julia's base least squares solver.
- `nHarm::Int64`: number of radial harmonics to include in the fit (see `FourierFitGSL.jl` or `FourierFitJuliaBase.jl`).
- `reltol`: relative tolerance for ODE solver.
- `abstol`: absolute tolerance for ODE solver.
- `h::Float64`: if use_FDM=true, then h is the step size for the ODE solver (and one does not specify nPointsGeodesic).
- `nPointsGeodesic::Int64`: if use_FDM=false, this sets the number of points in the geodesic.
- `JIT::Bool`: dummy run to JIT compile function.
- `data_path::String`: path to save data.
- `lmax_mass::Int64`: maximum mass-type multipole moment l mode to include in the flux and waveform computation with 2 ≤ lmax ≤ 4
- `lmax_current::Int64` maximum current-type multipole moment l mode to include in the flux and waveform computation with 1 ≤ lmax ≤ 3 (lmax = 1 excludes any current-type moment and only up to l=3 included at this time)
- `save_traj::Bool`: whether to save the trajectory data.
- `save_constants::Bool`: whether to save the constants of motion.
- `save_fluxes::Bool`: whether to save the fluxes.
- `save_gamma::Bool`: whether to save the Lorentz factor.
- `dt_save::Float64`: time interval between saving trajectory data.
- `save_every::Int64`: number of points in each chunk of data when saving to file.
"""

function compute_inspiral(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nPointsFit::Int64, nHarm::Int64, fit_time_range_factor::Float64, compute_SF::Float64, tInspiral::Float64, use_FDM::Bool, fit::String, reltol::Float64=1e-14, abstol::Float64=1e-14; h::Float64 = 0.001, nPointsGeodesic::Int64 = 500, data_path::String="Data/", JIT::Bool=false, lmax_mass::Int64, lmax_current::Int64, save_traj::Bool, save_constants::Bool, save_fluxes::Bool, save_gamma::Bool, dt_save::Float64, save_every::Int64=500)
    # println("Line 503")
    if iseven(nPointsFit)
        throw(DomainError(nPointsFit, "nPointsFit must be odd"))
    end

    if use_FDM
        nPointsGeodesic = floor(Int, compute_SF / h) + 1
        save_at_trajectory = h; Δλi=h/10;    # initial time step for geodesic integration
    else
        save_at_trajectory = compute_SF / (nPointsGeodesic - 1); Δλi=save_at_trajectory/10;    # initial time step for geodesic integration
    end

    if JIT
        tInspiral = 20.0 # dummy run for Δt = 20M
    end

    # create solution file
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)
    
    if isfile(sol_filename)
        rm(sol_filename)
    end
    
    file = h5open(sol_filename, "w")

    # second argument is chunk_size. Since each successive geodesic piece overlap at the end of the first and bgeinning of the second, we must manually save this point only once to avoid repeats in the data 
    FittedInspiral.initialize_solution_file!(file, save_every, lmax_mass, lmax_current, save_traj, save_constants, save_fluxes, save_gamma)

    # initialize data arrays for trajectory and multipole moments which will be used for post-processing
    idx_save_1 = 1;
    t_save = dt_save
    lambda = zeros(save_every)
    time = zeros(save_every)
    r = zeros(save_every)
    theta = zeros(save_every)
    phi = zeros(save_every)
    gamma = zeros(save_every)
    Mij2_data = [zeros(save_every) for i=1:3, j=1:3]
    Sij2_data = [zeros(save_every) for i=1:3, j=1:3];
    Mijk3_data = [zeros(save_every) for i=1:3, j=1:3, k=1:3];
    Sijk3_data = [zeros(save_every) for i=1:3, j=1:3, k=1:3];
    Mijkl4_data = [zeros(save_every) for i=1:3, j=1:3, k=1:3, l=1:3];

    # mulitpole arrays used for waveform computation
    wf_geodesic_num_points = 11
    Mij2_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3]
    Mijk2_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3, k=1:3]
    Mijkl2_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3, k=1:3, l=1:3]
    Sij1_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3]
    Sijk1_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3, k=1:3]
    Mijk3_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3, k=1:3];
    Mijkl4_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3, k=1:3, l=1:3];
    Sij2_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3];
    Sijk3_data_wf_temp = [zeros(wf_geodesic_num_points) for i=1:3, j=1:3, k=1:3];

    # arrays for self-force computation
    Mijk2_data_SF= [zeros(nPointsFit) for i=1:3, j=1:3, k=1:3]
    Mij2_data_SF= [zeros(nPointsFit) for i=1:3, j=1:3]
    Sij1_data_SF= [zeros(nPointsFit) for i=1:3, j=1:3]
    Mij5 = zeros(3, 3)
    Mij6 = zeros(3, 3)
    Mij7 = zeros(3, 3)
    Mij8 = zeros(3, 3)
    Mijk7 = zeros(3, 3, 3)
    Mijk8 = zeros(3, 3, 3)
    Sij5 = zeros(3, 3)
    Sij6 = zeros(3, 3)
    aSF_BL_temp = zeros(4)
    aSF_H_temp = zeros(4)

    # data arrays for fitting
    xBL_fit = [zeros(3) for i in 1:nPointsFit]; xBL_wf = [zeros(3) for i in 1:wf_geodesic_num_points];
    vBL_fit = [zeros(3) for i in 1:nPointsFit]; vBL_wf = [zeros(3) for i in 1:wf_geodesic_num_points];
    aBL_fit = [zeros(3) for i in 1:nPointsFit]; aBL_wf = [zeros(3) for i in 1:wf_geodesic_num_points];
    xH_fit = [zeros(3) for i in 1:nPointsFit];  xH_wf = [zeros(3) for i in 1:wf_geodesic_num_points];
    vH_fit = [zeros(3) for i in 1:nPointsFit];  vH_wf = [zeros(3) for i in 1:wf_geodesic_num_points];
    v_fit = zeros(nPointsFit);   v_wf = zeros(wf_geodesic_num_points);
    rH_fit = zeros(nPointsFit);  rH_wf = zeros(wf_geodesic_num_points);
    aH_fit = [zeros(3) for i in 1:nPointsFit];  aH_wf = [zeros(3) for i in 1:wf_geodesic_num_points];

    # compute number of fitting frequencies used in fits to the fourier series expansion of the multipole moments
    if e == 0.0 && θmin == π/2   # circular equatorial
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_1(nHarm);
    elseif e != 0.0 && θmin != π/2   # generic case
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_3(nHarm);
    else   # circular non-equatorial or non-circular equatorial — either way only two non-trivial frequencies
        n_freqs=FourierFitGSL.compute_num_fitting_freqs_2(nHarm);
    end

    # compute apastron
    ra = p / (1 - e);

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi, CCi = ConstantsOfMotion.compute_ELC(a, p, e, θmin, sign_Lz)   

    # store orbital params in arrays
    idx_save_2 = 1
    t_Fluxes = zeros(save_every);
    E_arr = zeros(save_every);
    E_dot_arr = zeros(save_every);
    L_arr = zeros(save_every); 
    L_dot_arr = zeros(save_every);
    C_arr = zeros(save_every);
    C_dot_arr = zeros(save_every);
    Q_arr = zeros(save_every);
    Q_dot_arr = zeros(save_every);
    p_arr = zeros(save_every);
    e_arr = zeros(save_every);
    θmin_arr = zeros(save_every);

    E_t = EEi; 
    dE_dt = 0.;
    L_t = LLi; 
    dL_dt = 0.;
    C_t = CCi;
    dC_dt = 0.;
    Q_t = QQi
    dQ_dt = 0.;
    p_t = p;
    e_t = e;
    θmin_t = θmin;

    rplus = Kerr.KerrMetric.rplus(a); rminus = Kerr.KerrMetric.rminus(a);
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    λ0 = 0.0
    geodesic_ics = @SArray [t0, psi_0, chi_0, phi_0];

    rLSO = FittedInspiral.LSO_p(a)

    use_custom_ics = true; use_specified_params = true;

    geodesic_time_length = compute_SF;
    num_points_geodesic = nPointsGeodesic;

    # initialize fluxes to zero (serves as a flag in the function which evolves the constants of motion)
    dE_dt = 0.0
    dL_dt = 0.0
    dQ_dt = 0.0
    dC_dt = 0.0
    # println("Line 614")
    while tInspiral > t0
        print("Completion: $(round(100 * t0/tInspiral; digits=5))%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t / (1.0 - e_t); rp=p_t / (1.0 + e_t);
        A = 1.0 / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t); p4 = r4 * (1.0 + e_t)    # Above Eq. 96

        # geodesic
        # println("Line 628")
        # λλ, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi, dt_dλλ = MinoTimeGeodesics.compute_kerr_geodesic(a, p_t, e_t, θmin_t, sign_Lz, num_points_geodesic,
        # use_specified_params, geodesic_time_length, Δλi, reltol, abstol; ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)
        sol = MinoTimeGeodesics.compute_kerr_geodesic(a, p_t, e_t, θmin_t, sign_Lz, num_points_geodesic,
        use_specified_params, geodesic_time_length, Δλi, reltol, abstol; ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false, interpolate_sol=false)
        λmin, λmax = sol.t
        tmin, tmax = sol(λmin, idxs = 1), sol(λmax, idxs = 1)

        if t0 == 0.0
            compute_waveform_moments!(sol, t0, λmin, λmax, a, E_t, L_t, Q_t, C_t, p_t, e_t, θmin_t, sign_Lz, q, xBL_wf, vBL_wf, aBL_wf, xH_wf, rH_wf, vH_wf, aH_wf, v_wf, Mij2_data_wf_temp, Mijk2_data_wf_temp, Mijkl2_data_wf_temp, Sij1_data_wf_temp, Sijk1_data_wf_temp, Mijk3_data_wf_temp, Mijkl4_data_wf_temp, Sij2_data_wf_temp, Sijk3_data_wf_temp, wf_geodesic_num_points, h, lmax_mass, lmax_current, Δλi, abstol, reltol, ra, p3, p4, zp, zm, idx_save_1)
            update_waveform_arrays!(idx_save_1, wf_geodesic_num_points, Mij2_data, Sij2_data, Mijk3_data, Sijk3_data, Mijkl4_data, Mij2_data_wf_temp, Sij2_data_wf_temp, Mijk3_data_wf_temp, Sijk3_data_wf_temp, Mijkl4_data_wf_temp)
            update_trajectory_arrays!(t0, idx_save_1, λ0, λmin, λmax, sol, lambda, time, r, theta, phi, gamma, a, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm, save_traj, save_gamma)
            idx_save_1 += 1

            # also save constants of motion
            t_Fluxes[idx_save_2] = t0;
            E_arr[idx_save_2] = E_t;
            E_dot_arr[idx_save_2] = dE_dt;
            L_arr[idx_save_2] = L_t; 
            L_dot_arr[idx_save_2] = dL_dt;
            C_arr[idx_save_2] = C_t;
            C_dot_arr[idx_save_2] = dC_dt;
            Q_arr[idx_save_2] = Q_t;
            Q_dot_arr[idx_save_2] = dQ_dt;
            p_arr[idx_save_2] = p_t;
            e_arr[idx_save_2] = e_t;
            θmin_arr[idx_save_2] = θmin_t;
            idx_save_2 += 1
        end

        # now determining whether or not waveform is to be computed on this geodesic
        while tmin <= t_save <= tmax
            # save trajectory and moments
            if idx_save_1 == save_every + 1
                FittedInspiral.save_traj!(file, save_every, time, lambda, r, theta, phi, gamma, save_traj, save_gamma)
                FittedInspiral.save_moments!(file, save_every, Mij2_data, Sij2_data, Mijk3_data, Sijk3_data, Mijkl4_data, lmax_mass, lmax_current)
                idx_save_1 = 1
                flush(file)
            end

            compute_waveform_moments!(sol, t_save, λmin, λmax, a, E_t, L_t, Q_t, C_t, p_t, e_t, θmin_t, sign_Lz, q, xBL_wf, vBL_wf, aBL_wf, xH_wf, rH_wf, vH_wf, aH_wf, v_wf, Mij2_data_wf_temp, Mijk2_data_wf_temp, Mijkl2_data_wf_temp, Sij1_data_wf_temp, Sijk1_data_wf_temp, Mijk3_data_wf_temp, Mijkl4_data_wf_temp, Sij2_data_wf_temp, Sijk3_data_wf_temp, wf_geodesic_num_points, h, lmax_mass, lmax_current, Δλi, abstol, reltol, ra, p3, p4, zp, zm, idx_save_1)
            update_waveform_arrays!(idx_save_1, wf_geodesic_num_points, Mij2_data, Sij2_data, Mijk3_data, Sijk3_data, Mijkl4_data, Mij2_data_wf_temp, Sij2_data_wf_temp, Mijk3_data_wf_temp, Sijk3_data_wf_temp, Mijkl4_data_wf_temp)
            update_trajectory_arrays!(t_save, idx_save_1, λ0, λmin, λmax, sol, lambda, time, r, theta, phi, gamma, a, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm, save_traj, save_gamma)
            idx_save_1 += 1
            t_save += dt_save
        end

          

        # extract initial conditions for next geodesic.
        t0, PSI0, CHI0, PHI0 = sol(λmax);
        λ0 = λmax + λ0;

        ###### COMPUTE SELF-FORCE ######
        # compute fundamental frequencies
        ω = ConstantsOfMotion.KerrFreqs(a, p_t, e_t, θmin_t, E_t, L_t, Q_t, C_t, rplus, rminus);    # Mino time frequencies
        ωr=ω[1]; ωθ=ω[2]; ωϕ=ω[3];   # mino time frequencies

        #  we want to perform each fit over a set of points which span a physical time range T_fit. In some cases, the frequencies are infinite, and we thus ignore them in our fitting procedure
        if e_t == 0.0 && θmin_t == π/2   # circular equatorial
            ωr = 1e50; ωθ =1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/ωϕ)
        elseif e_t == 0.0   # circular non-equatorial
            ωr = 1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/[ωθ, ωϕ])
        elseif θmin_t == π/2   # non-circular equatorial
            ωθ = 1e50;
            T_Fit = fit_time_range_factor * minimum(@. 2π/[ωr, ωϕ])
        else   # generic case
            T_Fit = fit_time_range_factor * minimum(@. 2π/ω[1:3])
        end

        saveat_fit = T_Fit / (nPointsFit-1);    # the user specifies the number of points in each fit, i.e., the resolution, which determines at which points the interpolator should save data points
        Δλi_fit = saveat_fit;
        # compute geodesic into future and past of the final point in the (physical) piecewise geodesic computed above
        geodesic_ics = @SArray [t0, PSI0, CHI0, PHI0];
        
        λλ_fit, tt_fit, rr_fit, θθ_fit, ϕϕ_fit, r_dot_fit, θ_dot_fit, ϕ_dot_fit, r_ddot_fit, θ_ddot_fit, ϕ_ddot_fit, Γ_fit, psi_fit, chi_fit, dt_dλ_fit = 
        MinoTimeGeodesics.compute_kerr_geodesic_past_and_future(a, p_t, e_t, θmin_t, sign_Lz, use_specified_params, nPointsFit, T_Fit, Δλi_fit, reltol, abstol; ics=geodesic_ics,
        E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)

        compute_at=(nPointsFit÷2)+1;   # by construction, the end point of the physical geoodesic is at the center of the geodesic computed for the fit
        # check that that the midpoint of the fit geodesic arrays are equal to the final point of the physical arrays
        λ_val = λmax
        t_val, rr, θθ, ϕϕ, dt_dτ, r_dot, θ_dot, ϕ_dot = compute_geodesic_arrays(sol, λ_val, a, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)
        if tt_fit[compute_at] != t_val || rr_fit[compute_at] != rr || θθ_fit[compute_at] != θθ || ϕϕ_fit[compute_at] != ϕϕ ||
            r_dot_fit[compute_at] != r_dot || θ_dot_fit[compute_at] != θ_dot || ϕ_dot_fit[compute_at] != ϕ_dot
            # || r_ddot_fit[compute_at] != last(r_ddot)|| θ_ddot_fit[compute_at] != last(θ_ddot)|| ϕ_ddot_fit[compute_at] != last(ϕ_ddot) ||
            # Γ_fit[compute_at] != last(Γ) || psi_fit[compute_at] != last(psi) || chi_fit[compute_at] != last(chi)
            println("Integration terminated at t = $(first(tt)). Reason: midpoint of fit geodesic does not align with final point of physical geodesic")
            break
        end

        chisq=[0.0];
        SelfAcceleration.FourierFit.selfAcc!(aSF_H_temp, aSF_BL_temp, xBL_fit, vBL_fit, aBL_fit, xH_fit, rH_fit, vH_fit, aH_fit, v_fit, λλ_fit,
            rr_fit, r_dot_fit, r_ddot_fit, θθ_fit, θ_dot_fit, θ_ddot_fit, ϕϕ_fit, ϕ_dot_fit, ϕ_ddot_fit, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6,
            Mij2_data_SF, Mijk2_data_SF, Sij1_data_SF, a, q, E_t, L_t, C_t, compute_at, nHarm, ωr, ωθ, ωϕ, nPointsFit, n_freqs, chisq, fit, lmax_mass, lmax_current);
        

        # update orbital constants and fluxes — function takes as argument the fluxes computed at the end of the previous geodesic (which overlaps with the start of the current geodesic piece) in order to update the fluxes using the trapezium rule
        Δt = t_val - sol(λmin, idxs = 1);
        E_1, dE_dt, L_1, dL_dt, Q_1, dQ_dt, C_1, dC_dt, p_1, e_1, θmin_1 = EvolveConstants.Evolve_BL(Δt, a, rr, θθ, ϕϕ, dt_dτ, r_dot, θ_dot, ϕ_dot, aSF_BL_temp, E_t, dE_dt, L_t, dL_dt, Q_t, dQ_dt, C_t, dC_dt, p_t, e_t, θmin_t)
        

        E_t = E_1; L_t = L_1; Q_t = Q_1; C_t = C_1; p_t = p_1; e_t = e_1; θmin_t = θmin_1;
        # flush(file)

        # save constants of motion
        t_Fluxes[idx_save_2] = t_val;
        E_arr[idx_save_2] = E_t;
        E_dot_arr[idx_save_2] = dE_dt;
        L_arr[idx_save_2] = L_t; 
        L_dot_arr[idx_save_2] = dL_dt;
        C_arr[idx_save_2] = C_t;
        C_dot_arr[idx_save_2] = dC_dt;
        Q_arr[idx_save_2] = Q_t;
        Q_dot_arr[idx_save_2] = dQ_dt;
        p_arr[idx_save_2] = p_t;
        e_arr[idx_save_2] = e_t;
        θmin_arr[idx_save_2] = θmin_t;
        idx_save_2 += 1

        # save constants and fluxes
        if idx_save_2 == save_every + 1
            FittedInspiral.save_constants!(file, save_every, t_Fluxes, E_arr, E_dot_arr, L_arr, L_dot_arr, Q_arr, Q_dot_arr, C_arr, C_dot_arr, p_arr, e_arr, θmin_arr, save_constants, save_fluxes)
            idx_save_2 = 1
            flush(file)
        end
    end
    print("Completion: 100%   \r")

    # save remaining data
    if idx_save_1 != 1
        @views FittedInspiral.save_traj!(file, idx_save_1-1, time, lambda, r, theta, phi, gamma, save_traj, save_gamma)
        @views FittedInspiral.save_moments!(file, idx_save_1-1, Mij2_data, Sij2_data, Mijk3_data, Sijk3_data, Mijkl4_data, lmax_mass, lmax_current)
    end

    if idx_save_2 != 1
        @views FittedInspiral.save_constants!(file, idx_save_2-1, t_Fluxes, E_arr, E_dot_arr, L_arr, L_dot_arr, Q_arr, Q_dot_arr, C_arr, C_dot_arr, p_arr, e_arr, θmin_arr, save_constants, save_fluxes)
    end

    if JIT
        rm(sol_filename)
        println("JIT compilation run complete.")
    else
        println("File created: " * solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path))
        close(file)
    end
end

function compute_geodesic_arrays(sol::SciMLBase.ODESolution, λ_val::Float64, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    # deconstruct solution
    t = sol(λ_val, idxs=1);
    psi = sol(λ_val, idxs=2);
    chi = mod.(sol(λ_val, idxs=3), 2π);
    ϕ = sol(λ_val, idxs=4);

    # compute time derivatives (wrt λ)
    dt_dλ = MinoTimeGeodesics.dt_dλ(t, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dψ_dλ = MinoTimeGeodesics.dψ_dλ(t, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dχ_dλ = MinoTimeGeodesics.dχ_dλ(t, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    dϕ_dλ = MinoTimeGeodesics.dϕ_dλ(t, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)

    # compute BL coordinates r, θ and their time derivatives (wrt λ)
    r = MinoTimeGeodesics.r(psi, p, e)
    θ = acos((π/2<chi<1.5π) ? -sqrt(MinoTimeGeodesics.z(chi, θmin)) : sqrt(MinoTimeGeodesics.z(chi, θmin)))

    dr_dλ = MinoTimeGeodesics.dr_dλ(dψ_dλ, psi, p, e);
    dθ_dλ = MinoTimeGeodesics.dθ_dλ(dχ_dλ, chi, θ, θmin);

    # compute derivatives wrt t
    dr_dt = dr_dλ / dt_dλ
    dθ_dt = dθ_dλ / dt_dλ 
    dϕ_dt = dϕ_dλ / dt_dλ 

    # compute gamma factor
    v = [dr_dt, dθ_dt, dϕ_dt] # v=dxi/dt
    dt_dτ = MinoTimeGeodesics.Γ(r, θ, ϕ, v, a)
    
    # # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    # d2r_dt2 = MinoTimeGeodesics.dr2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)
    # d2θ_dt2 = MinoTimeGeodesics.dθ2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)
    # d2ϕ_dt2 = MinoTimeGeodesics.dϕ2_dt2(r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, a)
    return t, r, θ, ϕ, dt_dτ, dr_dt, dθ_dt, dϕ_dt
end

function update_waveform_arrays!(idx_save::Int64, wf_geodesic_num_points::Int64, Mij2_data::AbstractArray, Sij2_data::AbstractArray, Mijk3_data::AbstractArray, Sijk3_data::AbstractArray, Mijkl4_data::AbstractArray, Mij2_data_wf_temp::AbstractArray, Sij2_data_wf_temp::AbstractArray, Mijk3_data_wf_temp::AbstractArray, Sijk3_data_wf_temp::AbstractArray, Mijkl4_data_wf_temp::AbstractArray)
    compute_at = (wf_geodesic_num_points÷2)+1;
    @inbounds for i = 1:3, j = 1:3
        Mij2_data[i, j][idx_save] = Mij2_data_wf_temp[i, j][compute_at];
        Sij2_data[i, j][idx_save] = Sij2_data_wf_temp[i, j][compute_at];
        @inbounds for k = 1:3
            Mijk3_data[i, j, k][idx_save] = Mijk3_data_wf_temp[i, j, k][compute_at];
            Sijk3_data[i, j, k][idx_save] = Sijk3_data_wf_temp[i, j, k][compute_at];
            @inbounds for l = 1:3
                Mijkl4_data[i, j, k, l][idx_save] = Mijkl4_data_wf_temp[i, j, k, l][compute_at];
            end
        end
    end
end

function update_trajectory_arrays!(t0::Float64, idx_save::Int64, λ0::Float64, λmin::Float64, λmax::Float64, sol::SciMLBase.ODESolution, lambda::Vector{Float64}, time::Vector{Float64}, r::Vector{Float64}, theta::Vector{Float64}, phi::Vector{Float64}, gamma::Vector{Float64}, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64, save_traj::Bool, save_gamma::Bool)
    λ_saveat = t0 == 0.0 ? 0.0 : find_λ(t0, sol, λmin, λmax, tol = 1e-14)
    t, psi, chi, ϕ  = sol(λ_saveat)
    time[idx_save] = t
    lambda[idx_save] = λ_saveat + λ0

    if save_traj || save_gamma
        rr = MinoTimeGeodesics.r(psi, p, e)
        θ = acos((π/2<chi<1.5π) ? -sqrt(MinoTimeGeodesics.z(chi, θmin)) : sqrt(MinoTimeGeodesics.z(chi, θmin)))

        if save_traj
            r[idx_save] = rr
            theta[idx_save] = θ
            phi[idx_save] = ϕ
        end

        if save_gamma
            dt_dλ = MinoTimeGeodesics.dt_dλ(t, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
            dψ_dλ = MinoTimeGeodesics.dψ_dλ(t, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
            dχ_dλ = MinoTimeGeodesics.dχ_dλ(t, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
            dϕ_dλ = MinoTimeGeodesics.dϕ_dλ(t, psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)

            dr_dλ = MinoTimeGeodesics.dr_dλ(dψ_dλ, psi, p, e);
            dθ_dλ = MinoTimeGeodesics.dθ_dλ(dχ_dλ, chi, θ, θmin);

            # compute derivatives wrt t
            dr_dt = dr_dλ / dt_dλ
            dθ_dt = dθ_dλ / dt_dλ 
            dϕ_dt = dϕ_dλ / dt_dλ 

            # compute gamma factor
            v = [dr_dt, dθ_dt, dϕ_dt] # v=dxi/dt
            gamma[idx_save] = MinoTimeGeodesics.Γ(rr, θ, ϕ, v, a)
        end
    end
end

function compute_waveform_moments!(sol::SciMLBase.ODESolution, t_save::Float64, λmin::Float64, λmax::Float64, a::Float64, E::Float64, L::Float64, Q::Float64, C::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, q::Float64, xBL_wf::AbstractArray, vBL_wf::AbstractArray, aBL_wf::AbstractArray, xH_wf::AbstractArray, rH_wf::AbstractArray, vH_wf::AbstractArray, aH_wf::AbstractArray, v_wf::AbstractArray, Mij2_data_wf_temp::AbstractArray, Mijk2_data_wf_temp::AbstractArray, Mijkl2_data_wf_temp::AbstractArray, Sij1_data_wf_temp::AbstractArray, Sijk1_data_wf_temp::AbstractArray, Mijk3_data_wf_temp::AbstractArray, Mijkl4_data_wf_temp::AbstractArray, Sij2_data_wf_temp::AbstractArray, Sijk3_data_wf_temp::AbstractArray, wf_geodesic_num_points::Int64, h::Float64, lmax_mass::Int64, lmax_current::Int64, Δλi_fit::Float64, abstol::Float64, reltol::Float64, ra::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64, idx_save::Int64)
    if iseven(wf_geodesic_num_points)
        throw(DomainError(wf_geodesic_num_points, "wf_geodesic_num_points must be odd"))
    end

    use_custom_ics = true; use_specified_params = true;
    T_Fit = (wf_geodesic_num_points - 1) * h 
    # find value of λ at which t = t_save
    λ_saveat = t_save == 0.0 ? 0.0 : find_λ(t_save, sol, λmin, λmax, tol=1e-14)
    ics = sol(λ_saveat)
    ics = SVector{4, Float64}([ics[1], ics[2], mod(ics[3], 2π), ics[4]]) # ensure that chi is in the range [0, 2π]

        
    λλ, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi, dt_dλ = 
    MinoTimeGeodesics.compute_kerr_geodesic_past_and_future(a, p, e, θmin, sign_Lz, use_specified_params, wf_geodesic_num_points, T_Fit, Δλi_fit, reltol, abstol; ics=ics, E=E, L=L, Q=Q, C=C, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)

    # check that the gap between each point is approximates equal to h (required for FD to hold)
    diff_λ = diff(λλ)
    if maximum(abs.(diff_λ .- h)) > 1e-8
        throw(DomainError(diff_λ, "The difference between each point in the geodesic is not equal to h. This is required for finite differences to hold."))
    end

    # also check that the midpoint of the computed geodesic matches the specified midpoint
    r_true = MinoTimeGeodesics.r(ics[2], p, e)
    θ_true = acos((π/2<ics[3]<1.5π) ? -sqrt(MinoTimeGeodesics.z(ics[3], θmin)) : sqrt(MinoTimeGeodesics.z(ics[3], θmin)))
    compute_at = (wf_geodesic_num_points÷2)+1
    if ics != [tt[compute_at], psi[compute_at], chi[compute_at], ϕϕ[compute_at]]
        throw(DomainError(ics, "The initial conditions of the geodesic do not match the specified initial conditions. ics = $(ics), [tt, psi, chi, phi] = [$(tt[compute_at]), $(psi[compute_at]), $(chi[compute_at]), $(ϕϕ[compute_at])], compute_at = $(compute_at)"))
    end
    if ics[1] != tt[compute_at] || r_true != rr[compute_at] || θ_true != θθ[compute_at] || ics[4] != ϕϕ[compute_at]
        throw(DomainError("The midpoint of the geodesic does not match the specified midpoint. idx_save = $(idx_save), t_true = $(t_save), t[compute_at] = $(tt[compute_at]), r_true = $(r_true), θ_true = $(θ_true), rr[compute_at] = $(rr[compute_at]), θθ[compute_at] = $(θθ[compute_at]), ϕ_true = $(ics[4]), ϕϕ[compute_at] = $(ϕϕ[compute_at])"))
    end
    
    EstimateMultipoleDerivs.FiniteDifferences.compute_waveform_moments_and_derivs!(a, E, L, C, q, xBL_wf, vBL_wf, aBL_wf, xH_wf, rH_wf, vH_wf, aH_wf, v_wf, tt, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, Mij2_data_wf_temp, Mijk2_data_wf_temp, Mijkl2_data_wf_temp, Sij1_data_wf_temp, Sijk1_data_wf_temp, Mijk3_data_wf_temp, Mijkl4_data_wf_temp, Sij2_data_wf_temp, Sijk3_data_wf_temp, wf_geodesic_num_points, h, lmax_mass, lmax_current)
end

function find_λ(T::Float64, sol::SciMLBase.ODESolution, a::Float64, b::Float64; tol::Float64=1e-8, max_iter::Int64=1000)
    fa = sol(a, idxs = 1) - T
    fb = sol(b, idxs = 1) - T

    if fa * fb > 0
        error("Function must have opposite signs at the endpoints a and b.")
    end

    for i in 1:max_iter
        c = (a + b) / 2
        fc = sol(c, idxs = 1) - T

        if abs(fc) < tol || (b - a)/2 < tol
            return c  # root found
        end

        if fa * fc < 0
            b = c
            fb = fc
        else
            a = c
            fa = fc
        end
    end

    error("Maximum iterations reached without convergence.")
end

function solution_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    return data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin, sigdigits=3))_q_$(q)_psi0_$(round(psi_0, sigdigits=3))_chi0_$(round(chi_0, sigdigits=3))_phi0_$(round(phi_0, sigdigits=3))_nHarm_$(nHarm)_fit_range_factor_$(fit_time_range_factor)_fourier_"*fit*"_fit_lmax_mass_$(lmax_mass)_lmax_current_$(lmax_current).h5"
end

function waveform_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    return data_path * "Waveform_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin, sigdigits=3))_q_$(q)_psi0_$(round(psi_0, sigdigits=3))_chi0_$(round(chi_0, sigdigits=3))_phi0_$(round(phi_0, sigdigits=3))_obsDist_$(round(obs_distance, sigdigits=3))_ThetaS_$(round(ThetaSource, sigdigits=3))_PhiS_$(round(PhiSource, sigdigits=3))_ThetaK_$(round(ThetaKerr, sigdigits=3))_PhiK_$(round(PhiKerr, sigdigits=3))_nHarm_$(nHarm)_fit_range_factor_$(fit_time_range_factor)_fourier_"*fit*"_fit_lmax_mass_$(lmax_mass)_lmax_current_$(lmax_current).h5"
end

function waveform_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, obs_distance::Float64, ThetaObs::Float64, PhiObs::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    return data_path * "Waveform_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin, sigdigits=3))_q_$(q)_psi0_$(round(psi_0, sigdigits=3))_chi0_$(round(chi_0, sigdigits=3))_phi0_$(round(phi_0, sigdigits=3))_obsDist_$(round(obs_distance, sigdigits=3))_ThetaObs_$(round(ThetaObs, sigdigits=3))_PhiObs_$(round(PhiObs, sigdigits=3))_nHarm_$(nHarm)_fit_range_factor_$(fit_time_range_factor)_fourier_"*fit*"_fit_lmax_mass_$(lmax_mass)_lmax_current_$(lmax_current).h5"
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)
    return FittedInspiral.load_trajectory(sol_filename)
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)
    return FittedInspiral.load_constants_of_motion(sol_filename)
end

function compute_waveform(obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    # load waveform multipole moments
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)

    t, Mij2, Mijk3, Mijkl4, Sij2, Sijk3 = FittedInspiral.load_waveform_moments(sol_filename, lmax_mass, lmax_current)
    num_points = length(Mij2[1, 1]);
    h_plus, h_cross = Waveform.compute_wave_polarizations(num_points, obs_distance, deg2rad(ThetaSource), deg2rad(PhiSource), deg2rad(ThetaKerr), deg2rad(PhiKerr), Mij2, Mijk3, Mijkl4, Sij2, Sijk3, q)

    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaSource, PhiSource, ThetaKerr, PhiKerr, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)
    h5open(wave_filename, "w") do file
        file["t"] = t
        file["hplus"] = h_plus
        file["hcross"] = h_cross
    end
    println("File created: " * wave_filename)
end

function compute_waveform(obs_distance::Float64, ThetaObs::Float64, PhiObs::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    # load waveform multipole moments
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)

    t, Mij2, Mijk3, Mijkl4, Sij2, Sijk3 = FittedInspiral.load_waveform_moments(sol_filename, lmax_mass, lmax_current)
    num_points = length(Mij2[1, 1]);
    h_plus, h_cross = Waveform.compute_wave_polarizations(num_points, obs_distance, deg2rad(ThetaObs), deg2rad(PhiObs), Mij2, Mijk3, Mijkl4, Sij2, Sijk3, q)

    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaObs, PhiObs, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)
    h5open(wave_filename, "w") do file
        file["t"] = t
        file["hplus"] = h_plus
        file["hcross"] = h_cross
    end
    println("File created: " * wave_filename)
end

function load_waveform(obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64,a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaSource, PhiSource, ThetaKerr, PhiKerr, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)
    file = h5open(wave_filename, "r")
    t = file["t"][:]
    h_plus = file["hplus"][:]
    h_cross = file["hcross"][:]
    close(file)
    return t, h_plus, h_cross    
end

function load_waveform(obs_distance::Float64, ThetaObs::Float64, PhiObs::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaObs, PhiObs, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)
    file = h5open(wave_filename, "r")
    t = file["t"][:]
    h_plus = file["hplus"][:]
    h_cross = file["hcross"][:]
    close(file)
    return t, h_plus, h_cross    
end

function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nHarm::Int64, fit_time_range_factor::Float64, fit::String, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, nHarm, fit_time_range_factor, fit, lmax_mass, lmax_current, data_path)
    rm(sol_filename)
end

end