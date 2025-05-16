include("params.jl")

using HDF5
import ..HDF5Helper: create_file_group!, create_dataset!, append_data!
using ..SymmetricTensors

Z_1(a::Float64) = 1 + (1 - a^2 / 1.0)^(1/3) * ((1 + a)^(1/3) + (1 - a)^(1/3))
Z_2(a::Float64) = sqrt(3 * a^2 + Z_1(a)^2)
LSO_r(a::Float64) = (3 + Z_2(a) - sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # retrograde LSO
LSO_p(a::Float64) = (3 + Z_2(a) + sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # prograde LSO


using LinearAlgebra
using Combinatorics
using StaticArrays
using HDF5
using DifferentialEquations
using ...Kerr
using ...ConstantsOfMotion
using ...BLTimeGeodesics
using ...CircularNonEquatorial
using ...HarmonicCoords
using ...SymmetricTensors
using ...SelfAcceleration
using ...EstimateMultipoleDerivs
using ...EvolveConstants
using ...Waveform
using ...HarmonicCoordDerivs
using ...AnalyticCoordinateDerivs
using ...AnalyticMultipoleDerivs
using JLD2
using Printf


function initialize_solution_file!(file::HDF5.File, chunk_size::Int64, lmax_mass::Int64, lmax_current::Int64, save_traj::Bool, save_SF::Bool, save_constants::Bool, save_fluxes::Bool, save_gamma::Bool)
    create_dataset!(file, "", "t", Float64, chunk_size);

    traj_group_name = "Trajectory"
    create_file_group!(file, traj_group_name);
    
    if save_traj
        create_dataset!(file, traj_group_name, "r", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "theta", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "phi", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "r_dot", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "theta_dot", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "phi_dot", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "r_ddot", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "theta_ddot", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "phi_ddot", Float64, chunk_size);
    end

    if save_gamma
        create_dataset!(file, traj_group_name, "Gamma", Float64, chunk_size);
    end

    constants_group_name = "ConstantsOfMotion"
    create_file_group!(file, constants_group_name);
    if save_constants || save_fluxes
        create_dataset!(file, constants_group_name, "t", Float64, 1);
    end

    if save_constants
        create_dataset!(file, constants_group_name, "Energy", Float64, 1);
        create_dataset!(file, constants_group_name, "AngularMomentum", Float64, 1);
        create_dataset!(file, constants_group_name, "CarterConstant", Float64, 1);
        create_dataset!(file, constants_group_name, "AltCarterConstant", Float64, 1);
        create_dataset!(file, constants_group_name, "p", Float64, 1);
        create_dataset!(file, constants_group_name, "eccentricity", Float64, 1);
        create_dataset!(file, constants_group_name, "theta_min", Float64, 1);
    end

    if save_fluxes
        create_dataset!(file, constants_group_name, "Edot", Float64, 1);
        create_dataset!(file, constants_group_name, "Ldot", Float64, 1);
        create_dataset!(file, constants_group_name, "Qdot", Float64, 1);
        create_dataset!(file, constants_group_name, "Cdot", Float64, 1);
    end


    SF_group_name = "SelfForce"
    create_file_group!(file, SF_group_name);
    if save_SF
        create_dataset!(file, SF_group_name, "self_acc_BL_t", Float64, 1);
        create_dataset!(file, SF_group_name, "self_acc_BL_r", Float64, 1);
        create_dataset!(file, SF_group_name, "self_acc_BL_θ", Float64, 1);
        create_dataset!(file, SF_group_name, "self_acc_BL_ϕ", Float64, 1);
        create_dataset!(file, SF_group_name, "self_acc_Harm_t", Float64, 1);
        create_dataset!(file, SF_group_name, "self_acc_Harm_x", Float64, 1);
        create_dataset!(file, SF_group_name, "self_acc_Harm_y", Float64, 1);
        create_dataset!(file, SF_group_name, "self_acc_Harm_z", Float64, 1);
    end

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


function save_traj!(file::HDF5.File, chunk_size::Int64, t::Vector{Float64}, r::Vector{Float64}, θ::Vector{Float64}, ϕ::Vector{Float64}, dr_dt::Vector{Float64}, dθ_dt::Vector{Float64}, dϕ_dt::Vector{Float64}, d2r_dt2::Vector{Float64}, d2θ_dt2::Vector{Float64}, d2ϕ_dt2::Vector{Float64}, dt_dτ::Vector{Float64}, save_traj::Bool, save_gamma::Bool)
    append_data!(file, "", "t", t[1:chunk_size], chunk_size);

    traj_group_name = "Trajectory"

    if save_traj
        append_data!(file, traj_group_name, "r", r[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "theta", θ[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "phi", ϕ[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "r_dot", dr_dt[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "theta_dot", dθ_dt[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "phi_dot", dϕ_dt[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "r_ddot", d2r_dt2[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "theta_ddot", d2θ_dt2[1:chunk_size], chunk_size);
        append_data!(file, traj_group_name, "phi_ddot", d2ϕ_dt2[1:chunk_size], chunk_size);
    end

    if save_gamma
        append_data!(file, traj_group_name, "Gamma", dt_dτ[1:chunk_size], chunk_size);
    end
end

function save_moments!(file::HDF5.File, chunk_size::Int64, Mij2::AbstractArray, Sij2::AbstractArray, Mijk3::AbstractArray, Sijk3::AbstractArray, Mijkl4::AbstractArray, lmax_mass::Int64, lmax_current::Int64)

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

function save_constants!(file::HDF5.File, t::Float64, E::Float64, dE_dt::Float64, L::Float64, dL_dt::Float64, Q::Float64, dQ_dt::Float64, C::Float64, dC_dt::Float64, p::Float64, e::Float64, θmin::Float64, save_constants::Bool, save_fluxes::Bool)
    constants_group_name = "ConstantsOfMotion"
    if save_constants || save_fluxes
        append_data!(file, constants_group_name, "t", t);
    end

    if save_constants
        append_data!(file, constants_group_name, "Energy", E);
        append_data!(file, constants_group_name, "AngularMomentum", L);
        append_data!(file, constants_group_name, "CarterConstant", C);
        append_data!(file, constants_group_name, "AltCarterConstant", Q);
        append_data!(file, constants_group_name, "p", p);
        append_data!(file, constants_group_name, "eccentricity", e);
        append_data!(file, constants_group_name, "theta_min", θmin);
    end

    if save_fluxes
        append_data!(file, constants_group_name, "Edot", dE_dt);
        append_data!(file, constants_group_name, "Ldot", dL_dt);
        append_data!(file, constants_group_name, "Qdot", dQ_dt);
        append_data!(file, constants_group_name, "Cdot", dC_dt);
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

    if lmax_mass > 2 || lmax_mass == 2 && lmax_current
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

    if lmax_mass > 3 || lmax_mass == 3 && lmax_current
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

function load_trajectory(sol_filename::String; Mino::Bool=false)
    h5f = h5open(sol_filename, "r")
    t = h5f["t"][:]
    r = h5f["Trajectory/r"][:]
    θ = h5f["Trajectory/theta"][:]
    ϕ = h5f["Trajectory/phi"][:]
    dr_dt = h5f["Trajectory/r_dot"][:]
    dθ_dt = h5f["Trajectory/theta_dot"][:]
    dϕ_dt = h5f["Trajectory/phi_dot"][:]
    d2r_dt2 = h5f["Trajectory/r_ddot"][:]
    d2θ_dt2 = h5f["Trajectory/theta_ddot"][:]
    d2ϕ_dt2 = h5f["Trajectory/phi_ddot"][:]
    # dt_dτ = h5f["Trajectory/Gamma"][:]
    close(h5f)
    return t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2
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

function solution_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    return data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin, sigdigits=3))_q_$(q)_psi0_$(round(psi_0, sigdigits=3))_chi0_$(round(chi_0, sigdigits=3))_phi0_$(round(phi_0, sigdigits=3))_BL_time_lmax_mass_$(lmax_mass)_lmax_current_$(lmax_current).h5"
end

function waveform_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    return data_path * "Waveform_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin, sigdigits=3))_q_$(q)_psi0_$(round(psi_0, sigdigits=3))_chi0_$(round(chi_0, sigdigits=3))_phi0_$(round(phi_0, sigdigits=3))_obsDist_$(round(obs_distance, sigdigits=3))_ThetaS_$(round(ThetaSource, sigdigits=3))_PhiS_$(round(PhiSource, sigdigits=3))_ThetaK_$(round(ThetaKerr, sigdigits=3))_PhiK_$(round(PhiKerr, sigdigits=3))_BL_time_lmax_mass_$(lmax_mass)_lmax_current_$(lmax_current).h5"
end

function waveform_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, obs_distance::Float64, ThetaObs::Float64, PhiObs::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    return data_path * "Waveform_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin, sigdigits=3))_q_$(q)_psi0_$(round(psi_0, sigdigits=3))_chi0_$(round(chi_0, sigdigits=3))_phi0_$(round(phi_0, sigdigits=3))_obsDist_$(round(obs_distance, sigdigits=3))_ThetaObs_$(round(ThetaObs, sigdigits=3))_PhiObs_$(round(PhiObs, sigdigits=3))_BL_time_lmax_mass_$(lmax_mass)_lmax_current_$(lmax_current).h5"
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, lmax_mass, lmax_current, data_path)
    return AnalyticInspiral.load_trajectory(sol_filename)
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, lmax_mass, lmax_current, data_path)
    return AnalyticInspiral.load_constants_of_motion(sol_filename)
end

function compute_waveform(obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    # load waveform multipole moments
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, lmax_mass, lmax_current, data_path)
    t, Mij2, Mijk3, Mijkl4, Sij2, Sijk3 = AnalyticInspiral.load_waveform_moments(sol_filename, lmax_mass, lmax_current)
    num_points = length(Mij2[1, 1]);
    h_plus, h_cross = Waveform.compute_wave_polarizations(num_points, obs_distance, deg2rad(ThetaSource), deg2rad(PhiSource), deg2rad(ThetaKerr), deg2rad(PhiKerr), Mij2, Mijk3, Mijkl4, Sij2, Sijk3, q)

    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaSource, PhiSource, ThetaKerr, PhiKerr, lmax_mass, lmax_current, data_path)
    h5open(wave_filename, "w") do file
        file["t"] = t
        file["hplus"] = h_plus
        file["hcross"] = h_cross
    end
    println("File created: " * wave_filename)
end

function compute_waveform(obs_distance::Float64, ThetaObs::Float64, PhiObs::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    # load waveform multipole moments
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, lmax_mass, lmax_current, data_path)
    t, Mij2, Mijk3, Mijkl4, Sij2, Sijk3 = AnalyticInspiral.load_waveform_moments(sol_filename, lmax_mass, lmax_current)
    num_points = length(Mij2[1, 1]);
    h_plus, h_cross = Waveform.compute_wave_polarizations(num_points, obs_distance, deg2rad(ThetaObs), deg2rad(PhiObs), Mij2, Mijk3, Mijkl4, Sij2, Sijk3, q)

    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaObs, PhiObs, lmax_mass, lmax_current, data_path)
    h5open(wave_filename, "w") do file
        file["t"] = t
        file["hplus"] = h_plus
        file["hcross"] = h_cross
    end
    println("File created: " * wave_filename)
end

function load_waveform(obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaSource, PhiSource, ThetaKerr, PhiKerr, lmax_mass, lmax_current, data_path)
    file = h5open(wave_filename, "r")
    t = file["t"][:]
    h_plus = file["hplus"][:]
    h_cross = file["hcross"][:]
    close(file)
    return t, h_plus, h_cross    
end

function load_waveform(obs_distance::Float64, ThetaObs::Float64, PhiObs::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaObs, PhiObs, lmax_mass, lmax_current, data_path)
    file = h5open(wave_filename, "r")
    t = file["t"][:]
    h_plus = file["hplus"][:]
    h_cross = file["hcross"][:]
    close(file)
    return t, h_plus, h_cross    
end

# useful for dummy runs (e.g., for resonances to estimate the duration of time needed by computing the time derivative of the fundamental frequencies)
function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, lmax_mass::Int64, lmax_current::Int64, data_path::String)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, lmax_mass, lmax_current, data_path)
    rm(sol_filename)
end

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
JIT = true;

a, p, e, θmin, sign_Lz, q, psi_0, chi_0, phi_0, nPointsGeodesic, compute_SF, tInspiral, reltol, abstol, data_path, JIT, lmax_mass, lmax_current, save_traj, save_SF, save_constants, save_fluxes, save_gamma = emri.a, emri.p, emri.e, emri.θmin, emri.sign_Lz, emri.mass_ratio, emri.psi0, emri.chi0, emri.phi0, nPointsGeodesic, compute_fluxes, t_max_M, emri.reltol, emri.abstol, emri.path, JIT, emri.lmax_mass, emri.lmax_current, emri.save_traj, emri.save_SF, emri.save_constants, emri.save_fluxes, emri.save_gamma

# create solution file
sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, lmax_mass, lmax_current, data_path)

if isfile(sol_filename)
    rm(sol_filename)
end

if JIT
    tInspiral = 20.0 # dummy run for Δt = 20M
end

file = h5open(sol_filename, "w")

# second argument is chunk_size. Since each successive geodesic piece overlap at the end of the first and bgeinning of the second, we must manually save this point only once to avoid repeats in the data 
AnalyticInspiral.initialize_solution_file!(file, nPointsGeodesic-1, lmax_mass, lmax_current, save_traj, save_SF, save_constants, save_fluxes, save_gamma)

# create arrays to store multipole moments necessary for waveform computation
Mij2_wf = [Float64[] for i=1:3, j=1:3];
Mijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];
Mijkl4_wf = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
Sij2_wf = [Float64[] for i=1:3, j=1:3];
Sijk3_wf = [Float64[] for i=1:3, j=1:3, k=1:3];

# initialize data arrays
aSF_BL = Vector{Vector{Float64}}()
aSF_H = Vector{Vector{Float64}}()

# initialize derivative arrays
dxBL_dt=zeros(3); d2xBL_dt=zeros(3); d3xBL_dt=zeros(3); d4xBL_dt=zeros(3);
d5xBL_dt=zeros(3); d6xBL_dt=zeros(3); d7xBL_dt=zeros(3); d8xBL_dt=zeros(3);

dx_dλ=zeros(3); d2x_dλ=zeros(3); d3x_dλ=zeros(3); d4x_dλ=zeros(3);
d5x_dλ=zeros(3); d6x_dλ=zeros(3); d7x_dλ=zeros(3); d8x_dλ=zeros(3);

xH=zeros(3); dxH_dt=zeros(3); d2xH_dt=zeros(3); d3xH_dt=zeros(3); d4xH_dt=zeros(3);
d5xH_dt=zeros(3); d6xH_dt=zeros(3); d7xH_dt=zeros(3); d8xH_dt=zeros(3);

# arrays for multipole moments
Mijk2_data = [Float64[] for i=1:3, j=1:3, k=1:3]
Mij2_data = [Float64[] for i=1:3, j=1:3]
Mijkl2_data = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
Sij1_data = [Float64[] for i=1:3, j=1:3]
Sijk1_data= [Float64[] for i=1:3, j=1:3, k=1:3]

# "temporary" mulitpole arrays which contain the multipole data for a given piecewise geodesic
Mij2_wf_temp = [Float64[] for i=1:3, j=1:3];
Mijk3_wf_temp = [Float64[] for i=1:3, j=1:3, k=1:3];
Mijkl4_wf_temp = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3];
Sij2_wf_temp = [Float64[] for i=1:3, j=1:3];
Sijk3_wf_temp = [Float64[] for i=1:3, j=1:3, k=1:3];

# arrays for self-force computation
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
xH_SF = zeros(3);
vH_SF = zeros(3);
aH_SF = zeros(3);

# compute apastron
ra = p / (1 - e);

# calculate integrals of motion from orbital parameters
EEi, LLi, QQi, CCi = ConstantsOfMotion.compute_ELC(a, p, e, θmin, sign_Lz)   

# store orbital params in arrays
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
geodesic_ics = @SArray [psi_0, chi_0, phi_0];

rLSO = AnalyticInspiral.LSO_p(a)

use_custom_ics = true; use_specified_params = true;
save_at_trajectory = compute_SF / (nPointsGeodesic - 1); Δti=save_at_trajectory;    # initial time step for geodesic integration

geodesic_time_length = compute_SF;
num_points_geodesic = nPointsGeodesic;

# initialize fluxes to zero (serves as a flag in the function which evolves the constants of motion)
dE_dt = 0.0
dL_dt = 0.0
dQ_dt = 0.0
dC_dt = 0.0

# iteration
# @time begin
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
@time tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi = BLTimeGeodesics.compute_kerr_geodesic(a, p_t, e_t, θmin_t, sign_Lz, num_points_geodesic, use_specified_params, geodesic_time_length, Δti, reltol, abstol;
ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false);

tt = tt .+ t0   # tt from the above function call starts from zero

# check that geodesic output is as expected
@time if (length(tt) != num_points_geodesic) || !isapprox(tt[nPointsGeodesic], t0 + compute_SF)
    println("Integration terminated at t = $(first(tt))")
    println("total_num_points - len(sol) = $(num_points_geodesic-length(tt))")
    println("tt[nPointsGeodesic] = $(tt[nPointsGeodesic])")
    println("t0 + compute_SF = $(t0 + compute_SF)")
end

# extract initial conditions for next geodesic.
t0 = last(tt); geodesic_ics = @SArray [last(psi), last(chi), last(ϕϕ)];

# store physical trajectory — this function omits last point since this overlaps with the start of the next geodesic
@time AnalyticInspiral.save_traj!(file, nPointsGeodesic-1, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, save_traj, save_gamma);
# end;

# @time begin
###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######
using BenchmarkTools
include("../AnalyticMultipoleDerivs.jl")
@benchmark AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_WF!(rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Mij2_data, Mijk3_wf_temp,
Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp, a, q, E_t, L_t, C_t, lmax_mass, lmax_current)

# store multipole data for waveforms — note that we only save the independent components
@time AnalyticInspiral.save_moments!(file, nPointsGeodesic-1, Mij2_data, Sij2_wf_temp, Mijk3_wf_temp, Sijk3_wf_temp, Mijkl4_wf_temp, lmax_mass, lmax_current);


###### COMPUTE MULTIPOLE MOMENTS FOR SELF FORCE ######
### COMPUTE BL COORDINATE DERIVATIVES ###
xBL_SF = [last(rr), last(θθ), last(ϕϕ)];
vBL_SF = [last(r_dot), last(θ_dot), last(ϕ_dot)];
aBL_SF = [last(r_ddot), last(θ_ddot), last(ϕ_ddot)];

@time AnalyticCoordinateDerivs.ComputeDerivs!(xBL_SF, sign(vBL_SF[1]), sign(vBL_SF[2]), dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,
dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, d8x_dλ, a, E_t, L_t, C_t);

# COMPUTE HARMONIC COORDINATE DERIVATIVES
@time HarmonicCoordDerivs.compute_harmonic_derivs!(xBL_SF, dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,
xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, a);

# COMPUTE MULTIPOLE DERIVATIVES
@time AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_SF!(xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, q, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, lmax_mass, lmax_current);


##### COMPUTE SELF-FORCE #####
HarmonicCoords.xBLtoH!(xH_SF, xBL_SF, a);
HarmonicCoords.vBLtoH!(vH_SF, xH_SF, vBL_SF, a); 
HarmonicCoords.aBLtoH!(aH_SF, xH_SF, vBL_SF, aBL_SF, a);
rH_SF = SelfAcceleration.norm_3d(xH_SF);
v_SF = SelfAcceleration.norm_3d(vH_SF);

@time SelfAcceleration.aRRα(aSF_H_temp, aSF_BL_temp, xH_SF, v_SF, vH_SF, xBL_SF, rH_SF, a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6);

# store self force values
if save_SF
    AnalyticInspiral.save_self_acceleration!(file, aSF_BL_temp, aSF_H_temp)
end

# update orbital constants and fluxes — function takes as argument the fluxes computed at the end of the previous geodesic (which overlaps with the start of the current geodesic piece) in order to update the fluxes using the trapezium rule
Δt = last(tt) - tt[1]
E_1, dE_dt, L_1, dL_dt, Q_1, dQ_dt, C_1, dC_dt, p_1, e_1, θmin_1 = EvolveConstants.Evolve_BL(Δt, a, last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot), aSF_BL_temp, E_t, dE_dt, L_t, dL_dt, Q_t, dQ_dt, C_t, dC_dt, p_t, e_t, θmin_t)

# save orbital constants and fluxes
@time AnalyticInspiral.save_constants!(file, last(tt), E_t, dE_dt, L_t, dL_dt, Q_t, dQ_dt, C_t, dC_dt, p_t, e_t, θmin_t, save_constants, save_fluxes);

E_t = E_1; L_t = L_1; Q_t = Q_1; C_t = C_1; p_t = p_1; e_t = e_1; θmin_t = θmin_1;
# end;