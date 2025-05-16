module AnalyticInspiral
using HDF5
import ..HDF5Helper: create_file_group!, create_dataset!, append_data!
using ..SymmetricTensors

Z_1(a::Float64) = 1 + (1 - a^2 / 1.0)^(1/3) * ((1 + a)^(1/3) + (1 - a)^(1/3))
Z_2(a::Float64) = sqrt(3 * a^2 + Z_1(a)^2)
LSO_r(a::Float64) = (3 + Z_2(a) - sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # retrograde LSO
LSO_p(a::Float64) = (3 + Z_2(a) + sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # prograde LSO


function initialize_solution_file!(file::HDF5.File, chunk_size::Int64, lmax_mass::Int64, lmax_current::Int64, save_traj::Bool, save_constants::Bool, save_fluxes::Bool, save_gamma::Bool)
    create_dataset!(file, "", "t", Float64, chunk_size);

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

@views function save_traj!(file::HDF5.File, chunk_size::Int64, t::Vector{Float64}, r::Vector{Float64}, θ::Vector{Float64}, ϕ::Vector{Float64}, dt_dτ::Vector{Float64}, save_traj::Bool, save_gamma::Bool)
    append_data!(file, "", "t", t[1:chunk_size], chunk_size);
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
    close(h5f)
    return t, r, θ, ϕ
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

module BLTime
using LinearAlgebra
using Combinatorics
using StaticArrays
using HDF5
using DifferentialEquations
using ....Kerr
using ....ConstantsOfMotion
using ....BLTimeGeodesics
using ....CircularNonEquatorial
using ....HarmonicCoords
using ....SymmetricTensors
using ....SelfAcceleration
using ....EstimateMultipoleDerivs
using ....EvolveConstants
using ....Waveform
using ....HarmonicCoordDerivs
using ....AnalyticCoordinateDerivs
using ....AnalyticMultipoleDerivs
using JLD2
using Printf
using ...AnalyticInspiral

"""
    compute_inspiral(args...)

Evolve inspiral with Boyer-Lindquist coordinate time parameterization and fully analyitc computation of the approximate self-force (as opposed to the Fourier fitting approach).

- `tInspiral::Float64`: total coordinate time to evolve the inspiral.
- `compute_SF::Float64`: BL time interval between self-force computations.
- `q::Float64`: mass ratio.
- `a::Float64`: black hole spin 0 < a < 1.
- `p::Float64`: initial semi-latus rectum.
- `e::Float64`: initial eccentricity.
- `θmin::Float64`: initial inclination angle.
- `sign_Lz::Int64`: sign of the z-component of the angular momentum (+1 for prograde, -1 for retrograde).
- `psi_0::Float64`: initial radial angle variable.
- `chi_0::Float64`: initial polar angle variable.
- `phi_0::Float64`: initial azimuthal angle.
- `reltol`: relative tolerance for ODE solver.
- `abstol`: absolute tolerance for ODE solver.
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

function compute_inspiral(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, compute_SF::Float64, tInspiral::Float64, dt_save::Float64, save_every::Int64, reltol::Float64=1e-14, abstol::Float64=1e-14; data_path::String="Data/", JIT::Bool=false, lmax_mass::Int64, lmax_current::Int64, save_traj::Bool, save_constants::Bool, save_fluxes::Bool, save_gamma::Bool)

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
    AnalyticInspiral.initialize_solution_file!(file, save_every, lmax_mass, lmax_current, save_traj, save_constants, save_fluxes, save_gamma)

    # initialize data arrays for trajectory and multipole moments which will be used for post-processing
    idx_save_1 = 1;
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

    # create arrays to store multipole moments necessary for waveform computation
    Mij2_data_wf_temp = zeros(3, 3);
    Mijk3_data_wf_temp = zeros(3, 3, 3);
    Mijkl4_data_wf_temp = zeros(3, 3, 3, 3);
    Sij2_data_wf_temp = zeros(3, 3);
    Sijk3_data_wf_temp = zeros(3, 3, 3);
    
    # initialize derivative arrays
    xBL = zeros(3); vBL = zeros(3); aBL = zeros(3);
    
    dxBL_dt=zeros(3); d2xBL_dt=zeros(3); d3xBL_dt=zeros(3); d4xBL_dt=zeros(3);
    d5xBL_dt=zeros(3); d6xBL_dt=zeros(3); d7xBL_dt=zeros(3); d8xBL_dt=zeros(3);

    dx_dλ=zeros(3); d2x_dλ=zeros(3); d3x_dλ=zeros(3); d4x_dλ=zeros(3);
    d5x_dλ=zeros(3); d6x_dλ=zeros(3); d7x_dλ=zeros(3); d8x_dλ=zeros(3);

    xH=zeros(3); dxH_dt=zeros(3); d2xH_dt=zeros(3); d3xH_dt=zeros(3); d4xH_dt=zeros(3);
    d5xH_dt=zeros(3); d6xH_dt=zeros(3); d7xH_dt=zeros(3); d8xH_dt=zeros(3);

    vH = zeros(3);
    aH = zeros(3);

    # arrays for self-force computation
    Mij5 = zeros(3, 3)
    Mij6 = zeros(3, 3)
    Mij7 = zeros(3, 3)
    Mij8 = zeros(3, 3)
    Mijk7 = zeros(3, 3, 3)
    Mijk8 = zeros(3, 3, 3)
    Sij5 = zeros(3, 3)
    Sij6 = zeros(3, 3)
    aSF_BL = zeros(4)
    aSF_H = zeros(4)

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
    rLSO = AnalyticInspiral.LSO_p(a)

    # initialize ODE problem
    E, L, Q, C, ra, p3, p4, zp, zm = BLTimeGeodesics.compute_ODE_params(a, p, e, θmin, sign_Lz);

    params = @SArray [a, E, L, p, e, θmin, p3, p4, zp, zm];
    ics = @SArray[psi_0, chi_0, phi_0];

    # initial conditions for Kerr geodesic trajectory
    tspan = (0.0, tInspiral);

    prob = e == 0.0 ? ODEProblem(BLTimeGeodesics.HJ_Eqns_circular, ics, tspan, params) : ODEProblem(BLTimeGeodesics.HJ_Eqns, ics, tspan, params);

    # times at which the integrator should be stopped (consists of times at which the self-force must be computed and times at which the solution must be saved)
    times_SF = range(start = compute_SF, stop = tInspiral, step = compute_SF) |> collect;
    times_WF = range(start = dt_save, stop = tInspiral, step = dt_save) |> collect;

    # initialize integrator
    # save_at_trajectory = compute_SF / (500 - 1); Δti=save_at_trajectory;
    integrator = init(prob, AutoTsit5(RK4()), adaptive=true, reltol = reltol, abstol = abstol)

    # save solution at time t = 0;
    compute_waveform_moments!(integrator, a, E_t, L_t, Q_t, C_t, p_t, e_t, θmin_t, q, Mij2_data_wf_temp, Mijk3_data_wf_temp, Mijkl4_data_wf_temp, Sij2_data_wf_temp, Sijk3_data_wf_temp, lmax_mass, lmax_current, ra, p3, p4, zp, zm, xBL, vBL, aBL, dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ,d7x_dλ, d8x_dλ,xH, dxH_dt, d2xH_dt,d3xH_dt, d4xH_dt, d5xH_dt,d6xH_dt, d7xH_dt, d8xH_dt)
    update_waveform_arrays!(idx_save_1, Mij2_data, Sij2_data, Mijk3_data, Sijk3_data, Mijkl4_data, Mij2_data_wf_temp, Sij2_data_wf_temp, Mijk3_data_wf_temp, Sijk3_data_wf_temp, Mijkl4_data_wf_temp)
    update_trajectory_arrays!(integrator, idx_save_1, time, r, theta, phi, gamma, a, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm, save_traj, save_gamma)
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

    WF_step = false
    SF_step = false

    t0 = 0.0
    while integrator.t < tInspiral
        print("Completion: $(round(100 * t0/tInspiral; digits=5))%   \r")
        flush(stdout)
        
        if length(times_SF) == 0 && length(times_WF) == 0
            break
        elseif length(times_WF) == 0
            tF = times_SF[1]
            popfirst!(times_SF)
            SF_step = true
            WF_step = false
        elseif length(times_SF) == 0
            tF = times_WF[1]
            popfirst!(times_WF)
            WF_step = true
            SF_step = false
        elseif times_SF[1] < times_WF[1]
            tF = times_SF[1]
            popfirst!(times_SF)
            SF_step = true
            WF_step = false
        else
            tF = times_WF[1]
            popfirst!(times_WF)
            WF_step = true
            SF_step = false
        end

        time_step = tF - integrator.t
        step!(integrator, time_step, true)

        if WF_step
            if idx_save_1 == save_every + 1
                AnalyticInspiral.save_traj!(file, save_every, time, r, theta, phi, gamma, save_traj, save_gamma)
                AnalyticInspiral.save_moments!(file, save_every, Mij2_data, Sij2_data, Mijk3_data, Sijk3_data, Mijkl4_data, lmax_mass, lmax_current)
                idx_save_1 = 1
            end

            compute_waveform_moments!(integrator, a, E_t, L_t, Q_t, C_t, p_t, e_t, θmin_t, q, Mij2_data_wf_temp, Mijk3_data_wf_temp, Mijkl4_data_wf_temp, Sij2_data_wf_temp, Sijk3_data_wf_temp, lmax_mass, lmax_current, ra, p3, p4, zp, zm, xBL, vBL, aBL, dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ,d7x_dλ, d8x_dλ,xH, dxH_dt, d2xH_dt,d3xH_dt, d4xH_dt, d5xH_dt,d6xH_dt, d7xH_dt, d8xH_dt)
            update_waveform_arrays!(idx_save_1, Mij2_data, Sij2_data, Mijk3_data, Sijk3_data, Mijkl4_data, Mij2_data_wf_temp, Sij2_data_wf_temp, Mijk3_data_wf_temp, Sijk3_data_wf_temp, Mijkl4_data_wf_temp)
            update_trajectory_arrays!(integrator, idx_save_1, time, r, theta, phi, gamma, a, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm, save_traj, save_gamma)
            idx_save_1 += 1
        else
            tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi = compute_geodesic_arrays(integrator, a, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm)

            ### COMPUTE BL COORDINATE DERIVATIVES ###
            xBL[1] = rr; xBL[2] = θθ; xBL[3] = ϕϕ;
            vBL[1] = r_dot; vBL[2] = θ_dot; vBL[3] = ϕ_dot;
            aBL[1] = r_ddot; aBL[2] = θ_ddot; aBL[3] = ϕ_ddot;

            AnalyticCoordinateDerivs.ComputeDerivs!(xBL, sign(vBL[1]), sign(vBL[2]), dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, d8x_dλ, a, E_t, L_t, C_t);

            # COMPUTE HARMONIC COORDINATE DERIVATIVES
            HarmonicCoordDerivs.compute_harmonic_derivs!(xBL, dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt, xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, a);

            # COMPUTE MULTIPOLE DERIVATIVES
            AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_SF!(xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, q, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, lmax_mass, lmax_current);

            ##### COMPUTE SELF-FORCE #####
            HarmonicCoords.xBLtoH!(xH, xBL, a);
            HarmonicCoords.vBLtoH!(vH, xH, vBL, a); 
            HarmonicCoords.aBLtoH!(aH, xH, vBL, aBL, a);
            rH = SelfAcceleration.norm_3d(xH);
            v = SelfAcceleration.norm_3d(vH);

            SelfAcceleration.aRRα(aSF_H, aSF_BL, xH, v, vH, xBL, rH, a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)

            # update orbital constants and fluxes — function takes as argument the fluxes computed at the end of the previous geodesic (which overlaps with the start of the current geodesic piece) in order to update the fluxes using the trapezium rule
            dt_flux = tF - t0
            E_1, dE_dt, L_1, dL_dt, Q_1, dQ_dt, C_1, dC_dt, p_1, e_1, θmin_1 = EvolveConstants.Evolve_BL(dt_flux, a, rr, θθ, ϕϕ, dt_dτ, r_dot, θ_dot, ϕ_dot, aSF_BL, E_t, dE_dt, L_t, dL_dt, Q_t, dQ_dt, C_t, dC_dt, p_t, e_t, θmin_t)

            E_t = E_1; L_t = L_1; Q_t = Q_1; C_t = C_1; p_t = p_1; e_t = e_1; θmin_t = θmin_1;
            # flush(file)

            # save constants of motion
            t_Fluxes[idx_save_2] = tF;
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
                AnalyticInspiral.save_constants!(file, save_every, t_Fluxes, E_arr, E_dot_arr, L_arr, L_dot_arr, Q_arr, Q_dot_arr, C_arr, C_dot_arr, p_arr, e_arr, θmin_arr, save_constants, save_fluxes)
                idx_save_2 = 1
            end

            # update ODE params
            zm = cos(θmin_t)^2
            zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
            ra=p_t / (1.0 - e_t); rp=p_t / (1.0 + e_t);
            A = 1.0 / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
            B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
            r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
            p3 = r3 * (1.0 - e_t); p4 = r4 * (1.0 + e_t)    # Above Eq. 96
            integrator.p = @SArray [a, E_t, L_t, p_t, e_t, θmin_t, p3, p4, zp, zm];
            t0 = tF
        end
    end
    print("Completion: 100%   \r")

    # save remaining data
    if idx_save_1 != 1
        @views AnalyticInspiral.save_traj!(file, idx_save_1-1, time, r, theta, phi, gamma, save_traj, save_gamma)
        @views AnalyticInspiral.save_moments!(file, idx_save_1-1, Mij2_data, Sij2_data, Mijk3_data, Sijk3_data, Mijkl4_data, lmax_mass, lmax_current)
    end

    if idx_save_2 != 1
        @views AnalyticInspiral.save_constants!(file, idx_save_2-1, t_Fluxes, E_arr, E_dot_arr, L_arr, L_dot_arr, Q_arr, Q_dot_arr, C_arr, C_dot_arr, p_arr, e_arr, θmin_arr, save_constants, save_fluxes)
    end

    if JIT
        rm(sol_filename)
        println("JIT compilation run complete.")
    else
        println("File created: " * sol_filename)
    end
    close(file)
end

function compute_geodesic_arrays(integrator, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64)
    # deconstruct solution
    t = integrator.t;
    psi = integrator.u[1];
    chi = mod.(integrator.u[2], 2π);
    ϕ = integrator.u[3];

    # compute time derivatives
    psi_dot = BLTimeGeodesics.psi_dot(psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    chi_dot = BLTimeGeodesics.chi_dot(psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)
    ϕ_dot = BLTimeGeodesics.phi_dot(psi, chi, ϕ, a, E, L, p, e, θmin, p3, p4, zp, zm)

    # compute BL coordinates t, r, θ and their time derivatives
    r = BLTimeGeodesics.r(psi, p, e)
    θ = acos((π/2<chi<1.5π) ? -sqrt(BLTimeGeodesics.z(chi, θmin)) : sqrt(BLTimeGeodesics.z(chi, θmin)))
    r_dot = BLTimeGeodesics.dr_dt(psi_dot, psi, p, e);
    θ_dot = BLTimeGeodesics.dθ_dt(chi_dot, chi, θ, θmin);
    v = [r_dot, θ_dot, ϕ_dot];
    dt_dτ = BLTimeGeodesics.Γ(r, θ, ϕ, v, a)

    # substitute solution back into geodesic equation to find second derivatives of BL coordinates (wrt t)
    r_ddot = BLTimeGeodesics.dr2_dt2(r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a)
    θ_ddot = BLTimeGeodesics.dθ2_dt2(r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a)
    ϕ_ddot = BLTimeGeodesics.dϕ2_dt2(r, θ, ϕ, r_dot, θ_dot, ϕ_dot, a)

    return t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi
end


function compute_waveform_moments!(integrator, a::Float64, E::Float64, L::Float64, Q::Float64, C::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, Mij2_data_wf_temp::AbstractArray, Mijk3_data_wf_temp::AbstractArray, Mijkl4_data_wf_temp::AbstractArray, Sij2_data_wf_temp::AbstractArray, Sijk3_data_wf_temp::AbstractArray, lmax_mass::Int64, lmax_current::Int64, ra::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64, xBL::AbstractVector{Float64}, vBL::AbstractVector{Float64}, aBL::AbstractVector{Float64}, dxBL_dt::AbstractVector{Float64}, d2xBL_dt::AbstractVector{Float64}, d3xBL_dt::AbstractVector{Float64}, d4xBL_dt::AbstractVector{Float64}, d5xBL_dt::AbstractVector{Float64}, d6xBL_dt::AbstractVector{Float64}, d7xBL_dt::AbstractVector{Float64}, d8xBL_dt::AbstractVector{Float64},dx_dλ::AbstractVector{Float64}, d2x_dλ::AbstractVector{Float64}, d3x_dλ::AbstractVector{Float64}, d4x_dλ::AbstractVector{Float64}, d5x_dλ::AbstractVector{Float64}, d6x_dλ::AbstractVector{Float64},d7x_dλ::AbstractVector{Float64}, d8x_dλ::AbstractVector{Float64},xH::AbstractVector{Float64}, dxH_dt::AbstractVector{Float64}, d2xH_dt::AbstractVector{Float64},d3xH_dt::AbstractVector{Float64}, d4xH_dt::AbstractVector{Float64}, d5xH_dt::AbstractVector{Float64},d6xH_dt::AbstractVector{Float64}, d7xH_dt::AbstractVector{Float64}, d8xH_dt::AbstractVector{Float64})

    t, r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi = compute_geodesic_arrays(integrator, a, E, L, p, e, θmin, p3, p4, zp, zm)

    AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_WF!(r, θ, ϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Mij2_data_wf_temp, Mijk3_data_wf_temp, Mijkl4_data_wf_temp, Sij2_data_wf_temp, Sijk3_data_wf_temp, a, q, E, L, C, lmax_mass, lmax_current, xBL, vBL, aBL, dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ,d7x_dλ, d8x_dλ,xH, dxH_dt, d2xH_dt,d3xH_dt, d4xH_dt, d5xH_dt,d6xH_dt, d7xH_dt, d8xH_dt)  
end


function update_waveform_arrays!(idx_save::Int64, Mij2_data::AbstractArray, Sij2_data::AbstractArray, Mijk3_data::AbstractArray, Sijk3_data::AbstractArray, Mijkl4_data::AbstractArray, Mij2_data_wf_temp::AbstractArray, Sij2_data_wf_temp::AbstractArray, Mijk3_data_wf_temp::AbstractArray, Sijk3_data_wf_temp::AbstractArray, Mijkl4_data_wf_temp::AbstractArray)
    @inbounds for i = 1:3, j = 1:3
        Mij2_data[i, j][idx_save] = Mij2_data_wf_temp[i, j];
        Sij2_data[i, j][idx_save] = Sij2_data_wf_temp[i, j];
        @inbounds for k = 1:3
            Mijk3_data[i, j, k][idx_save] = Mijk3_data_wf_temp[i, j, k];
            Sijk3_data[i, j, k][idx_save] = Sijk3_data_wf_temp[i, j, k];
            @inbounds for l = 1:3
                Mijkl4_data[i, j, k, l][idx_save] = Mijkl4_data_wf_temp[i, j, k, l];
            end
        end
    end
end

function update_trajectory_arrays!(integrator, idx_save::Int64, time::Vector{Float64}, r::Vector{Float64}, theta::Vector{Float64}, phi::Vector{Float64}, gamma::Vector{Float64}, a::Float64, E::Float64, L::Float64, p::Float64, e::Float64, θmin::Float64, p3::Float64, p4::Float64, zp::Float64, zm::Float64, save_traj::Bool, save_gamma::Bool)
    t = integrator.t;
    psi = integrator.u[1];
    chi = mod.(integrator.u[2], 2π);
    ϕ = integrator.u[3];

    time[idx_save] = t

    if save_traj || save_gamma
        tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, dt_dτ, psi, chi = compute_geodesic_arrays(integrator, a, E, L, p, e, θmin, p3, p4, zp, zm)
    end

    if save_traj
        r[idx_save] = rr
        theta[idx_save] = θθ
        phi[idx_save] = ϕϕ
    end

    if save_gamma
        gamma[idx_save] = dt_dτ
    end
end


# evolve inspiral along one piecewise geodesic
function evolve_inspiral!(integrator, h::Number, tt::Vector{<:Number}, dt_dτ::Vector{<:Number}, rr::Vector{<:Number}, r_dot::Vector{<:Number}, r_ddot::Vector{<:Number}, θθ::Vector{<:Number}, θ_dot::Vector{<:Number}, θ_ddot::Vector{<:Number}, 
    ϕϕ::Vector{<:Number}, ϕ_dot::Vector{<:Number}, ϕ_ddot::Vector{<:Number}, dt_dλλ::Vector{<:Number})
    a, E, L, p, e, θmin, p3, p4, zp, zm = integrator.p
    track_num_steps = 0
    @inbounds for i = 1:length(tt)
        track_num_steps += 1
        compute_BL_coords_traj!(integrator, i, λλ, tt, dt_dτ, rr, r_dot, r_ddot, θθ, θ_dot, θ_ddot, ϕϕ, ϕ_dot, ϕ_ddot, dt_dλλ, a, E, L, p, e, θmin, p3, p4, zp, zm)
        step!(integrator, h, true)
    end
    if track_num_steps != length(λλ)
        throw(ArgumentError("Length of λλ array does not match the number ($(i)) of steps taken"))
    end
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

end

module MinoTime
using LinearAlgebra
using Combinatorics
using StaticArrays
using HDF5
using DifferentialEquations
using ....Kerr
using ....ConstantsOfMotion
using ....BLTimeGeodesics
using ....CircularNonEquatorial
using ....HarmonicCoords
using ....SymmetricTensors
using ....SelfAcceleration
using ....EstimateMultipoleDerivs
using ....EvolveConstants
using ....Waveform
using ....HarmonicCoordDerivs
using ....AnalyticCoordinateDerivs
using ....AnalyticMultipoleDerivs
using JLD2
using Printf
using ...AnalyticInspiral



function compute_inspiral(tInspiral::Float64, compute_SF::Float64, nPointsGeodesic::Int64, a::Float64, p::Float64, e::Float64, θmin::Float64, reltol::Float64=1e-14, abstol::Float64=1e-14; data_path::String="Data/")

    # create arrays for trajectory
    λ = Float64[]; t = Float64[]; r = Float64[]; θ = Float64[]; ϕ = Float64[];
    dt_dτ = Float64[]; dr_dt = Float64[]; dθ_dt = Float64[]; dϕ_dt = Float64[];
    d2r_dt2 = Float64[]; d2θ_dt2 = Float64[]; d2ϕ_dt2 = Float64[]; dt_dλ = Float64[];
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

    # compute apastron
    ra = p / (1 - e);

    # calculate integrals of motion from orbital parameters
    EEi, LLi, QQi, CCi = ConstantsOfMotion.compute_ELC(a, p, e, θmin)   

    # store orbital params in arrays
    EE = ones(1) * EEi; 
    Edot = zeros(1);
    LL = ones(1) * LLi; 
    Ldot = zeros(1);
    CC = ones(1) * CCi;
    Cdot = zeros(1);
    QQ = ones(1) * QQi
    Qdot = zeros(1);
    pArray = ones(1) * p;
    ecc = ones(1) * e;
    θminArray = ones(1) * θmin;

    rplus = Kerr.KerrMetric.rplus(a); rminus = Kerr.KerrMetric.rminus(a);
    # initial condition for Kerr geodesic trajectory
    t0 = 0.0
    t_Fluxes = ones(1) * t0
    λ0 = 0.0
    geodesic_icsBL.Mino_ics(t0, ra, p, e);

    rLSO = AnalyticInspiral.LSO_p(a)

    use_custom_ics = true; use_specified_params = true;
    save_at_trajectory = compute_SF / (nPointsGeodesic - 1); Δλi=save_at_trajectory;    # initial time step for geodesic integration

    # in the code, we will want to compute the geodesic with an additional time step at the end so that these coordinate values can be used as initial conditions for the
    # subsequent geodesic
    geodesic_time_length = compute_SF + save_at_trajectory;
    num_points_geodesic = nPointsGeodesic + 1;

    while tInspiral > t0
        print("Completion: $(round(100 * t0/tInspiral; digits=5))%   \r")
        flush(stdout) 

        ###### COMPUTE PIECEWISE GEODESIC ######
        # orbital parameters
        E_t = last(EE); L_t = last(LL); C_t = last(CC); Q_t = last(QQ); p_t = last(pArray); θmin_t = last(θminArray); e_t = last(ecc);

        # compute roots of radial function R(r)
        zm = cos(θmin_t)^2
        zp = C_t / (a^2 * (1.0-E_t^2) * zm)    # Eq. E23
        ra=p_t / (1.0 - e_t); rp=p_t / (1.0 + e_t);
        A = 1.0 / (1.0 - E_t^2) - (ra + rp) / 2.0    # Eq. E20
        B = a^2 * C_t / ((1.0 - E_t^2) * ra * rp)    # Eq. E21
        r3 = A + sqrt(A^2 - B); r4 = A - sqrt(A^2 - B);    # Eq. E19
        p3 = r3 * (1.0 - e_t); p4 = r4 * (1.0 + e_t)    # Above Eq. 96

        # geodesic
        λλ, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi, dt_dλλBL.compute_kerr_geodesic(a, p_t, e_t, θmin_t, num_points_geodesic, use_custom_ics,
        use_specified_params, geodesic_time_length, Δλi, reltol, abstol; ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)
        
        λλ = λλ .+ λ0   # λλ from the above function call starts from zero 

        # check that geodesic output is as expected
        if (length(λλ) != num_points_geodesic) || !isapprox(λλ[nPointsGeodesic], λ0 + compute_SF)
            println("Integration terminated at t = $(last(t))")
            println("total_num_points - len(sol) = $(num_points_geodesic-length(λλ))")
            println("λλ[nPointsGeodesic] = $(λλ[nPointsGeodesic])")
            println("λ0 + compute_SF = $(λ0 + compute_SF)")
            break
        end

        # extract initial conditions for next geodesic, then remove these points from the data array
        λ0 = last(λλ); t0 = last(tt); geodesic_ics = @SArray [t0, last(psi), last(chi), last(ϕϕ)];

        pop!(λλ); pop!(tt); pop!(rr); pop!(θθ); pop!(ϕϕ); pop!(r_dot); pop!(θ_dot); pop!(ϕ_dot);
        pop!(r_ddot); pop!(θ_ddot); pop!(ϕ_ddot); pop!(Γ); pop!(psi); pop!(chi); pop!(dt_dλλ)

        # store physical trajectory
        append!(λ, λλ); append!(t, tt); append!(dt_dτ, Γ); append!(r, rr); append!(dr_dt, r_dot); append!(d2r_dt2, r_ddot); 
        append!(θ, θθ); append!(dθ_dt, θ_dot); append!(d2θ_dt2, θ_ddot); append!(ϕ, ϕϕ); append!(dϕ_dt, ϕ_dot);
        append!(d2ϕ_dt2, ϕ_ddot); append!(dt_dλ, dt_dλλ);

        ### COMPUTE BL COORDINATE DERIVATIVES ###
        xBL_SF = [last(rr), last(θθ), last(ϕϕ)];
        vBL_SF = [last(r_dot), last(θ_dot), last(ϕ_dot)];
        aBL_SF = [last(r_ddot), last(θ_ddot), last(ϕ_ddot)];

        AnalyticCoordinateDerivs.ComputeDerivs!(xBL_SF, sign(vBL_SF[1]), sign(vBL_SF[2]), dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,
        dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, d8x_dλ, a, E_t, L_t, C_t)

        ### COMPUTE HARMONIC COORDINATE DERIVATIVES ###
        HarmonicCoordDerivs.compute_harmonic_derivs!(xBL_SF, dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,
        xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, a)

        ###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######

        # AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_WF!(rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Mij2_data, Mijk3_wf_temp,
        # Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp, a, q, E_t, L_t, C_t)

        # store multipole data for waveforms — note that we only save the independent components
        @inbounds Threads.@threads for indices in SymmetricTensors.waveform_indices
            if length(indices)==2
                i1, i2 = indices
                append!(Mij2_wf[i1, i2], Mij2_data[i1, i2])
                append!(Sij2_wf[i1, i2], Sij2_wf_temp[i1, i2])
            elseif length(indices)==3
                i1, i2, i3 = indices
                append!(Mijk3_wf[i1, i2, i3], Mijk3_wf_temp[i1, i2, i3])
                append!(Sijk3_wf[i1, i2, i3], Sijk3_wf_temp[i1, i2, i3])
            else
                i1, i2, i3, i4 = indices
                append!(Mijkl4_wf[i1, i2, i3, i4], Mijkl4_wf_temp[i1, i2, i3, i4])
            end
        end

        ###### COMPUTE MULTIPOLE MOMENTS FOR SELF FORCE ######
        AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_SF!(xBL_SF, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, q, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)

        ##### COMPUTE SELF-FORCE #####
        xH_SF = HarmonicCoords.xBLtoH(xBL_SF, a);;
        vH_SF = HarmonicCoords.vBLtoH(xH_SF, vBL_SF, a); vH_SF = vH_SF;
        v_SF = SelfAcceleration.norm_3d(vH_SF);;
        rH_SF = SelfAcceleration.norm_3d(xH_SF);

        SelfAcceleration.aRRα(aSF_H_temp, aSF_BL_temp, 0.0, xH_SF, v_SF, vH_SF, vH_SF, xBL_SF, rH_SF, a,
            Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6, Γαμν, g_μν, g_tt, g_tϕ, g_rr, g_θθ, g_ϕϕ, gTT, gTΦ, gRR, gThTh, gΦΦ)
        

        
        Δt = last(tt) - tt[1]
        EvolveConstants.Evolve_BL(Δt, a, last(tt), last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot),
        aSF_BL_temp, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin, nPointsGeodesic)
        push!(t_Fluxes, last(tt))

        # store self force values
        push!(aSF_H, aSF_H_temp)
        push!(aSF_BL, aSF_BL_temp)
    end
    print("Completion: 100%   \r")

    # delete final "extra" energies and fluxes
    pop!(EE)
    pop!(LL)
    pop!(QQ)
    pop!(CC)
    pop!(pArray)
    pop!(ecc)
    pop!(θminArray)

    pop!(Edot)
    pop!(Ldot)
    pop!(Qdot)
    pop!(Cdot)
    pop!(t_Fluxes)

    # save data 
    mkpath(data_path)
    # matrix of SF values- rows are components, columns are component values at different times
    aSF_H = hcat(aSF_H...)
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end

    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.jld2"
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    # save params
    constants = (t_Fluxes, EE, LL, QQ, CC, pArray, ecc, θminArray)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, reltol::Float64, data_path::String)
    # load ODE solution
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    sol = readdlm(ODE_filename)
    λ=sol[1,:]; t=sol[2,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt2=sol[9,:]; d2θ_dt2=sol[10,:]; d2ϕ_dt2=sol[11,:]; dt_dτ=sol[12,:]; dt_dλ=sol[13,:]
    return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ
end


function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, reltol::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    constants=readdlm(constants_filename)
    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    constants_derivs = readdlm(constants_derivs_filename)
    t_Fluxes, EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :], constants[8, :]
    Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, t::AbstractVector{Float64}, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, 
    reltol::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.jld2"
    waveform_data = load(waveform_filename)["data"]
    Mij2 = waveform_data["Mij2"]; SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij2);
    Mijk3 = waveform_data["Mijk3"]; SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3);
    Mijkl4 = waveform_data["Mijkl4"]; SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    Sij2 = waveform_data["Sij2"]; SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2);
    Sijk3 = waveform_data["Sijk3"]; SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);

    # compute h_{ij} tensor
    num_points = length(t);
    hij = [zeros(num_points) for i=1:3, j=1:3];
    Waveform.hij!(hij, num_points, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

    # project h_{ij} tensor
    h_plus, h_cross = Waveform.h_plus_cross(hij, Θ, Φ);
    return h_plus, h_cross

end

function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, reltol::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(constants_filename)

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(constants_derivs_filename)

    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(ODE_filename)

    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(SF_filename)
    
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(SF_filename)

    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin, sigdigits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.jld2"
    rm(waveform_filename)
end

end
end