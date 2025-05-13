module AnalyticInspiral
using HDF5
import ..HDF5Helper: create_file_group!, create_dataset!, append_data!
using ..SymmetricTensors

Z_1(a::Float64) = 1 + (1 - a^2 / 1.0)^(1/3) * ((1 + a)^(1/3) + (1 - a)^(1/3))
Z_2(a::Float64) = sqrt(3 * a^2 + Z_1(a)^2)
LSO_r(a::Float64) = (3 + Z_2(a) - sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # retrograde LSO
LSO_p(a::Float64) = (3 + Z_2(a) + sqrt((3 - Z_1(a)) * (3 + Z_1(a) * 2 * Z_2(a))))   # prograde LSO

function initialize_solution_file!(file::HDF5.File, chunk_size::Int64; Mino::Bool=false)
    ## ΤRAJECTORY ##
    traj_group_name = "Trajectory"
    create_file_group!(file, traj_group_name);
    
    if Mino
        create_dataset!(file, traj_group_name, "lambda", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "dt_dlambda", Float64, chunk_size);
    end

    create_dataset!(file, traj_group_name, "t", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "r", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "theta", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "phi", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "r_dot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "theta_dot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "phi_dot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "r_ddot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "theta_ddot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "phi_ddot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "Gamma", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "t_Fluxes", Float64, 1);
    create_dataset!(file, traj_group_name, "Energy", Float64, 1);
    create_dataset!(file, traj_group_name, "AngularMomentum", Float64, 1);
    create_dataset!(file, traj_group_name, "CarterConstant", Float64, 1);
    create_dataset!(file, traj_group_name, "AltCarterConstant", Float64, 1);
    create_dataset!(file, traj_group_name, "p", Float64, 1);
    create_dataset!(file, traj_group_name, "eccentricity", Float64, 1);
    create_dataset!(file, traj_group_name, "theta_min", Float64, 1);
    create_dataset!(file, traj_group_name, "Edot", Float64, 1);
    create_dataset!(file, traj_group_name, "Ldot", Float64, 1);
    create_dataset!(file, traj_group_name, "Qdot", Float64, 1);
    create_dataset!(file, traj_group_name, "Cdot", Float64, 1);

    create_dataset!(file, traj_group_name, "self_acc_BL_t", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_BL_r", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_BL_θ", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_BL_ϕ", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_Harm_t", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_Harm_x", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_Harm_y", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_Harm_z", Float64, 1);

    ## INDEPENDENT WAVEFORM MOMENT COMPONENTS ##
    wave_group_name = "WaveformMoments"
    create_file_group!(file, wave_group_name);

    # mass and current quadrupole second time derivs
    create_dataset!(file, wave_group_name, "Mij11_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mij12_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Mij13_2", Float64, chunk_size);
    create_dataset!(file, wave_group_name, "Mij22_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mij23_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Mij33_2", Float64, chunk_size);

    create_dataset!(file, wave_group_name, "Sij11_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sij12_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Sij13_2", Float64, chunk_size);
    create_dataset!(file, wave_group_name, "Sij22_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sij23_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Sij33_2", Float64, chunk_size);

    # mass and current octupole third time derivs
    create_dataset!(file, wave_group_name, "Mijk111_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk112_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk122_3", Float64, chunk_size); 
    create_dataset!(file, wave_group_name, "Mijk113_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk133_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk123_3", Float64, chunk_size); 
    create_dataset!(file, wave_group_name, "Mijk222_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk223_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijk233_3", Float64, chunk_size); 
    create_dataset!(file, wave_group_name, "Mijk333_3", Float64, chunk_size);

    create_dataset!(file, wave_group_name, "Sijk111_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk112_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk122_3", Float64, chunk_size); 
    create_dataset!(file, wave_group_name, "Sijk113_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk133_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk123_3", Float64, chunk_size); 
    create_dataset!(file, wave_group_name, "Sijk222_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk223_3", Float64, chunk_size); create_dataset!(file, wave_group_name, "Sijk233_3", Float64, chunk_size); 
    create_dataset!(file, wave_group_name, "Sijk333_3", Float64, chunk_size);

    # mass hexadecapole fourth time deriv
    create_dataset!(file, wave_group_name, "Mijkl1111_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1112_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1122_4", Float64, chunk_size);
    create_dataset!(file, wave_group_name, "Mijkl1222_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1113_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1133_4", Float64, chunk_size);
    create_dataset!(file, wave_group_name, "Mijkl1333_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1123_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl1223_4", Float64, chunk_size);
    create_dataset!(file, wave_group_name, "Mijkl1233_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl2222_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl2223_4", Float64, chunk_size);
    create_dataset!(file, wave_group_name, "Mijkl2233_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl2333_4", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mijkl3333_4", Float64, chunk_size);
    return file
end

function initialize_solution_file_quad!(file::HDF5.File, chunk_size::Int64; Mino::Bool=false)
    ## ΤRAJECTORY ##
    traj_group_name = "Trajectory"
    create_file_group!(file, traj_group_name);
    
    if Mino
        create_dataset!(file, traj_group_name, "lambda", Float64, chunk_size);
        create_dataset!(file, traj_group_name, "dt_dlambda", Float64, chunk_size);
    end

    create_dataset!(file, traj_group_name, "t", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "r", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "theta", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "phi", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "r_dot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "theta_dot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "phi_dot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "r_ddot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "theta_ddot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "phi_ddot", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "Gamma", Float64, chunk_size);
    create_dataset!(file, traj_group_name, "t_Fluxes", Float64, 1);
    create_dataset!(file, traj_group_name, "Energy", Float64, 1);
    create_dataset!(file, traj_group_name, "AngularMomentum", Float64, 1);
    create_dataset!(file, traj_group_name, "CarterConstant", Float64, 1);
    create_dataset!(file, traj_group_name, "AltCarterConstant", Float64, 1);
    create_dataset!(file, traj_group_name, "p", Float64, 1);
    create_dataset!(file, traj_group_name, "eccentricity", Float64, 1);
    create_dataset!(file, traj_group_name, "theta_min", Float64, 1);
    create_dataset!(file, traj_group_name, "Edot", Float64, 1);
    create_dataset!(file, traj_group_name, "Ldot", Float64, 1);
    create_dataset!(file, traj_group_name, "Qdot", Float64, 1);
    create_dataset!(file, traj_group_name, "Cdot", Float64, 1);

    create_dataset!(file, traj_group_name, "self_acc_BL_t", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_BL_r", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_BL_θ", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_BL_ϕ", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_Harm_t", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_Harm_x", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_Harm_y", Float64, 1);
    create_dataset!(file, traj_group_name, "self_acc_Harm_z", Float64, 1);

    ## INDEPENDENT WAVEFORM MOMENT COMPONENTS ##
    wave_group_name = "WaveformMoments"
    create_file_group!(file, wave_group_name);

    # mass and current quadrupole second time derivs
    create_dataset!(file, wave_group_name, "Mij11_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mij12_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Mij13_2", Float64, chunk_size);
    create_dataset!(file, wave_group_name, "Mij22_2", Float64, chunk_size); create_dataset!(file, wave_group_name, "Mij23_2", Float64, chunk_size);  create_dataset!(file, wave_group_name, "Mij33_2", Float64, chunk_size);

    return file
end

function save_traj!(file::HDF5.File, chunk_size::Int64, t::Vector{Float64}, r::Vector{Float64}, θ::Vector{Float64}, ϕ::Vector{Float64}, dr_dt::Vector{Float64}, dθ_dt::Vector{Float64}, dϕ_dt::Vector{Float64}, d2r_dt2::Vector{Float64}, d2θ_dt2::Vector{Float64}, d2ϕ_dt2::Vector{Float64}, dt_dτ::Vector{Float64})

    traj_group_name = "Trajectory"
    append_data!(file, traj_group_name, "t", t[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "r", r[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "theta", θ[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "phi", ϕ[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "r_dot", dr_dt[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "theta_dot", dθ_dt[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "phi_dot", dϕ_dt[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "r_ddot", d2r_dt2[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "theta_ddot", d2θ_dt2[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "phi_ddot", d2ϕ_dt2[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "Gamma", dt_dτ[1:chunk_size], chunk_size);
end

function save_λ_traj!(file::HDF5.File, chunk_size::Int64, λ::Vector{Float64}, dt_dλ::Vector{Float64})

    traj_group_name = "Trajectory"
    append_data!(file, traj_group_name, "lambda", λ[1:chunk_size], chunk_size);
    append_data!(file, traj_group_name, "dt_dlambda", dt_dλ[1:chunk_size], chunk_size);
end

function save_moments!(file::HDF5.File, chunk_size::Int64, Mij2::AbstractArray, Sij2::AbstractArray, Mijk3::AbstractArray, Sijk3::AbstractArray, Mijkl4::AbstractArray)

    ## INDEPENDENT WAVEFORM MOMENT COMPONENTS ##
    wave_group_name = "WaveformMoments"
    # mass and current quadrupole second time derivs
    append_data!(file, wave_group_name, "Mij11_2", Mij2[1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mij12_2", Mij2[1,2][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Mij13_2", Mij2[1,3][1:chunk_size], chunk_size);
    append_data!(file, wave_group_name, "Mij22_2", Mij2[2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mij23_2", Mij2[2,3][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Mij33_2", Mij2[3,3][1:chunk_size], chunk_size);

    append_data!(file, wave_group_name, "Sij11_2", Sij2[1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sij12_2", Sij2[1,2][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Sij13_2", Sij2[1,3][1:chunk_size], chunk_size);
    append_data!(file, wave_group_name, "Sij22_2", Sij2[2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sij23_2", Sij2[2,3][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Sij33_2", Sij2[3,3][1:chunk_size], chunk_size);

    # mass and current octupole third time derivs
    append_data!(file, wave_group_name, "Mijk111_3", Mijk3[1,1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk112_3", Mijk3[1,1,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk122_3", Mijk3[1,2,2][1:chunk_size], chunk_size); 
    append_data!(file, wave_group_name, "Mijk113_3", Mijk3[1,1,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk133_3", Mijk3[1,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk123_3", Mijk3[1,2,3][1:chunk_size], chunk_size); 
    append_data!(file, wave_group_name, "Mijk222_3", Mijk3[2,2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk223_3", Mijk3[2,2,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijk233_3", Mijk3[2,3,3][1:chunk_size], chunk_size); 
    append_data!(file, wave_group_name, "Mijk333_3", Mijk3[3,3,3][1:chunk_size], chunk_size);

    append_data!(file, wave_group_name, "Sijk111_3", Sijk3[1,1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk112_3", Sijk3[1,1,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk122_3", Sijk3[1,2,2][1:chunk_size], chunk_size); 
    append_data!(file, wave_group_name, "Sijk113_3", Sijk3[1,1,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk133_3", Sijk3[1,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk123_3", Sijk3[1,2,3][1:chunk_size], chunk_size); 
    append_data!(file, wave_group_name, "Sijk222_3", Sijk3[2,2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk223_3", Sijk3[2,2,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Sijk233_3", Sijk3[2,3,3][1:chunk_size], chunk_size); 
    append_data!(file, wave_group_name, "Sijk333_3", Sijk3[3,3,3][1:chunk_size], chunk_size);

    # mass hexadecapole fourth time deriv
    append_data!(file, wave_group_name, "Mijkl1111_4", Mijkl4[1,1,1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1112_4", Mijkl4[1,1,1,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1122_4", Mijkl4[1,1,2,2][1:chunk_size], chunk_size);
    append_data!(file, wave_group_name, "Mijkl1222_4", Mijkl4[1,2,2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1113_4", Mijkl4[1,1,1,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1133_4", Mijkl4[1,1,3,3][1:chunk_size], chunk_size);
    append_data!(file, wave_group_name, "Mijkl1333_4", Mijkl4[1,3,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1123_4", Mijkl4[1,1,2,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl1223_4", Mijkl4[1,2,2,3][1:chunk_size], chunk_size);
    append_data!(file, wave_group_name, "Mijkl1233_4", Mijkl4[1,2,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl2222_4", Mijkl4[2,2,2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl2223_4", Mijkl4[2,2,2,3][1:chunk_size], chunk_size);
    append_data!(file, wave_group_name, "Mijkl2233_4", Mijkl4[2,2,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl2333_4", Mijkl4[2,3,3,3][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mijkl3333_4", Mijkl4[3,3,3,3][1:chunk_size], chunk_size);
end

function save_moments_quad!(file::HDF5.File, chunk_size::Int64, Mij2::AbstractArray)

    ## INDEPENDENT WAVEFORM MOMENT COMPONENTS ##
    wave_group_name = "WaveformMoments"
    # mass and current quadrupole second time derivs
    append_data!(file, wave_group_name, "Mij11_2", Mij2[1,1][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mij12_2", Mij2[1,2][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Mij13_2", Mij2[1,3][1:chunk_size], chunk_size);
    append_data!(file, wave_group_name, "Mij22_2", Mij2[2,2][1:chunk_size], chunk_size); append_data!(file, wave_group_name, "Mij23_2", Mij2[2,3][1:chunk_size], chunk_size);  append_data!(file, wave_group_name, "Mij33_2", Mij2[3,3][1:chunk_size], chunk_size);
end

function save_constants!(file::HDF5.File, t::Float64, E::Float64, dE_dt::Float64, L::Float64, dL_dt::Float64, Q::Float64, dQ_dt::Float64, C::Float64, dC_dt::Float64, p::Float64, e::Float64, θmin::Float64)
    traj_group_name = "Trajectory"
    append_data!(file, traj_group_name, "t_Fluxes", t);
    append_data!(file, traj_group_name, "Energy", E);
    append_data!(file, traj_group_name, "AngularMomentum", L);
    append_data!(file, traj_group_name, "CarterConstant", C);
    append_data!(file, traj_group_name, "AltCarterConstant", Q);
    append_data!(file, traj_group_name, "p", p);
    append_data!(file, traj_group_name, "eccentricity", e);
    append_data!(file, traj_group_name, "theta_min", θmin);
    append_data!(file, traj_group_name, "Edot", dE_dt);
    append_data!(file, traj_group_name, "Ldot", dL_dt);
    append_data!(file, traj_group_name, "Qdot", dQ_dt);
    append_data!(file, traj_group_name, "Cdot", dC_dt);
end

function save_self_acceleration!(file::HDF5.File, acc_BL::Vector{Float64}, acc_Harm::Vector{Float64})
    traj_group_name = "Trajectory"
    append_data!(file, traj_group_name, "self_acc_BL_t", acc_BL[1]);
    append_data!(file, traj_group_name, "self_acc_BL_r", acc_BL[2]);
    append_data!(file, traj_group_name, "self_acc_BL_θ", acc_BL[3]);
    append_data!(file, traj_group_name, "self_acc_BL_ϕ", acc_BL[4]);
    append_data!(file, traj_group_name, "self_acc_Harm_t", acc_Harm[1]);
    append_data!(file, traj_group_name, "self_acc_Harm_x", acc_Harm[2]);
    append_data!(file, traj_group_name, "self_acc_Harm_y", acc_Harm[3]);
    append_data!(file, traj_group_name, "self_acc_Harm_z", acc_Harm[4]);
end

function load_waveform_moments(sol_filename::String)
    Mij2 = [Float64[] for i=1:3, j=1:3]
    Sij2 = [Float64[] for i=1:3, j=1:3]
    Mijk3 = [Float64[] for i=1:3, j=1:3, k=1:3]
    Sijk3 = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mijkl4 = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
   
    h5f = h5open(sol_filename, "r")

    # mass and current quadrupole second time derivs
    Mij2[1,1] = h5f["WaveformMoments/Mij11_2"][:];
    Mij2[1,2] = h5f["WaveformMoments/Mij12_2"][:];
    Mij2[1,3] = h5f["WaveformMoments/Mij13_2"][:];
    Mij2[2,2] = h5f["WaveformMoments/Mij22_2"][:];
    Mij2[2,3] = h5f["WaveformMoments/Mij23_2"][:];
    Mij2[3,3] = h5f["WaveformMoments/Mij33_2"][:];

    Sij2[1,1] = h5f["WaveformMoments/Sij11_2"][:];
    Sij2[1,2] = h5f["WaveformMoments/Sij12_2"][:];
    Sij2[1,3] = h5f["WaveformMoments/Sij13_2"][:];
    Sij2[2,2] = h5f["WaveformMoments/Sij22_2"][:];
    Sij2[2,3] = h5f["WaveformMoments/Sij23_2"][:];
    Sij2[3,3] = h5f["WaveformMoments/Sij33_2"][:];


    # mass and current octupole third time derivs
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


    # mass hexadecapole fourth time deriv
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

    t = h5f["Trajectory/t"][:];
    close(h5f)

    # symmetrize 
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij2);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3);
    SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);
    
    return t, Mij2, Mijk3, Mijkl4, Sij2, Sijk3
end

function load_waveform_moments_quad(sol_filename::String)
    Mij2 = [Float64[] for i=1:3, j=1:3]
    Sij2 = [Float64[] for i=1:3, j=1:3]
    Mijk3 = [Float64[] for i=1:3, j=1:3, k=1:3]
    Sijk3 = [Float64[] for i=1:3, j=1:3, k=1:3]
    Mijkl4 = [Float64[] for i=1:3, j=1:3, k=1:3, l=1:3]
   
    h5f = h5open(sol_filename, "r")

    # mass and current quadrupole second time derivs
    Mij2[1,1] = h5f["WaveformMoments/Mij11_2"][:];
    Mij2[1,2] = h5f["WaveformMoments/Mij12_2"][:];
    Mij2[1,3] = h5f["WaveformMoments/Mij13_2"][:];
    Mij2[2,2] = h5f["WaveformMoments/Mij22_2"][:];
    Mij2[2,3] = h5f["WaveformMoments/Mij23_2"][:];
    Mij2[3,3] = h5f["WaveformMoments/Mij33_2"][:];

    t = h5f["Trajectory/t"][:];
    close(h5f)

    # symmetrize 
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij2);
    
    return t, Mij2
end


function load_trajectory(sol_filename::String; Mino::Bool=false)
    h5f = h5open(sol_filename, "r")
    t = h5f["Trajectory/t"][:]
    r = h5f["Trajectory/r"][:]
    θ = h5f["Trajectory/theta"][:]
    ϕ = h5f["Trajectory/phi"][:]
    dr_dt = h5f["Trajectory/r_dot"][:]
    dθ_dt = h5f["Trajectory/theta_dot"][:]
    dϕ_dt = h5f["Trajectory/phi_dot"][:]
    d2r_dt2 = h5f["Trajectory/r_ddot"][:]
    d2θ_dt2 = h5f["Trajectory/theta_ddot"][:]
    d2ϕ_dt2 = h5f["Trajectory/phi_ddot"][:]
    dt_dτ = h5f["Trajectory/Gamma"][:]
    if Mino
        λ = h5f["Trajectory/lambda"][:]
        dt_dλ = h5f["Trajectory/dt_dlambda"][:]
        close(h5f)
        return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ
    else
        close(h5f)
        return t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ
    end
end

function load_constants_of_motion(sol_filename::String)
    h5f = h5open(sol_filename, "r")
    t_Fluxes = h5f["Trajectory/t_Fluxes"][:]
    EE = h5f["Trajectory/Energy"][:]
    LL = h5f["Trajectory/AngularMomentum"][:]
    CC = h5f["Trajectory/CarterConstant"][:]
    QQ = h5f["Trajectory/AltCarterConstant"][:]
    pArray = h5f["Trajectory/p"][:]
    ecc = h5f["Trajectory/eccentricity"][:]
    θminArray = h5f["Trajectory/theta_min"][:]
    Edot = h5f["Trajectory/Edot"][:]
    Ldot = h5f["Trajectory/Ldot"][:]
    Qdot = h5f["Trajectory/Qdot"][:]
    Cdot = h5f["Trajectory/Cdot"][:]
    close(h5f)
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θminArray
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
using ...ChimeraInspiral


function compute_inspiral(a::Float64, p::Float64, e::Float64, θmin::Float64, sign_Lz::Int64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, nPointsGeodesic::Int64,
    compute_SF::Float64, tInspiral::Float64, reltol::Float64=1e-14, abstol::Float64=1e-14; data_path::String="Data/", mass_quad::Bool=false)

    # create solution file
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, data_path; mass_quad=mass_quad)
    
    if isfile(sol_filename)
        rm(sol_filename)
    end

    file = h5open(sol_filename, "w")

    # second argument is chunk_size. Since each successive geodesic piece overlap at the end of the first and bgeinning of the second, we must manually save this point only once to avoid repeats in the data 
    if mass_quad
        ChimeraInspiral.initialize_solution_file_quad!(file, nPointsGeodesic-1; Mino=true)
    else
        ChimeraInspiral.initialize_solution_file!(file, nPointsGeodesic-1; Mino=true)
    end

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

    rLSO = ChimeraInspiral.LSO_p(a)

    use_custom_ics = true; use_specified_params = true;
    save_at_trajectory = compute_SF / (nPointsGeodesic - 1); Δti=save_at_trajectory;    # initial time step for geodesic integration

    geodesic_time_length = compute_SF;
    num_points_geodesic = nPointsGeodesic;

    # initialize fluxes to zero (serves as a flag in the function which evolves the constants of motion)
    dE_dt = 0.0
    dL_dt = 0.0
    dQ_dt = 0.0
    dC_dt = 0.0

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
        tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi = BLTimeGeodesics.compute_kerr_geodesic(a, p_t, e_t, θmin_t, sign_Lz, num_points_geodesic, use_specified_params, geodesic_time_length, Δti, reltol, abstol;
        ics=geodesic_ics, E=E_t, L=L_t, Q=Q_t, C=C_t, ra=ra, p3=p3,p4=p4, zp=zp, zm=zm, save_to_file=false)

        tt = tt .+ t0   # tt from the above function call starts from zero

        # check that geodesic output is as expected
        if (length(tt) != num_points_geodesic) || !isapprox(tt[nPointsGeodesic], t0 + compute_SF)
            println("Integration terminated at t = $(first(tt))")
            println("total_num_points - len(sol) = $(num_points_geodesic-length(tt))")
            println("tt[nPointsGeodesic] = $(tt[nPointsGeodesic])")
            println("t0 + compute_SF = $(t0 + compute_SF)")
            break
        end

        # extract initial conditions for next geodesic.
        t0 = last(tt); geodesic_ics = @SArray [last(psi), last(chi), last(ϕϕ)];
        
        # store physical trajectory — this function omits last point since this overlaps with the start of the next geodesic
        ChimeraInspiral.save_traj!(file, nPointsGeodesic-1, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ)

        ###### COMPUTE MULTIPOLE MOMENTS FOR WAVEFORMS ######
        if mass_quad
            AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_WF!(rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Mij2_data, a, q, E_t, L_t, C_t);
        else
            AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_WF!(rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Mij2_data, Mijk3_wf_temp,
            Mijkl4_wf_temp, Sij2_wf_temp, Sijk3_wf_temp, a, q, E_t, L_t, C_t);
        end

        # store multipole data for waveforms — note that we only save the independent components
        if mass_quad
            ChimeraInspiral.save_moments_quad!(file, nPointsGeodesic-1, Mij2_data)
        else
            ChimeraInspiral.save_moments!(file, nPointsGeodesic-1, Mij2_data, Sij2_wf_temp, Mijk3_wf_temp, Sijk3_wf_temp, Mijkl4_wf_temp)
        end 

        ###### COMPUTE MULTIPOLE MOMENTS FOR SELF FORCE ######
        ### COMPUTE BL COORDINATE DERIVATIVES ###
        xBL_SF = [last(rr), last(θθ), last(ϕϕ)];
        vBL_SF = [last(r_dot), last(θ_dot), last(ϕ_dot)];
        aBL_SF = [last(r_ddot), last(θ_ddot), last(ϕ_ddot)];

        AnalyticCoordinateDerivs.ComputeDerivs!(xBL_SF, sign(vBL_SF[1]), sign(vBL_SF[2]), dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,
        dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, d8x_dλ, a, E_t, L_t, C_t);

        # COMPUTE HARMONIC COORDINATE DERIVATIVES
        HarmonicCoordDerivs.compute_harmonic_derivs!(xBL_SF, dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,
        xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, a);

        # COMPUTE MULTIPOLE DERIVATIVES
        if mass_quad
            AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_SF!(xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, q, Mij5, Mij6, Mij7, Mij8);
        else
            AnalyticMultipoleDerivs.AnalyticMultipoleDerivs_SF!(xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, q, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6);
        end
   
        ##### COMPUTE SELF-FORCE #####
        HarmonicCoords.xBLtoH!(xH_SF, xBL_SF, a);
        HarmonicCoords.vBLtoH!(vH_SF, xH_SF, vBL_SF, a); 
        HarmonicCoords.aBLtoH!(aH_SF, xH_SF, vBL_SF, aBL_SF, a);
        rH_SF = SelfAcceleration.norm_3d(xH_SF);
        v_SF = SelfAcceleration.norm_3d(vH_SF);

        SelfAcceleration.aRRα(aSF_H_temp, aSF_BL_temp, xH_SF, v_SF, vH_SF, xBL_SF, rH_SF, a, Mij5, Mij6, Mij7, Mij8, Mijk7, Mijk8, Sij5, Sij6)

        # store self force values
        ChimeraInspiral.save_self_acceleration!(file, aSF_BL_temp, aSF_H_temp)

        # update orbital constants and fluxes — function takes as argument the fluxes computed at the end of the previous geodesic (which overlaps with the start of the current geodesic piece) in order to update the fluxes using the trapezium rule
        Δt = last(tt) - tt[1]
        E_1, dE_dt, L_1, dL_dt, Q_1, dQ_dt, C_1, dC_dt, p_1, e_1, θmin_1 = EvolveConstants.Evolve_BL(Δt, a, last(rr), last(θθ), last(ϕϕ), last(Γ), last(r_dot), last(θ_dot), last(ϕ_dot), aSF_BL_temp, E_t, dE_dt, L_t, dL_dt, Q_t, dQ_dt, C_t, dC_dt, p_t, e_t, θmin_t)

        # save orbital constants and fluxes
        ChimeraInspiral.save_constants!(file, last(tt), E_t, dE_dt, L_t, dL_dt, Q_t, dQ_dt, C_t, dC_dt, p_t, e_t, θmin_t)

        E_t = E_1; L_t = L_1; Q_t = Q_1; C_t = C_1; p_t = p_1; e_t = e_1; θmin_t = θmin_1;
        # flush(file)
    end
    print("Completion: 100%   \r")

    print("Completion: 100%   \r")
    println("File created: " * sol_filename)
    close(file)
end

function solution_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, data_path::String; mass_quad::Bool=false)
    return data_path * "EMRI_sol_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_BL_time_mass_quad_$mass_quad.h5"
end

function waveform_fname(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, data_path::String; mass_quad::Bool=false)
    return data_path * "Waveform_a_$(a)_p_$(p)_e_$(e)_θmin_$(round(θmin; digits=3))_q_$(q)_psi0_$(round(psi_0; digits=3))_chi0_$(round(chi_0; digits=3))_phi0_$(round(phi_0; digits=3))_obsDist_$(round(obs_distance; digits=3))_ThetaS_$(round(ThetaSource; digits=3))_PhiS_$(round(PhiSource; digits=3))_ThetaK_$(round(ThetaKerr; digits=3))_PhiK_$(round(PhiKerr; digits=3))_BL_time_mass_quad_$mass_quad.h5"
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, data_path::String; mass_quad::Bool=false)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, data_path; mass_quad=mass_quad)
    return ChimeraInspiral.load_trajectory(sol_filename, Mino=false)
end

function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, data_path::String; mass_quad::Bool=false)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, data_path; mass_quad=mass_quad)
    return ChimeraInspiral.load_constants_of_motion(sol_filename)
end

function compute_waveform(obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, data_path::String; mass_quad::Bool=false)
    # load waveform multipole moments
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, data_path; mass_quad=mass_quad)
    if mass_quad
        t, Mij2 = ChimeraInspiral.load_waveform_moments_quad(sol_filename)
        num_points = length(Mij2[1, 1]);
        h_plus, h_cross = Waveform.compute_wave_polarizations(num_points, obs_distance, ThetaSource, PhiSource, ThetaKerr, PhiKerr, Mij2, q)
    else
        t, Mij2, Mijk3, Mijkl4, Sij2, Sijk3 = ChimeraInspiral.load_waveform_moments(sol_filename)
        num_points = length(Mij2[1, 1]);
        h_plus, h_cross = Waveform.compute_wave_polarizations(num_points, obs_distance, ThetaSource, PhiSource, ThetaKerr, PhiKerr, Mij2, Mijk3, Mijkl4, Sij2, Sijk3, q)
    end
    
    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaSource, PhiSource, ThetaKerr, PhiKerr, data_path; mass_quad=mass_quad)
    h5open(wave_filename, "w") do file
        file["t"] = t
        file["hplus"] = h_plus
        file["hcross"] = h_cross
    end
    println("File created: " * wave_filename)
end

function load_waveform(obs_distance::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, data_path::String; mass_quad::Bool=false)
    # save waveform to file
    wave_filename=waveform_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, obs_distance, ThetaSource, PhiSource, ThetaKerr, PhiKerr, data_path; mass_quad=mass_quad)
    file = h5open(wave_filename, "r")
    t = file["t"][:]
    h_plus = file["hplus"][:]
    h_cross = file["hcross"][:]
    close(file)
    return t, h_plus, h_cross    
end

# useful for dummy runs (e.g., for resonances to estimate the duration of time needed by computing the time derivative of the fundamental frequencies)
function delete_EMRI_data(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, psi_0::Float64, chi_0::Float64, phi_0::Float64, data_path::String; mass_quad::Bool=false)
    sol_filename=solution_fname(a, p, e, θmin, q, psi_0, chi_0, phi_0, data_path; mass_quad=mass_quad)
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
using ...ChimeraInspiral



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
    geodesic_ics = MinoTimeGeodesics.Mino_ics(t0, ra, p, e);

    rLSO = ChimeraInspiral.LSO_p(a)

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
        λλ, tt, rr, θθ, ϕϕ, r_dot, θ_dot, ϕ_dot, r_ddot, θ_ddot, ϕ_ddot, Γ, psi, chi, dt_dλλ = MinoTimeGeodesics.compute_kerr_geodesic(a, p_t, e_t, θmin_t, num_points_geodesic, use_custom_ics,
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
    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_H)
    end

    # matrix of SF values- rows are components, columns are component values at different times
    aSF_BL = hcat(aSF_BL...)
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(SF_filename, "w") do io
        writedlm(io, aSF_BL)
    end

    # save trajectory- rows are: τRange, t, r, θ, ϕ, tdot, rdot, θdot, ϕdot, tddot, rddot, θddot, ϕddot, columns are component values at different times
    sol = transpose(stack([λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ]))
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(ODE_filename, "w") do io
        writedlm(io, sol)
    end

    # save waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.jld2"
    waveform_dictionary = Dict{String, AbstractArray}("Mij2" => Mij2_wf, "Mijk3" => Mijk3_wf, "Mijkl4" => Mijkl4_wf, "Sij2" => Sij2_wf, "Sijk3" => Sijk3_wf)
    save(waveform_filename, "data", waveform_dictionary)

    # save params
    constants = (t_Fluxes, EE, LL, QQ, CC, pArray, ecc, θminArray)
    constants = vcat(transpose.(constants)...)
    derivs = (Edot, Ldot, Qdot, Cdot)
    derivs = vcat(transpose.(derivs)...)

    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(constants_filename, "w") do io
        writedlm(io, constants)
    end

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    open(constants_derivs_filename, "w") do io
        writedlm(io, derivs)
    end
end

function load_trajectory(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, reltol::Float64, data_path::String)
    # load ODE solution
    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    sol = readdlm(ODE_filename)
    λ=sol[1,:]; t=sol[2,:]; r=sol[3,:]; θ=sol[4,:]; ϕ=sol[5,:]; dr_dt=sol[6,:]; dθ_dt=sol[7,:]; dϕ_dt=sol[8,:]; d2r_dt2=sol[9,:]; d2θ_dt2=sol[10,:]; d2ϕ_dt2=sol[11,:]; dt_dτ=sol[12,:]; dt_dλ=sol[13,:]
    return λ, t, r, θ, ϕ, dr_dt, dθ_dt, dϕ_dt, d2r_dt2, d2θ_dt2, d2ϕ_dt2, dt_dτ, dt_dλ
end


function load_constants_of_motion(a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, reltol::Float64, data_path::String)
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    constants=readdlm(constants_filename)
    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    constants_derivs = readdlm(constants_derivs_filename)
    t_Fluxes, EE, LL, QQ, CC, pArray, ecc, θmin = constants[1, :], constants[2, :], constants[3, :], constants[4, :], constants[5, :], constants[6, :], constants[7, :], constants[8, :]
    Edot, Ldot, Qdot, Cdot = constants_derivs[1, :], constants_derivs[2, :], constants_derivs[3, :], constants_derivs[4, :]
    return t_Fluxes, EE, Edot, LL, Ldot, QQ, Qdot, CC, Cdot, pArray, ecc, θmin
end

function compute_waveform(obs_distance::Float64, Θ::Float64, Φ::Float64, t::AbstractVector{Float64}, a::Float64, p::Float64, e::Float64, θmin::Float64, q::Float64, 
    reltol::Float64, data_path::String)
    # load waveform multipole moments
    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.jld2"
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
    constants_filename=data_path * "constants_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(constants_filename)

    constants_derivs_filename=data_path * "constants_derivs_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(constants_derivs_filename)

    ODE_filename=data_path * "EMRI_ODE_sol_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(ODE_filename)

    SF_filename=data_path * "aSF_H_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(SF_filename)
    
    SF_filename=data_path * "aSF_BL_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.txt"
    rm(SF_filename)

    waveform_filename=data_path * "Waveform_moments_a_$(a)_p_$(p)_e_$(e)_θi_$(round(θmin; digits=3))_q_$(q)_tol_$(reltol)_Analytic_Mino.jld2"
    rm(waveform_filename)
end

end
end