using HDF5, LaTeXStrings, Plots.PlotMeasures, Plots, JLD2
include("params.jl");

# @time ChimeraInspiral.FourierFit.BLTime.compute_waveform(obs_distance, Θ, Φ, a, p, e, θmin, q, psi0, chi0, phi0, nHarmGSL, t_range_factor_BL, gsl_fit, data_path);

# t_wf_BL, h_plus_BL, h_cross_BL = ChimeraInspiral.FourierFit.BLTime.load_waveform(obs_distance, Θ, Φ, a, p, e, θmin, q, psi0, chi0, phi0, nHarmGSL, t_range_factor_BL, gsl_fit, data_path);

# load waveform multipole moments
waveform_filename=ChimeraInspiral.FourierFit.BLTime.waveform_moments_fname(a, p, e, θmin, q, psi0, chi0, phi0, nHarmGSL, t_range_factor_BL, gsl_fit, "./Results/Data/")

waveform_data = load(waveform_filename)["data"];
Mij2 = waveform_data["Mij2"]; SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij2);
Mijk3 = waveform_data["Mijk3"]; SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3);
Mijkl4 = waveform_data["Mijkl4"]; SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
Sij2 = waveform_data["Sij2"]; SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2);
Sijk3 = waveform_data["Sijk3"]; SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);

# compute h_{ij} tensor
num_points = length(Mij2[1, 1]);
hij = [zeros(num_points) for i=1:3, j=1:3];
Waveform.hij!(hij, num_points, obs_distance, Θ, Φ, Mij2, Mijk3, Mijkl4, Sij2, Sijk3)

hij[1, 1][1:10]