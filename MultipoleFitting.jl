#=

    In this module we write code which estimates the high-order derivatives of the multipole moments by fitting their time series data to the fourier expansion of orbital functions in both BL coordinate time and in Mino time. For further details,
    see Sopuerta & Yunes arXiv:1109.0572v2, Drasco & Hughes arXiv:astro-ph/0308479v3. There are four functions. Two of these compute the moment derivatives using BL coordinate time by carrying out the fits with respect to coordinate time and with
    the fundamental frequencies wrt t. The other two functions compute the moment derivatives using Mino time by carrying out the fits with respect to Mino time and with the fundamental frequencies wrt λ. Note that BOTH of these functions compute
    the time derivatives of the multipole moments wrt coordinate time---the fits carried out in Mino time compute derivatives wrt λ which are then transformed to derivatives wrt t. The reason we do this is because BL coordinates r and θ are not
    truly periodic since the r and θ motions are coupled in BL time, whereas they are truly periodic wrt Mino time. Thus, from a theoretical perspective, one might expect the fits in Mino time to be more accurate. Unfortunately, we did not find
    this to be the case generally, although it did turn out the fits are faster wrt Mino time than BL time. For both time parameters, there are two functions: one which estimates the multipole derivatives at a single point in order to compute
    the self-acceleration there, and the other which computes the moments at every point along the piecewise geodesic in order to compute the waveform.

=#

module MultipoleFitting
using StaticArrays
using ...MinoTimeDerivs
using ...MinoDerivs1
using ...MinoDerivs2
using ...MinoDerivs3
using ...MinoDerivs4
using ...MinoDerivs5
using ...MinoDerivs6
using ...HarmonicCoords
using ...FourierFitGSL
using ...FourierFitJuliaBase
using ...ParameterizedDerivs
using ...SymmetricTensors

"""
# Common Arguments in this module
- `r::Float64`: Boyer-Lindquist radial coordinate.
- `θ::Float64`: Boyer-Lindquist polar coordinate.
- `ϕ::Float64`: Boyer-Lindquist azimuthal coordinate.
- `xBL::AbstractArray`: array of arrays of Boyer-Lindquist coordinates, xBL = [[r[1], θ[1], ϕ[1]], [r[2], θ[2], ϕ[2]],....] at each time the multipole derivatives are to be approximated.
- `rH::Float64`: rH = sqrt(xH^2 + yH^2 + zH^2).
- `dr_dt::Float64`: Coordinate-time first derivative of the radial coordinate.
- `sign_dr::Float64`: sign of the dr_dt.
- `sign_dθ::Float64`: sign of the dθ_dt.
- `dθ_dt::Float64`: Coordinate-time first derivative of the polar coordinate.
- `Mij2::AbstractArray`: second derivative of the mass quadrupole (Eq. 48, arXiv:1109.0572v2).
- `Mij5::AbstractArray`: fifth derivative of the mass quadrupole.
- `Mij6::AbstractArray`: sixth derivative of the mass quadrupole.
- `Mij7::AbstractArray`: seventh derivative of the mass quadrupole.
- `Mij8::AbstractArray`: eighth derivative of the mass quadrupole.
- `Mijk2::AbstractArray`: second derivative of the mass octupole (Eq. 48).
- `Mijk3::AbstractArray`: third derivative of the mass octupole.
- `Mijk7::AbstractArray`: seventh derivative of the mass octupole.
- `Mijk8::AbstractArray`: eighth derivative of the mass octupole.
- `Mijkl2::AbstractArray`: second derivative of the mass hexadecapole (Eq. 85).
- `Mijkl4::AbstractArray`: fourth derivative of the mass hexadecapole.
- `Sij1::AbstractArray`: first derivative of the current quadrupole (Eq. 49).
- `Sij5::AbstractArray`: fifth derivative of the current quadrupole.
- `Sij6::AbstractArray`: sixth derivative of the current quadrupole.
- `Sijk1::AbstractArray`: first derivative of the current octupole (Eq. 86).
- `Sijk3::AbstractArray`: third derivative of the current octupole.
- `a::Float64`: Kerr black hole spin parameter.
- `q::Float64`: mass ratio.
- `E::Float64`: energy per unit mass (Eq. 14).
- `L::Float64`: axial (i.e., z-component of the) angular momentum per unit mass (Eq. 15).
- `C::Float64`: Carter constant---note that this C is what is commonly referred to as 'Q' elsewhere (Eq. 17).
- `nPoints::Int64`: number of points at which the derivatives are to be approximated.
- `nHarm::Int64`: number of (radial) harmonic frequencies.
- `ωr::Float64`: radial frequency wrt Mino time.
- `ωθ::Float64`: polar frequency wrt Mino time.
- `ωϕ::Float64`: azimuthal frequency wrt Mino time.
- `Ωr::Float64`: radial frequency wrt BL time.
- `Ωθ::Float64`: polar frequency wrt BL time.
- `Ωϕ::Float64`: azimuthal frequency wrt BL time.
- `compute_at::Int64`: index (in the stencil) at which the derivatives are to be approximated.
- `chisq::AbstractVector{Float64}`: chi-squared value of the fit. In practice this is just the output of the GSL solver, so one only passes as argument [0.0].
- `fit::String`: type of fitting to perform. Either "GSL" for fits using Julia's GSL wrapper, or "Julia" to use Julia's base least squares solver.
"""

# multipole moments
const multipole_moments = ["MassQuad", "MassOct", "MassHex", "CurrentQuad", "CurrentOct"]

# independent components of two, three, and four index tensors
const two_index_components::Vector{Tuple{Int64, Int64}} = [(1, 2), (1, 3), (2, 3), (1, 1), (2, 2), (3, 3)];
const three_index_components::Vector{Tuple{Int64, Int64, Int64}} = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 3, 3), (1, 2, 3), (2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3)];
const four_index_components::Vector{Tuple{Int64, Int64, Int64, Int64}} = [(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 2, 2), (1, 1, 1, 3), (1, 1, 3, 3), (1, 3, 3, 3), (1, 1, 2, 3), (1, 2, 2, 3),
(1, 2, 3, 3), (2, 2, 2, 2), (2, 2, 2, 3), (2, 2, 3, 3), (2, 3, 3, 3), (3, 3, 3, 3)];

# multipole moments and their independent components
const mass_quad_moments = SVector{length(two_index_components)}(["MassQuad", indices] for indices in two_index_components)
const mass_oct_moments = SVector{length(three_index_components)}(["MassOct", indices] for indices in three_index_components)
const mass_hex_moments = SVector{length(four_index_components)}(["MassHex", indices] for indices in four_index_components)
const current_quad_moments = SVector{length(two_index_components)}(["CurrentQuad", indices] for indices in two_index_components)
const current_oct_moments = SVector{length(three_index_components)}(["CurrentOct", indices] for indices in three_index_components)

# waveform moments and trajectory (flux) moments. This construction is so that we can use threads to parallelize the computation of the derivatives.
const moments_wf = SVector(vcat(mass_oct_moments, mass_hex_moments, current_quad_moments, current_oct_moments)...)
const moments_tr = SVector(vcat(mass_quad_moments, mass_oct_moments, current_quad_moments)...)
const moments_tr_quad = SVector(vcat(mass_quad_moments)...)

# compute derivatives of the multipole moments wrt to BL time. This function is for computing the moment derivatives necessary for the fluxes. This is computed at a single point.
function fit_moments_tr_BL!(tdata::AbstractArray, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray,
    Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, compute_at::Int64, nHarm::Int64, Ωr::Float64, Ωθ::Float64, Ωϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::AbstractVector{Float64}, fit::String)
    Ω = [Ωr, Ωθ, Ωϕ];
    if fit=="GSL"
        gsl_fit = true
        julia_fit = false
    elseif fit=="Julia"
        julia_fit = true
        gsl_fit = false
    else
        throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
    end

    @inbounds Threads.@threads for multipole_moment in moments_tr
        fit_params = zeros(2 * n_freqs + 1);
        type = multipole_moment[1];

        if isequal(type, "MassQuad")
            i1, i2 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Mij2data[i1, i2], nPoints, nHarm, chisq,  Ω, fit_params)
                @views Mij5[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
                @views Mij6[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                @views Mij7[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                @views Mij8[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(tdata, Mij2data[i1, i2], nPoints, nHarm, Ω, fit_params)
                @views Mij5[i1, i2] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 3)[compute_at]
                @views Mij6[i1, i2] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 4)[compute_at]
                @views Mij7[i1, i2] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 5)[compute_at]
                @views Mij8[i1, i2] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 6)[compute_at]
            end

        elseif isequal(type, "MassOct")
            i1, i2, i3 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Mijk2data[i1, i2, i3], nPoints, nHarm, chisq,  Ω, fit_params)
                @views Mijk7[i1, i2, i3] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                @views Mijk8[i1, i2, i3] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(tdata, Mijk2data[i1, i2, i3], nPoints, nHarm, Ω, fit_params)
                @views Mijk7[i1, i2, i3] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 5)[compute_at]
                @views Mijk8[i1, i2, i3] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 6)[compute_at]
            else
                throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
            end
        elseif isequal(type, "CurrentQuad")
            i1, i2 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Sij1data[i1, i2], nPoints, nHarm, chisq,  Ω, fit_params)
                @views Sij5[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                @views Sij6[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(tdata, Sij1data[i1, i2], nPoints, nHarm, Ω, fit_params)
                @views Sij5[i1, i2] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 4)[compute_at]
                @views Sij6[i1, i2] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 5)[compute_at]
            else
                throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
            end
        end
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij5); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij6);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij7); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij8);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk7); SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk8);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij5); SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij6); 
end

# compute derivatives of the multipole moments wrt to BL time. This function is for computing the moment derivatives necessary for the waveform, and computes them at each point in the piecewise geodesic)
function fit_moments_wf_BL!(tdata::AbstractArray, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray, Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nHarm::Int64, Ωr::Float64, Ωθ::Float64,
    Ωϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::AbstractVector{Float64}, fit::String)
    Ω = [Ωr, Ωθ, Ωϕ];
    if fit=="GSL"
        gsl_fit = true
        julia_fit = false
    elseif fit=="Julia"
        julia_fit = true
        gsl_fit = false
    else
        throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
    end

    @inbounds Threads.@threads for multipole_moment in moments_wf
        fit_params = zeros(2 * n_freqs + 1);
        type = multipole_moment[1];

        if isequal(type, "MassOct")
            i1, i2, i3 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Mijk2data[i1, i2, i3], nPoints, nHarm, chisq,  Ω, fit_params)
                @views Mijk3[i1, i2, i3] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 1)
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(tdata, Mijk2data[i1, i2, i3], nPoints, nHarm, Ω, fit_params)
                @views Mijk3[i1, i2, i3] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 1)
            else
                throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
            end
        elseif isequal(type, "MassHex")
            i1, i2, i3, i4 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Mijkl2data[i1, i2, i3, i4], nPoints, nHarm, chisq,  Ω, fit_params)
                @views Mijkl4[i1, i2, i3, i4] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 2)
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(tdata, Mijkl2data[i1, i2, i3, i4], nPoints, nHarm, Ω, fit_params)
                @views Mijkl4[i1, i2, i3, i4] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 2)
            else
                throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
            end
        
        elseif isequal(type, "CurrentQuad")
            i1, i2 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Sij1data[i1, i2], nPoints, nHarm, chisq,  Ω, fit_params)
                @views Sij2[i1, i2] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 1)
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(tdata, Sij1data[i1, i2], nPoints, nHarm, Ω, fit_params)
                @views Sij2[i1, i2] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 1)
            else
                throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
            end

        elseif isequal(type, "CurrentOct")
            i1, i2, i3 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(tdata, Sijk1data[i1, i2, i3], nPoints, nHarm, chisq,  Ω, fit_params)
                @views Sijk3[i1, i2, i3] = FourierFitGSL.curve_fit_functional_derivs(tdata, Ω_fit, fit_params, n_freqs, nPoints, 2)
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(tdata, Sijk1data[i1, i2, i3], nPoints, nHarm, Ω, fit_params)
                @views Sijk3[i1, i2, i3] = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, tdata, 2)
            end
        end
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3); SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2); SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);
end

# compute derivatives of the multipole moments wrt to Mino time and converts to BL time. This function is for computing the moment derivatives necessary for the fluxes. This is computed at a single point (whereas the waveform moments are computed
# at each point in the piecewise geodesic)
function fit_moments_tr_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, λ::AbstractArray, x::AbstractArray, sign_dr::Float64, sign_dθ::Float64, Mij2data::AbstractArray, Mijk2data::AbstractArray, Sij1data::AbstractArray,
    Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray,
    compute_at::Int64, nHarm::Int64, γr::Float64, γθ::Float64, γϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::AbstractVector{Float64}, fit::String)
    γ = [γr, γθ, γϕ];
    if fit=="GSL"
        gsl_fit = true
        julia_fit = false
    elseif fit=="Julia"
        julia_fit = true
        gsl_fit = false
    else
        throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
    end

    # println("x = ", x)
    # println("a = ", a)
    # println("E = ", E)
    # println("L = ", L)
    # println("C = ", C)
    # compute derivatives of coordinates wrt to lambda
    dx_dλ = [MinoDerivs1.dr_dλ(x, a, E, L, C) * sign_dr, MinoDerivs1.dθ_dλ(x, a, E, L, C) * sign_dθ, MinoDerivs1.dϕ_dλ(x, a, E, L, C)]
    d2x_dλ = [MinoDerivs2.d2r_dλ(x, dx_dλ, a, E, L, C), MinoDerivs2.d2θ_dλ(x, dx_dλ, a, E, L, C), MinoDerivs2.d2ϕ_dλ(x, dx_dλ, a, E, L, C)]
    d3x_dλ = [MinoDerivs3.d3r_dλ(x, dx_dλ, d2x_dλ, a, E, L, C), MinoDerivs3.d3θ_dλ(x, dx_dλ, d2x_dλ, a, E, L, C), MinoDerivs3.d3ϕ_dλ(x, dx_dλ, d2x_dλ, a, E, L, C)]
    d4x_dλ = [MinoDerivs4.d4r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C), MinoDerivs4.d4θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C), MinoDerivs4.d4ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C)]
    d5x_dλ = [MinoDerivs5.d5r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C), MinoDerivs5.d5θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C), MinoDerivs5.d5ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C)]

    # compute derivatives of coordinate time wrt lambda
    dt_dλ = MinoDerivs1.dt_dλ(x, a, E, L, C);
    d2t_dλ = MinoDerivs2.d2t_dλ(x, dx_dλ, a, E, L, C);
    d3t_dλ = MinoDerivs3.d3t_dλ(x, dx_dλ, d2x_dλ, a, E, L, C);
    d4t_dλ = MinoDerivs4.d4t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C);
    d5t_dλ = MinoDerivs5.d5t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C);
    d6t_dλ = MinoDerivs6.d6t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, E, L, C);

    # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
    dλ_dt = MinoTimeDerivs.dλ_dt(dt_dλ)
    d2λ_dt = MinoTimeDerivs.d2λ_dt(dt_dλ, d2t_dλ)
    d3λ_dt = MinoTimeDerivs.d3λ_dt(dt_dλ, d2t_dλ, d3t_dλ)
    d4λ_dt = MinoTimeDerivs.d4λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ)
    d5λ_dt = MinoTimeDerivs.d5λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ)
    d6λ_dt = MinoTimeDerivs.d6λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ)

    @inbounds Threads.@threads for multipole_moment in moments_tr
        fit_params = zeros(2 * n_freqs + 1);
        type = multipole_moment[1];

        if isequal(type, "MassQuad")
            i1, i2 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mij2data[i1, i2], nPoints, nHarm, chisq,  γ, fit_params)
                df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)[compute_at]
                d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)[compute_at]
                d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
                d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(λ, Mij2data[i1, i2], nPoints, nHarm, γ, fit_params)
                df_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 1)[compute_at]
                d2f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 2)[compute_at]
                d3f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 3)[compute_at]
                d4f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 4)[compute_at]
                d5f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 5)[compute_at]
                d6f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 6)[compute_at]
            end

            d3f_dt = ParameterizedDerivs.d3f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt)
            d4f_dt = ParameterizedDerivs.d4f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt)
            d5f_dt = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
            d6f_dt = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)
            @views Mij5[i1, i2] = d3f_dt
            @views Mij6[i1, i2] = d4f_dt
            @views Mij7[i1, i2] = d5f_dt
            @views Mij8[i1, i2] = d6f_dt

        elseif isequal(type, "MassOct")
            i1, i2, i3 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mijk2data[i1, i2, i3], nPoints, nHarm, chisq,  γ, fit_params)
                df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)[compute_at]
                d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)[compute_at]
                d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
                d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(λ,  Mijk2data[i1, i2, i3], nPoints, nHarm, γ, fit_params)
                df_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 1)[compute_at]
                d2f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 2)[compute_at]
                d3f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 3)[compute_at]
                d4f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 4)[compute_at]
                d5f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 5)[compute_at]
                d6f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 6)[compute_at]
            end

            @views Mijk7[i1, i2, i3] = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
            @views Mijk8[i1, i2, i3] = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)

        elseif isequal(type, "CurrentQuad")
            i1, i2 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Sij1data[i1, i2], nPoints, nHarm, chisq,  γ, fit_params)
                df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)[compute_at]
                d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)[compute_at]
                d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
                d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
                d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
                d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(λ, Sij1data[i1, i2], nPoints, nHarm, γ, fit_params)
                df_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 1)[compute_at]
                d2f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 2)[compute_at]
                d3f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 3)[compute_at]
                d4f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 4)[compute_at]
                d5f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 5)[compute_at]
                d6f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 6)[compute_at]
            end
    
            @views Sij5[i1, i2] = ParameterizedDerivs.d4f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt)
            @views Sij6[i1, i2] = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
        end
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij5); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij6);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij7); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij8);
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk7); SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk8);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij5); SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij6);
end

function fit_moments_tr_Mino_quad!(a::Float64, E::Float64, L::Float64, C::Float64, λ::AbstractArray, x::AbstractArray, sign_dr::Float64, sign_dθ::Float64, Mij2data::AbstractArray,
    Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray,
    compute_at::Int64, nHarm::Int64, γr::Float64, γθ::Float64, γϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::AbstractVector{Float64}, fit::String)
    γ = [γr, γθ, γϕ];
    if fit=="GSL"
        gsl_fit = true
        julia_fit = false
    elseif fit=="Julia"
        julia_fit = true
        gsl_fit = false
    else
        throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
    end

    # compute derivatives of coordinates wrt to lambda
    dx_dλ = [MinoDerivs1.dr_dλ(x, a, E, L, C) * sign_dr, MinoDerivs1.dθ_dλ(x, a, E, L, C) * sign_dθ, MinoDerivs1.dϕ_dλ(x, a, E, L, C)]
    d2x_dλ = [MinoDerivs2.d2r_dλ(x, dx_dλ, a, E, L, C), MinoDerivs2.d2θ_dλ(x, dx_dλ, a, E, L, C), MinoDerivs2.d2ϕ_dλ(x, dx_dλ, a, E, L, C)]
    d3x_dλ = [MinoDerivs3.d3r_dλ(x, dx_dλ, d2x_dλ, a, E, L, C), MinoDerivs3.d3θ_dλ(x, dx_dλ, d2x_dλ, a, E, L, C), MinoDerivs3.d3ϕ_dλ(x, dx_dλ, d2x_dλ, a, E, L, C)]
    d4x_dλ = [MinoDerivs4.d4r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C), MinoDerivs4.d4θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C), MinoDerivs4.d4ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C)]
    d5x_dλ = [MinoDerivs5.d5r_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C), MinoDerivs5.d5θ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C), MinoDerivs5.d5ϕ_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C)]

    # compute derivatives of coordinate time wrt lambda
    dt_dλ = MinoDerivs1.dt_dλ(x, a, E, L, C);
    d2t_dλ = MinoDerivs2.d2t_dλ(x, dx_dλ, a, E, L, C);
    d3t_dλ = MinoDerivs3.d3t_dλ(x, dx_dλ, d2x_dλ, a, E, L, C);
    d4t_dλ = MinoDerivs4.d4t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, a, E, L, C);
    d5t_dλ = MinoDerivs5.d5t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, a, E, L, C);
    d6t_dλ = MinoDerivs6.d6t_dλ(x, dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, a, E, L, C);

    # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
    dλ_dt = MinoTimeDerivs.dλ_dt(dt_dλ)
    d2λ_dt = MinoTimeDerivs.d2λ_dt(dt_dλ, d2t_dλ)
    d3λ_dt = MinoTimeDerivs.d3λ_dt(dt_dλ, d2t_dλ, d3t_dλ)
    d4λ_dt = MinoTimeDerivs.d4λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ)
    d5λ_dt = MinoTimeDerivs.d5λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ)
    d6λ_dt = MinoTimeDerivs.d6λ_dt(dt_dλ, d2t_dλ, d3t_dλ, d4t_dλ, d5t_dλ, d6t_dλ)

    @inbounds Threads.@threads for multipole_moment in moments_tr_quad
        fit_params = zeros(2 * n_freqs + 1);

        i1, i2 = multipole_moment[2];
        if gsl_fit
            Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mij2data[i1, i2], nPoints, nHarm, chisq,  γ, fit_params)
            df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)[compute_at]
            d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)[compute_at]
            d3f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 3)[compute_at]
            d4f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 4)[compute_at]
            d5f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 5)[compute_at]
            d6f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 6)[compute_at]
        elseif julia_fit
            Ω_fit = FourierFitJuliaBase.Fit_master!(λ, Mij2data[i1, i2], nPoints, nHarm, γ, fit_params)
            df_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 1)[compute_at]
            d2f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 2)[compute_at]
            d3f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 3)[compute_at]
            d4f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 4)[compute_at]
            d5f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 5)[compute_at]
            d6f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 6)[compute_at]
        end

        d3f_dt = ParameterizedDerivs.d3f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt)
        d4f_dt = ParameterizedDerivs.d4f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt)
        d5f_dt = ParameterizedDerivs.d5f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt)
        d6f_dt = ParameterizedDerivs.d6f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt, d3f_dλ, d3λ_dt, d4f_dλ, d4λ_dt, d5f_dλ, d5λ_dt, d6f_dλ, d6λ_dt)
        @views Mij5[i1, i2] = d3f_dt
        @views Mij6[i1, i2] = d4f_dt
        @views Mij7[i1, i2] = d5f_dt
        @views Mij8[i1, i2] = d6f_dt
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij5); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij6);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij7); SymmetricTensors.SymmetrizeTwoIndexTensor!(Mij8);
end

# compute derivatives of the multipole moments wrt to Mino time and converts to BL time. This function is for computing the moment derivatives necessary for the waveform, and computes them at each point in the piecewise geodesic)
function fit_moments_wf_Mino!(a::Float64, E::Float64, L::Float64, C::Float64, λ::AbstractArray, x::AbstractArray, rdot::AbstractVector{Float64}, θdot::AbstractVector{Float64}, Mijk2data::AbstractArray, Mijkl2data::AbstractArray, Sij1data::AbstractArray,
    Sijk1data::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, nHarm::Int64, γr::Float64, γθ::Float64, γϕ::Float64, nPoints::Int64, n_freqs::Int64, chisq::AbstractVector{Float64}, fit::String)
    γ = [γr, γθ, γϕ];
    if fit=="GSL"
        gsl_fit = true
        julia_fit = false
    elseif fit=="Julia"
        julia_fit = true
        gsl_fit = false
    else
        throw(ValueError("argument `fit` must be either `GSL` or `Julia`"))
    end

    # compute derivatives of coordinates wrt to lambda
    dx_dλ = [zeros(3) for i in eachindex(x)]
    d2x_dλ = [zeros(3) for i in eachindex(x)]

    # compute derivatives of coordinate time wrt lambda
    dt_dλ = zeros(length(x))
    d2t_dλ = zeros(length(x))


    # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
    dλ_dt = zeros(length(x))
    d2λ_dt = zeros(length(x))

    for i in eachindex(x)
        # compute derivatives of coordinates wrt to lambda
        dx_dλ[i] = [MinoDerivs1.dr_dλ(x[i], a, E, L, C) * sign(rdot[i]), MinoDerivs1.dθ_dλ(x[i], a, E, L, C) * sign(θdot[i]), MinoDerivs1.dϕ_dλ(x[i], a, E, L, C)]
        d2x_dλ[i] = [MinoDerivs2.d2r_dλ(x[i], dx_dλ[i], a, E, L, C), MinoDerivs2.d2θ_dλ(x[i], dx_dλ[i], a, E, L, C), MinoDerivs2.d2ϕ_dλ(x[i], dx_dλ[i], a, E, L, C)]

        # compute derivatives of coordinate time wrt lambda
        dt_dλ[i] = MinoDerivs1.dt_dλ(x[i], a, E, L, C);
        d2t_dλ[i] = MinoDerivs2.d2t_dλ(x[i], dx_dλ[i], a, E, L, C);

        # use chain rule to compute derivatives of lambda wrt coordinate time (this works because dt_dλ ≠ 0)
        dλ_dt[i] = MinoTimeDerivs.dλ_dt(dt_dλ[i])
        d2λ_dt[i] = MinoTimeDerivs.d2λ_dt(dt_dλ[i], d2t_dλ[i])
    end
 

    @inbounds Threads.@threads for multipole_moment in moments_wf
        fit_params = zeros(2 * n_freqs + 1);
        type = multipole_moment[1];

        if isequal(type, "MassOct")
            i1, i2, i3 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mijk2data[i1, i2, i3], nPoints, nHarm, chisq,  γ, fit_params)
                df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(λ,  Mijk2data[i1, i2, i3], nPoints, nHarm, γ, fit_params)
                df_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 1)
            end

            @views Mijk3[i1, i2, i3] = @. ParameterizedDerivs.df_dt(df_dλ, dλ_dt)
        elseif isequal(type, "MassHex")
            i1, i2, i3, i4 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Mijkl2data[i1, i2, i3, i4], nPoints, nHarm, chisq,  γ, fit_params)
                df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)
                d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(λ,  Mijkl2data[i1, i2, i3, i4], nPoints, nHarm, γ, fit_params)
                df_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 1)
                d2f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 2)
            end
            @views Mijkl4[i1, i2, i3, i4] = @. ParameterizedDerivs.d2f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt)

        elseif isequal(type, "CurrentQuad")
            i1, i2 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Sij1data[i1, i2], nPoints, nHarm, chisq,  γ, fit_params)
                df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(λ, Sij1data[i1, i2], nPoints, nHarm, γ, fit_params)
                df_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 1)
            end
            @views Sij2[i1, i2] = @. ParameterizedDerivs.df_dt(df_dλ, dλ_dt)
        elseif isequal(type, "CurrentOct")
            i1, i2, i3 = multipole_moment[2];
            if gsl_fit
                Ω_fit = FourierFitGSL.GSL_fit_master!(λ, Sijk1data[i1, i2, i3], nPoints, nHarm, chisq,  γ, fit_params)
                df_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 1)
                d2f_dλ = FourierFitGSL.curve_fit_functional_derivs(λ, Ω_fit, fit_params, n_freqs, nPoints, 2)
            elseif julia_fit
                Ω_fit = FourierFitJuliaBase.Fit_master!(λ,  Sijk1data[i1, i2, i3], nPoints, nHarm, γ, fit_params)
                df_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 1)
                d2f_dλ = FourierFitJuliaBase.curve_fit_functional_derivs(fit_params, Ω_fit, λ, 2)
            end
            @views Sijk3[i1, i2, i3] = @. ParameterizedDerivs.d2f_dt(df_dλ, dλ_dt, d2f_dλ, d2λ_dt)
        end
    end

    # symmetrize moments
    SymmetricTensors.SymmetrizeThreeIndexTensor!(Mijk3); SymmetricTensors.SymmetrizeFourIndexTensor!(Mijkl4);
    SymmetricTensors.SymmetrizeTwoIndexTensor!(Sij2); SymmetricTensors.SymmetrizeThreeIndexTensor!(Sijk3);
end


end