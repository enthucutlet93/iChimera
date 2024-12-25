#=

    In this module we project the metric perturbation from the kludge scheme in arXiv:1109.0572v2 into the TT gauge.

=#

module Waveform
using Combinatorics, LinearAlgebra

"""
# Common Arguments in this module
- `r::Float64`: observer distance.
- `Θ::Float64`: observer polar orientation.
- `ϕ::Float64`: observer azimuthal orientation.
- `Mij2::AbstractArray`: second derivative of the mass quadrupole (Eq. 48).
- `Mijk3::AbstractArray`: third derivative of the mass quadrupole (Eq. 48).
- `Mijkl4::AbstractArray`: fourth derivative of the mass quadrupole (Eq. 85).
- `Sij2::AbstractArray`: second derivative of the current quadrupole (Eq. 49).
- `Sijk3::AbstractArray`: third derivative of the current quadrupole (Eq. 86).
"""

const spatial_indices_3::Array = [[x, y, z] for x=1:3, y=1:3, z=1:3]
const εkl::Array{Vector} = [[levicivita(spatial_indices_3[k, l, i]) for i = 1:3] for k=1:3, l=1:3]

δ(i::Int, j::Int) = i == j ? 1.0 : 0.0
@inline outer(x::Vector{Float64}, y::Vector{Float64}) = [x[i] * y[j] for i in eachindex(x), j in eachindex(y)]

# returns plus and cross polarized waveforms in the wave frame (not detector frame) taking as argument the conventional sky location angles ThetaS, PhiS, ThetaS, ThetaK
function compute_wave_polarizations(nPoints::Int, r::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, Mij2::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray)
    hij = [zeros(nPoints) for i=1:3, j=1:3];
    hij_TT = [zeros(nPoints) for i=1:3, j=1:3];
    hplus = zeros(nPoints);
    hcross = zeros(nPoints);
    
    # R_ssb ≡ unit vector pointing from solar system barycenter (SSB) in the direction of the EMRI system in the SSB frame
    R_ssb = [sin(ThetaSource) * cos(PhiSource), sin(ThetaSource) * sin(PhiSource), cos(ThetaSource)]

    # S_src ≡ unit vector pointing from center of source- (EMRI-) frame coordinate system in the direction of the MBH's spin vector in the src frame
    S_src = [0., 0., 1.]

    # n_ssb ≡ unit vector pointing from LISA in the direction of the EMRI system in the SSB frame — this is an approximation (it should be time dependendent)
    n_ssb = R_ssb

    # orientation of source- (EMRI-) frame coordinate system in the SSB frame. The picture is to start with the src frame aligned with the SSB frame, and then to rotate the
    # basis vectors of the src frame so that the z-axis is aligned with the direction of the MBH's spin vector in the SSB frame. This is achieved by first carrying out a 
    # right-handed rotation of the src frame basis vectors about the y-axis by an angle ThetaKerr, and then a right-handed rotation about the z-axis by PhiKerr. The action
    # of the the two rotation matrices on each basis vector gives the components of the basis vectors of the src frame in the SSB frame.
    x_src = [cos(ThetaKerr) * cos(PhiKerr), cos(ThetaKerr) * sin(PhiKerr), -sin(ThetaKerr)]
    y_src = [-sin(PhiKerr), cos(PhiKerr), 0.]
    z_src = [sin(ThetaKerr) * cos(PhiKerr), sin(ThetaKerr) * sin(PhiKerr), cos(ThetaKerr)]

    # n_to_source_src ≡ unit vector pointing from center of source- (EMRI-) frame coordinate system in the direction of the observation point
    n_to_source_src = [dot(n_ssb, x_src), dot(n_ssb, y_src), dot(n_ssb, z_src)] 

    # define plus and cross polarization tensors (in wave frame)
    p = cross(n_to_source_src, S_src)

    # check of n_to_source_src and S_src are parallel
    norm(p) < 1e-10 ? p = [1., 0., 0.] : p /= norm(p)

    q = cross(p, n_to_source_src); q /= norm(q);

    Hplus = [p[i] * p[j] - q[i] * q[j] for i=1:3, j=1:3]
    Hcross = [p[i] * q[j] + q[i] * p[j] for i=1:3, j=1:3]
    
    # calculate metric perturbations in source frame
    n_to_obs_src = -n_to_source_src # unit vector pointing from center of source- (EMRI-) frame coordinate system in the direction of the observation point
    @inbounds Threads.@threads for t=1:nPoints
        @inbounds for i=1:3
            @inbounds for j=1:3
                hij[i, j][t] = 0.    # set all entries to zero

                hij[i, j][t] += 2.0 * Mij2[i, j][t] / r    # first term in Eq. 84 

                @inbounds for k=1:3
                    hij[i, j][t] += 2.0 * Mijk3[i, j, k][t] * n_to_obs_src[k] / (3.0r)    # second term in Eq. 84
    
                    @inbounds for l=1:3
                        hij[i, j][t] += 4.0 * (εkl[k, l][i] * Sij2[j, k][t] * n_to_obs_src[l] + εkl[k, l][j] * Sij2[i, k][t] * n_to_obs_src[l]) / (3.0r) + Mijkl4[i, j, k, l][t] * n_to_obs_src[k] * n_to_obs_src[l] / (6.0r)    # third and fourth terms in Eq. 84
        
                        @inbounds for m=1:3
                            hij[i, j][t] += (εkl[k, l][i] * Sijk3[j, k, m][t] * n_to_obs_src[l] * n_to_obs_src[m] + εkl[k, l][j] * Sijk3[i, k, m][t] * n_to_obs_src[l] * n_to_obs_src[m]) / (2.0r)
                        end
                    end
                end
            end
        end
    end

    # project into TT gauge and compute plus and cross polarizations
    P = [δ(i, j) - n_to_obs_src[i] * n_to_obs_src[j] for i=1:3, j=1:3];
    Πijmn = [P[i, m] * P[j, n] - 0.5 * P[i,j] * P[m,n] for i=1:3, j=1:3, m=1:3, n=1:3];

    @inbounds for i = 1:3, j = 1:3
        for m=1:3, n=1:3
            hij_TT[i, j] += Πijmn[i, j, m, n] * hij[m, n]
        end

        hplus[:] += 0.5 * Hplus[i, j] * hij_TT[i, j]
        hcross[:] += 0.5 * Hcross[i, j] * hij_TT[i, j]
    end

    return hplus, hcross
end

# returns plus and cross polarized waveforms in the wave frame (not detector frame) taking as argument the observation angles Θ, Φ
@views function compute_wave_polarizations(nPoints::Int, r::Float64, Θ::Float64, Φ::Float64, Mij2::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray)
    hij = [zeros(nPoints) for i=1:3, j=1:3];
    hij_TT = [zeros(nPoints) for i=1:3, j=1:3];
    hplus = zeros(nPoints);
    hcross = zeros(nPoints);
    
    # n ≡ unit vector pointing in direction of far away observer
    nx = sin(Θ) * cos(Φ)
    ny = sin(Θ) * sin(Φ)
    nz = cos(Θ)
    n_to_obs = [nx, ny, nz]
    n_to_source = -n_to_obs

    # S ≡ unit vector pointing from center of source- (EMRI-) frame coordinate system in the direction of the MBH's spin vector in the src frame
    S = [0., 0., 1.]

    # define plus and cross polarization tensors (in wave frame)
    p = cross(n_to_source, S) # first polarization vector — defined as the x-unit vector in the wave frame

    # check of n_to_source and S are parallel
    norm(p) < 1e-10 ? p = [1., 0., 0.] : p /= norm(p)

    q = cross(p, n_to_source); q /= norm(q); # second polarization vector — defined as the y-unit vector in the wave frame

    Hplus = [p[i] * p[j] - q[i] * q[j] for i=1:3, j=1:3] # plus polarization tensor
    Hcross = [p[i] * q[j] + q[i] * p[j] for i=1:3, j=1:3] # cross polarization tensor

    # calculate perturbations (Eq. 84)
    @inbounds Threads.@threads for t=1:nPoints
        for i=1:3
            @inbounds for j=1:3

                hij[i, j][t] = 0.    # set all entries to zero

                hij[i, j][t] += 2.0 * Mij2[i, j][t] / r    # first term in Eq. 84 

                @inbounds for k=1:3
                    hij[i, j][t] += 2.0 * Mijk3[i, j, k][t] * n_to_obs[k] / (3.0r)    # second term in Eq. 84
    
                    @inbounds for l=1:3
                        hij[i, j][t] += 4.0 * (εkl[k, l][i] * Sij2[j, k][t] * n_to_obs[l] + εkl[k, l][j] * Sij2[i, k][t] * n_to_obs[l]) / (3.0r) + Mijkl4[i, j, k, l][t] * n_to_obs[k] * n_to_obs[l] / (6.0r)    # third and fourth terms in Eq. 84
        
                        @inbounds for m=1:3
                            hij[i, j][t] += (εkl[k, l][i] * Sijk3[j, k, m][t] * n_to_obs[l] * n_to_obs[m] + εkl[k, l][j] * Sijk3[i, k, m][t] * n_to_obs[l] * n_to_obs[m]) / (2.0r)
                        end
                    end
                end
            end
        end
    end

    # project into TT gauge and compute plus and cross polarizations, e.g., see Eqs. 59, 237 in https://arxiv.org/pdf/gr-qc/0202016
    P = [δ(i, j) - n_to_obs[i] * n_to_obs[j] for i=1:3, j=1:3];
    Πijmn = [P[i, m] * P[j, n] - 0.5 * P[i,j] * P[m,n] for i=1:3, j=1:3, m=1:3, n=1:3];

    @inbounds for i = 1:3, j = 1:3
        for m=1:3, n=1:3
            hij_TT[i, j] += Πijmn[i, j, m, n] * hij[m, n]
        end

        hplus[:] += 0.5 * Hplus[i, j] * hij_TT[i, j]
        hcross[:] += 0.5 * Hcross[i, j] * hij_TT[i, j]
    end

    return hplus, hcross
end


# project h into TT gauge (Reference: https://arxiv.org/pdf/gr-qc/0607007)
hΘΘ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = (cos(Θ)^2) * (h[1, 1][t] * cos(Φ)^2 + h[1, 2][t] * sin(2Φ) + h[2, 2][t] * sin(Φ)^2) + h[3, 3][t] * sin(Θ)^2 - sin(2Θ) * (h[1, 3][t] * cos(Φ) + h[2, 3][t] * sin(Φ))   # Eq. 6.15
hΘΦ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = cos(Θ) * (((-0.5) * h[1, 1][t] * sin(2Φ)) + h[1, 2][t] * cos(2Φ) + 0.5 * h[2, 2][t] * sin(2Φ)) + sin(Θ) * (h[1, 3][t] * sin(Φ) - h[2, 3][t] * cos(Φ))   # Eq. 6.16
hΦΦ(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = h[1, 1][t] * sin(Φ)^2 - h[1, 2][t] * sin(2Φ) + h[2, 2][t] * cos(Φ)^2   # Eq. 6.17

# define h_{+} and h_{×} components of GW (https://arxiv.org/pdf/gr-qc/0607007)
hplus(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = (1/2) *  (hΘΘ(h, Θ, Φ, t) - hΦΦ(h, Θ, Φ, t))
hcross(h::AbstractArray, Θ::Float64, Φ::Float64, t::Int64) = hΘΦ(h, Θ, Φ, t)


function h_plus_cross(hij::AbstractArray, Θ::Float64, Φ::Float64)
    nPoints = length(hij[1, 1])
    hplus = zeros(nPoints)
    hcross = zeros(nPoints)
    @inbounds Threads.@threads for i in 1:nPoints
        hplus[i] = Waveform.hplus(hij, Θ, Φ, i)
        hcross[i] = Waveform.hcross(hij, Θ, Φ, i)
    end
    return hplus, hcross
end

# Eq. 8 in https://arxiv.org/pdf/2104.04582
function rotate_to_SSB_frame(h_plus::Vector{Float64}, h_cross::Vector{Float64}, ThetaSource::Float64, PhiSource::Float64, ThetaK::Float64, PhiK::Float64)
    tan_psi_denominator = sin(ThetaK) * sin(PhiSource - PhiK)
    
    if abs(tan_psi_denominator) < 1e-10
        psi = π / 2
    else
        tan_psi_numerator = cos(ThetaSource) * sin(ThetaK) * cos(PhiSource - PhiK) - sin(ThetaSource) * cos(ThetaK)
        psi = -atan2(tan_psi_numerator, tan_psi_denominator)
    end

    h_plus_SSB = h_plus * cos(2psi) - h_cross * sin(2psi)
    h_cross_SSB = h_plus * sin(2psi) + h_cross * cos(2psi)
    return h_plus_SSB, h_cross_SSB
end

end