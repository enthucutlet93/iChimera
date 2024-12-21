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

@inline outer(x::Vector{Float64}, y::Vector{Float64}) = [x[i] * y[j] for i in eachindex(x), j in eachindex(y)]

# returns pluss and cross polarized waveforms in the wave frame (not detector frame)
@views function compute_wave_polarizations!(hij::AbstractArray, nPoints::Int, r::Float64, ThetaSource::Float64, PhiSource::Float64, ThetaKerr::Float64, PhiKerr::Float64, Mij2::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray)
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

    # n_src ≡ unit vector pointing from center of source- (EMRI-) frame coordinate system in the direction of the observation point
    n_src = -[dot(n_ssb, x_src), dot(n_ssb, y_src), dot(n_ssb, z_src)] 

    # define plus and cross polarization tensors (in wave frame)
    p = cross(n_src, S_src)

    # check of n_src and S_ssb are parallel
    norm(p) < 1e-10 ? p = [1., 0., 0.] : p /= norm(p)

    q = cross(p, n_src); q /= norm(q);

    Hplus = [p[i] * p[j] - q[i] * q[j] for i=1:3, j=1:3]
    Hcross = [p[i] * q[j] + q[i] * p[j] for i=1:3, j=1:3]
    
    # calculate metric perturbations in source frame
    @inbounds Threads.@threads for t=1:nPoints
        @inbounds for i=1:3
            @inbounds for j=1:3
                hij[i, j][t] = 0.    # set all entries to zero

                hij[i, j][t] += 2.0 * Mij2[i, j][t] / r    # first term in Eq. 84 

                @inbounds for k=1:3
                    hij[i, j][t] += 2.0 * Mijk3[i, j, k][t] * n_src[k] / (3.0r)    # second term in Eq. 84
    
                    @inbounds for l=1:3
                        hij[i, j][t] += 4.0 * (εkl[k, l][i] * Sij2[j, k][t] * n_src[l] + εkl[k, l][j] * Sij2[i, k][t] * n_src[l]) / (3.0r) + Mijkl4[i, j, k, l][t] * n_src[k] * n_src[l] / (6.0r)    # third and fourth terms in Eq. 84
        
                        @inbounds for m=1:3
                            hij[i, j][t] += (εkl[k, l][i] * Sijk3[j, k, m][t] * n_src[l] * n_src[m] + εkl[k, l][j] * Sijk3[i, k, m][t] * n_src[l] * n_src[m]) / (2.0r)
                        end
                    end
                end
            end
        end
    end

    # compute plus and cross polarizations
    hplus = zeros(nPoints)
    hcross = zeros(nPoints)

    @inbounds for i = 1:3, j = 1:3
        hplus[:] += 0.5 * Hplus[i, j] * hij[i, j]
        hcross[:] += 0.5 * Hcross[i, j] * hij[i, j]
    end

    return hplus, hcross
end

end