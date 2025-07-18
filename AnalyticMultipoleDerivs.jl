#=

  In this module we write a functions to analytically compute the derivatives of the various multipole moments. (This file is depracated. We do not use the functions in this file.)

=#

module AnalyticMultipoleDerivs
using ..AnalyticCoordinateDerivs
using ..HarmonicCoordDerivs

@inline function ε(i::Int, j::Int, k::Int)::Int
    if (i == 1 && j == 2 && k == 3) || 
       (i == 2 && j == 3 && k == 1) || 
       (i == 3 && j == 1 && k == 2)
        return 1
    elseif (i == 3 && j == 2 && k == 1) || 
           (i == 2 && j == 1 && k == 3) || 
           (i == 1 && j == 3 && k == 2)
        return -1
    else
        return 0
    end
end

δ(x::Int, y::Int)::Int = x == y ? 1 : 0

# define mass-ratio parameter
η(q::Float64) = q/((1+q)^2)   # q = mass ratio

Mass_quad_prefactor(q::Float64) = η(q) * (1.0 + q)
Mass_oct_prefactor(q::Float64) = -η(q) * (1.0 - q)
Mass_hex_prefactor(q::Float64) = η(q) * (1.0 + q)
Current_quad_prefactor(q::Float64) = -η(q) * (1.0 - q)
Current_oct_prefactor(q::Float64) = η(q) * (1.0 + q)


### MASS MultipoleMoments ###

Mij2(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = 2*dx[i]*dx[j] - (2*δ(i,j)*(dx[1]^2 + dx[2]^2 + dx[3]^2 + x[1]*d2x[1] + x[2]*d2x[2] + x[3]*d2x[3]))/3. +
x[j]*d2x[i] + x[i]*d2x[j]

Mij3(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = 3*dx[j]*d2x[i] + 3*dx[i]*d2x[j] - (2*δ(i,j)*(3*dx[1]*d2x[1] + 3*dx[2]*d2x[2] + 3*dx[3]*d2x[3] + x[1]*d3x[1] + x[2]*d3x[2] + x[3]*d3x[3]))/3. + x[j]*d3x[i] + x[i]*d3x[j]

Mij4(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = 6*d2x[i]*d2x[j] + 4*dx[j]*d3x[i] + 4*dx[i]*d3x[j] - (2*δ(i,j)*(3*d2x[1]^2 + 3*d2x[2]^2 + 3*d2x[3]^2 + 4*dx[1]*d3x[1] + 4*dx[2]*d3x[2] + 4*dx[3]*d3x[3] + x[1]*d4x[1] + x[2]*d4x[2] + x[3]*d4x[3]))/3. + x[j]*d4x[i] + x[i]*d4x[j]

Mij5(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = 10*d2x[j]*d3x[i] + 10*d2x[i]*d3x[j] +
5*dx[j]*d4x[i] + 5*dx[i]*d4x[j] - (2*δ(i,j)*(10*d2x[1]*d3x[1] + 10*d2x[2]*d3x[2] + 10*d2x[3]*d3x[3] + 5*dx[1]*d4x[1] + 5*dx[2]*d4x[2] + 5*dx[3]*d4x[3] + x[1]*d5x[1] + x[2]*d5x[2] + x[3]*d5x[3]))/3. + x[j]*d5x[i] +
x[i]*d5x[j]

Mij6(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = 20*d3x[i]*d3x[j] +
15*d2x[j]*d4x[i] + 15*d2x[i]*d4x[j] + 6*dx[j]*d5x[i] + 6*dx[i]*d5x[j] - (2*δ(i,j)*(10*d3x[1]^2 + 10*d3x[2]^2 + 10*d3x[3]^2 + 15*d2x[1]*d4x[1] + 15*d2x[2]*d4x[2] + 15*d2x[3]*d4x[3] + 6*dx[1]*d5x[1] + 6*dx[2]*d5x[2] +
6*dx[3]*d5x[3] + x[1]*d6x[1] + x[2]*d6x[2] + x[3]*d6x[3]))/3. + x[j]*d6x[i] + x[i]*d6x[j]

Mij7(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = 35*d3x[j]*d4x[i] +
35*d3x[i]*d4x[j] + 21*d2x[j]*d5x[i] + 21*d2x[i]*d5x[j] + 7*dx[j]*d6x[i] + 7*dx[i]*d6x[j] - (2*δ(i,j)*(35*d3x[1]*d4x[1] + 35*d3x[2]*d4x[2] + 35*d3x[3]*d4x[3] + 21*d2x[1]*d5x[1] + 21*d2x[2]*d5x[2] + 21*d2x[3]*d5x[3] +
7*dx[1]*d6x[1] + 7*dx[2]*d6x[2] + 7*dx[3]*d6x[3] + x[1]*d7x[1] + x[2]*d7x[2] + x[3]*d7x[3]))/3. + x[j]*d7x[i] + x[i]*d7x[j]

Mij8(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, d8x::AbstractVector{Float64}, i::Int64,
j::Int64)::Float64 = 70*d4x[i]*d4x[j] + 56*d3x[j]*d5x[i] + 56*d3x[i]*d5x[j] + 28*d2x[j]*d6x[i] + 28*d2x[i]*d6x[j] + 8*dx[j]*d7x[i] + 8*dx[i]*d7x[j] - (2*δ(i,j)*(35*d4x[1]^2 + 35*d4x[2]^2 + 35*d4x[3]^2 +
56*d3x[1]*d5x[1] + 56*d3x[2]*d5x[2] + 56*d3x[3]*d5x[3] + 28*d2x[1]*d6x[1] + 28*d2x[2]*d6x[2] + 28*d2x[3]*d6x[3] + 8*dx[1]*d7x[1] + 8*dx[2]*d7x[2] + 8*dx[3]*d7x[3] + x[1]*d8x[1] + x[2]*d8x[2] + x[3]*d8x[3]))/3. +
x[j]*d8x[i] + x[i]*d8x[j]

Mijk3(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, i::Int64, j::Int64, k::Int64)::Float64 = (6*(δ(j,k)*dx[i] + δ(i,k)*dx[j] + δ(i,j)*dx[k])*(dx[1]^2 + dx[2]^2 + dx[3]^2 +
x[1]*d2x[1] + x[2]*d2x[2] + x[3]*d2x[3]) - 15*(x[k]*dx[j] + x[j]*dx[k])*d2x[i] + 6*(x[1]*dx[1] + x[2]*dx[2] + x[3]*dx[3])*(δ(j,k)*d2x[i] + δ(i,k)*d2x[j] + δ(i,j)*d2x[k]) - 15*dx[i]*(2*dx[j]*dx[k] + x[k]*d2x[j] +
x[j]*d2x[k]) + 2*(δ(j,k)*x[i] + δ(i,k)*x[j] + δ(i,j)*x[k])*(3*dx[1]*d2x[1] + 3*dx[2]*d2x[2] + 3*dx[3]*d2x[3] + x[1]*d3x[1] + x[2]*d3x[2] + x[3]*d3x[3]) - 5*x[j]*x[k]*d3x[i] + (x[1]^2 + x[2]^2 + x[3]^2)*(δ(j,k)*d3x[i] +
δ(i,k)*d3x[j] + δ(i,j)*d3x[k]) - 5*x[i]*(3*dx[k]*d2x[j] + 3*dx[j]*d2x[k] + x[k]*d3x[j] + x[j]*d3x[k]))/5.

Mijk7(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, i::Int64, j::Int64,
k::Int64)::Float64 = (70*(δ(j,k)*d3x[i] + δ(i,k)*d3x[j] + δ(i,j)*d3x[k])*(3*d2x[1]^2 + 3*d2x[2]^2 + 3*d2x[3]^2 + 4*dx[1]*d3x[1] + 4*dx[2]*d3x[2] + 4*dx[3]*d3x[3] + x[1]*d4x[1] + x[2]*d4x[2] + x[3]*d4x[3]) -
175*(3*dx[k]*d2x[j] + 3*dx[j]*d2x[k] + x[k]*d3x[j] + x[j]*d3x[k])*d4x[i] + 70*(3*dx[1]*d2x[1] + 3*dx[2]*d2x[2] + 3*dx[3]*d2x[3] + x[1]*d3x[1] + x[2]*d3x[2] + x[3]*d3x[3])*(δ(j,k)*d4x[i] + δ(i,k)*d4x[j] +
δ(i,j)*d4x[k]) - 175*d3x[i]*(6*d2x[j]*d2x[k] + 4*dx[k]*d3x[j] + 4*dx[j]*d3x[k] + x[k]*d4x[j] + x[j]*d4x[k]) + 42*(δ(j,k)*d2x[i] + δ(i,k)*d2x[j] + δ(i,j)*d2x[k])*(10*d2x[1]*d3x[1] + 10*d2x[2]*d3x[2] +
10*d2x[3]*d3x[3] + 5*dx[1]*d4x[1] + 5*dx[2]*d4x[2] + 5*dx[3]*d4x[3] + x[1]*d5x[1] + x[2]*d5x[2] + x[3]*d5x[3]) - 105*(2*dx[j]*dx[k] + x[k]*d2x[j] + x[j]*d2x[k])*d5x[i] + 42*(dx[1]^2 + dx[2]^2 + dx[3]^2 +
x[1]*d2x[1] + x[2]*d2x[2] + x[3]*d2x[3])*(δ(j,k)*d5x[i] + δ(i,k)*d5x[j] + δ(i,j)*d5x[k]) - 105*d2x[i]*(10*d2x[k]*d3x[j] + 10*d2x[j]*d3x[k] + 5*dx[k]*d4x[j] + 5*dx[j]*d4x[k] + x[k]*d5x[j] + x[j]*d5x[k]) +
14*(δ(j,k)*dx[i] + δ(i,k)*dx[j] + δ(i,j)*dx[k])*(10*d3x[1]^2 + 10*d3x[2]^2 + 10*d3x[3]^2 + 15*d2x[1]*d4x[1] + 15*d2x[2]*d4x[2] + 15*d2x[3]*d4x[3] + 6*dx[1]*d5x[1] + 6*dx[2]*d5x[2] + 6*dx[3]*d5x[3] + x[1]*d6x[1] +
x[2]*d6x[2] + x[3]*d6x[3]) - 35*(x[k]*dx[j] + x[j]*dx[k])*d6x[i] + 14*(x[1]*dx[1] + x[2]*dx[2] + x[3]*dx[3])*(δ(j,k)*d6x[i] + δ(i,k)*d6x[j] + δ(i,j)*d6x[k]) - 35*dx[i]*(20*d3x[j]*d3x[k] + 15*d2x[k]*d4x[j] +
15*d2x[j]*d4x[k] + 6*dx[k]*d5x[j] + 6*dx[j]*d5x[k] + x[k]*d6x[j] + x[j]*d6x[k]) + 2*(δ(j,k)*x[i] + δ(i,k)*x[j] + δ(i,j)*x[k])*(35*d3x[1]*d4x[1] + 35*d3x[2]*d4x[2] + 35*d3x[3]*d4x[3] + 21*d2x[1]*d5x[1] +
21*d2x[2]*d5x[2] + 21*d2x[3]*d5x[3] + 7*dx[1]*d6x[1] + 7*dx[2]*d6x[2] + 7*dx[3]*d6x[3] + x[1]*d7x[1] + x[2]*d7x[2] + x[3]*d7x[3]) - 5*x[j]*x[k]*d7x[i] + (x[1]^2 + x[2]^2 + x[3]^2)*(δ(j,k)*d7x[i] + δ(i,k)*d7x[j] +
δ(i,j)*d7x[k]) - 5*x[i]*(35*d3x[k]*d4x[j] + 35*d3x[j]*d4x[k] + 21*d2x[k]*d5x[j] + 21*d2x[j]*d5x[k] + 7*dx[k]*d6x[j] + 7*dx[j]*d6x[k] + x[k]*d7x[j] + x[j]*d7x[k]))/5.

Mijk8(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, d8x::AbstractVector{Float64}, i::Int64, j::Int64,
k::Int64)::Float64 = (140*(3*d2x[1]^2 + 3*d2x[2]^2 + 3*d2x[3]^2 + 4*dx[1]*d3x[1] + 4*dx[2]*d3x[2] + 4*dx[3]*d3x[3] + x[1]*d4x[1] + x[2]*d4x[2] + x[3]*d4x[3])*(δ(j,k)*d4x[i] + δ(i,k)*d4x[j] + δ(i,j)*d4x[k]) -
350*d4x[i]*(6*d2x[j]*d2x[k] + 4*dx[k]*d3x[j] + 4*dx[j]*d3x[k] + x[k]*d4x[j] + x[j]*d4x[k]) + 112*(δ(j,k)*d3x[i] + δ(i,k)*d3x[j] + δ(i,j)*d3x[k])*(10*d2x[1]*d3x[1] + 10*d2x[2]*d3x[2] + 10*d2x[3]*d3x[3] +
5*dx[1]*d4x[1] + 5*dx[2]*d4x[2] + 5*dx[3]*d4x[3] + x[1]*d5x[1] + x[2]*d5x[2] + x[3]*d5x[3]) - 280*(3*dx[k]*d2x[j] + 3*dx[j]*d2x[k] + x[k]*d3x[j] + x[j]*d3x[k])*d5x[i] + 112*(3*dx[1]*d2x[1] + 3*dx[2]*d2x[2] +
3*dx[3]*d2x[3] + x[1]*d3x[1] + x[2]*d3x[2] + x[3]*d3x[3])*(δ(j,k)*d5x[i] + δ(i,k)*d5x[j] + δ(i,j)*d5x[k]) - 280*d3x[i]*(10*d2x[k]*d3x[j] + 10*d2x[j]*d3x[k] + 5*dx[k]*d4x[j] + 5*dx[j]*d4x[k] + x[k]*d5x[j] +
x[j]*d5x[k]) + 56*(δ(j,k)*d2x[i] + δ(i,k)*d2x[j] + δ(i,j)*d2x[k])*(10*d3x[1]^2 + 10*d3x[2]^2 + 10*d3x[3]^2 + 15*d2x[1]*d4x[1] + 15*d2x[2]*d4x[2] + 15*d2x[3]*d4x[3] + 6*dx[1]*d5x[1] + 6*dx[2]*d5x[2] + 6*dx[3]*d5x[3] +
x[1]*d6x[1] + x[2]*d6x[2] + x[3]*d6x[3]) - 140*(2*dx[j]*dx[k] + x[k]*d2x[j] + x[j]*d2x[k])*d6x[i] + 56*(dx[1]^2 + dx[2]^2 + dx[3]^2 + x[1]*d2x[1] + x[2]*d2x[2] + x[3]*d2x[3])*(δ(j,k)*d6x[i] + δ(i,k)*d6x[j] +
δ(i,j)*d6x[k]) - 140*d2x[i]*(20*d3x[j]*d3x[k] + 15*d2x[k]*d4x[j] + 15*d2x[j]*d4x[k] + 6*dx[k]*d5x[j] + 6*dx[j]*d5x[k] + x[k]*d6x[j] + x[j]*d6x[k]) + 16*(δ(j,k)*dx[i] + δ(i,k)*dx[j] +
δ(i,j)*dx[k])*(35*d3x[1]*d4x[1] + 35*d3x[2]*d4x[2] + 35*d3x[3]*d4x[3] + 21*d2x[1]*d5x[1] + 21*d2x[2]*d5x[2] + 21*d2x[3]*d5x[3] + 7*dx[1]*d6x[1] + 7*dx[2]*d6x[2] + 7*dx[3]*d6x[3] + x[1]*d7x[1] + x[2]*d7x[2] +
x[3]*d7x[3]) - 40*(x[k]*dx[j] + x[j]*dx[k])*d7x[i] + 16*(x[1]*dx[1] + x[2]*dx[2] + x[3]*dx[3])*(δ(j,k)*d7x[i] + δ(i,k)*d7x[j] + δ(i,j)*d7x[k]) - 40*dx[i]*(35*d3x[k]*d4x[j] + 35*d3x[j]*d4x[k] + 21*d2x[k]*d5x[j] +
21*d2x[j]*d5x[k] + 7*dx[k]*d6x[j] + 7*dx[j]*d6x[k] + x[k]*d7x[j] + x[j]*d7x[k]) + 2*(δ(j,k)*x[i] + δ(i,k)*x[j] + δ(i,j)*x[k])*(35*d4x[1]^2 + 35*d4x[2]^2 + 35*d4x[3]^2 + 56*d3x[1]*d5x[1] + 56*d3x[2]*d5x[2] +
56*d3x[3]*d5x[3] + 28*d2x[1]*d6x[1] + 28*d2x[2]*d6x[2] + 28*d2x[3]*d6x[3] + 8*dx[1]*d7x[1] + 8*dx[2]*d7x[2] + 8*dx[3]*d7x[3] + x[1]*d8x[1] + x[2]*d8x[2] + x[3]*d8x[3]) - 5*x[j]*x[k]*d8x[i] + (x[1]^2 + x[2]^2 +
x[3]^2)*(δ(j,k)*d8x[i] + δ(i,k)*d8x[j] + δ(i,j)*d8x[k]) - 5*x[i]*(70*d4x[j]*d4x[k] + 56*d3x[k]*d5x[j] + 56*d3x[j]*d5x[k] + 28*d2x[k]*d6x[j] + 28*d2x[j]*d6x[k] + 8*dx[k]*d7x[j] + 8*dx[j]*d7x[k] + x[k]*d8x[j] +
x[j]*d8x[k]))/5.

Mijkl(x::AbstractVector{Float64}, i::Int64, j::Int64, k::Int64, l::Int64)::Float64 = ((δ(i,l)*δ(j,k) + δ(i,k)*δ(j,l) + δ(i,j)*δ(k,l))*(x[1]^2 + x[2]^2 + x[3]^2)^2)/35. + x[i]*x[j]*x[k]*x[l] - ((x[1]^2 + x[2]^2 +
x[3]^2)*(δ(k,l)*x[i]*x[j] + δ(j,l)*x[i]*x[k] + δ(i,l)*x[j]*x[k] + (δ(j,k)*x[i] + δ(i,k)*x[j] + δ(i,j)*x[k])*x[l]))/7.

Mijkl4(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, i::Int64, j::Int64, k::Int64, l::Int64)::Float64 = 6*(2*dx[i]*dx[j] + x[j]*d2x[i] +
x[i]*d2x[j])*(2*dx[k]*dx[l] + x[l]*d2x[k] + x[k]*d2x[l]) - (12*(dx[1]^2 + dx[2]^2 + dx[3]^2 + x[1]*d2x[1] + x[2]*d2x[2] + x[3]*d2x[3])*(2*δ(i,l)*dx[j]*dx[k] + 2*δ(j,k)*dx[i]*dx[l] + 2*δ(i,k)*dx[j]*dx[l] +
2*δ(i,j)*dx[k]*dx[l] + δ(j,k)*x[l]*d2x[i] + δ(i,l)*x[k]*d2x[j] + δ(i,k)*x[l]*d2x[j] + δ(k,l)*(2*dx[i]*dx[j] + x[j]*d2x[i] + x[i]*d2x[j]) + δ(i,l)*x[j]*d2x[k] + δ(i,j)*x[l]*d2x[k] + δ(j,l)*(2*dx[i]*dx[k] +
x[k]*d2x[i] + x[i]*d2x[k]) + δ(j,k)*x[i]*d2x[l] + δ(i,k)*x[j]*d2x[l] + δ(i,j)*x[k]*d2x[l]))/7. - (8*(δ(j,k)*x[l]*dx[i] + δ(i,l)*x[k]*dx[j] + δ(i,k)*x[l]*dx[j] + δ(k,l)*(x[j]*dx[i] + x[i]*dx[j]) + δ(i,l)*x[j]*dx[k] +
δ(i,j)*x[l]*dx[k] + δ(j,l)*(x[k]*dx[i] + x[i]*dx[k]) + δ(j,k)*x[i]*dx[l] + δ(i,k)*x[j]*dx[l] + δ(i,j)*x[k]*dx[l])*(3*dx[1]*d2x[1] + 3*dx[2]*d2x[2] + 3*dx[3]*d2x[3] + x[1]*d3x[1] + x[2]*d3x[2] + x[3]*d3x[3]))/7. +
4*(x[l]*dx[k] + x[k]*dx[l])*(3*dx[j]*d2x[i] + 3*dx[i]*d2x[j] + x[j]*d3x[i] + x[i]*d3x[j]) + 4*(x[j]*dx[i] + x[i]*dx[j])*(3*dx[l]*d2x[k] + 3*dx[k]*d2x[l] + x[l]*d3x[k] + x[k]*d3x[l]) - (8*(x[1]*dx[1] + x[2]*dx[2] +
x[3]*dx[3])*(3*δ(k,l)*dx[j]*d2x[i] + 3*δ(j,l)*dx[k]*d2x[i] + 3*δ(k,l)*dx[i]*d2x[j] + 3*δ(i,l)*dx[k]*d2x[j] + 3*δ(j,l)*dx[i]*d2x[k] + 3*δ(i,l)*dx[j]*d2x[k] + 3*dx[l]*(δ(j,k)*d2x[i] + δ(i,k)*d2x[j] + δ(i,j)*d2x[k]) +
3*(δ(j,k)*dx[i] + δ(i,k)*dx[j] + δ(i,j)*dx[k])*d2x[l] + δ(k,l)*x[j]*d3x[i] + δ(j,l)*x[k]*d3x[i] + δ(k,l)*x[i]*d3x[j] + δ(i,l)*x[k]*d3x[j] + δ(j,l)*x[i]*d3x[k] + δ(i,l)*x[j]*d3x[k] + x[l]*(δ(j,k)*d3x[i] +
δ(i,k)*d3x[j] + δ(i,j)*d3x[k]) + (δ(j,k)*x[i] + δ(i,k)*x[j] + δ(i,j)*x[k])*d3x[l]))/7. - (2*(δ(k,l)*x[i]*x[j] + δ(j,l)*x[i]*x[k] + δ(i,l)*x[j]*x[k] + δ(j,k)*x[i]*x[l] + δ(i,k)*x[j]*x[l] +
δ(i,j)*x[k]*x[l])*(3*d2x[1]^2 + 3*d2x[2]^2 + 3*d2x[3]^2 + 4*dx[1]*d3x[1] + 4*dx[2]*d3x[2] + 4*dx[3]*d3x[3] + x[1]*d4x[1] + x[2]*d4x[2] + x[3]*d4x[3]))/7. + (4*(δ(i,l)*δ(j,k) + δ(i,k)*δ(j,l) +
δ(i,j)*δ(k,l))*(6*(dx[1]^2 + dx[2]^2 + dx[3]^2 + x[1]*d2x[1] + x[2]*d2x[2] + x[3]*d2x[3])^2 + 8*(x[1]*dx[1] + x[2]*dx[2] + x[3]*dx[3])*(3*dx[1]*d2x[1] + 3*dx[2]*d2x[2] + 3*dx[3]*d2x[3] + x[1]*d3x[1] +
x[2]*d3x[2] + x[3]*d3x[3]) + (x[1]^2 + x[2]^2 + x[3]^2)*(3*d2x[1]^2 + 3*d2x[2]^2 + 3*d2x[3]^2 + 4*dx[1]*d3x[1] + 4*dx[2]*d3x[2] + 4*dx[3]*d3x[3] + x[1]*d4x[1] + x[2]*d4x[2] + x[3]*d4x[3])))/35. +
x[k]*x[l]*(6*d2x[i]*d2x[j] + 4*dx[j]*d3x[i] + 4*dx[i]*d3x[j] + x[j]*d4x[i] + x[i]*d4x[j]) + x[i]*x[j]*(6*d2x[k]*d2x[l] + 4*dx[l]*d3x[k] + 4*dx[k]*d3x[l] + x[l]*d4x[k] + x[k]*d4x[l]) - ((x[1]^2 + x[2]^2 +
x[3]^2)*(6*δ(k,l)*d2x[i]*d2x[j] + 6*δ(j,l)*d2x[i]*d2x[k] + 6*δ(i,l)*d2x[j]*d2x[k] + 6*(δ(j,k)*d2x[i] + δ(i,k)*d2x[j] + δ(i,j)*d2x[k])*d2x[l] + 4*δ(k,l)*dx[j]*d3x[i] + 4*δ(j,l)*dx[k]*d3x[i] + 4*δ(k,l)*dx[i]*d3x[j] +
4*δ(i,l)*dx[k]*d3x[j] + 4*δ(j,l)*dx[i]*d3x[k] + 4*δ(i,l)*dx[j]*d3x[k] + 4*dx[l]*(δ(j,k)*d3x[i] + δ(i,k)*d3x[j] + δ(i,j)*d3x[k]) + 4*(δ(j,k)*dx[i] + δ(i,k)*dx[j] + δ(i,j)*dx[k])*d3x[l] + δ(k,l)*x[j]*d4x[i] +
δ(j,l)*x[k]*d4x[i] + δ(k,l)*x[i]*d4x[j] + δ(i,l)*x[k]*d4x[j] + δ(j,l)*x[i]*d4x[k] + δ(i,l)*x[j]*d4x[k] + x[l]*(δ(j,k)*d4x[i] + δ(i,k)*d4x[j] + δ(i,j)*d4x[k]) + (δ(j,k)*x[i] + δ(i,k)*x[j] + δ(i,j)*x[k])*d4x[l]))/7.

### CURRENT MultipoleMoments ###
Sijkl2(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, i::Int64, j::Int64, k::Int64, l::Int64)::Float64 = (dx[l]*(2*(2*δ(i,j)*(ε(k,l,1)*dx[1] + ε(k,l,2)*dx[2] + ε(k,l,3)*dx[3]) -
3*(ε(k,l,j)*dx[i] + ε(k,l,i)*dx[j]))*dx[k] + x[k]*(2*δ(i,j)*(ε(k,l,1)*d2x[1] + ε(k,l,2)*d2x[2] + ε(k,l,3)*d2x[3]) - 3*(ε(k,l,j)*d2x[i] + ε(k,l,i)*d2x[j])) + (2*δ(i,j)*(ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3]) -
3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j]))*d2x[k]) + 2*(-3*x[k]*(ε(k,l,j)*dx[i] + ε(k,l,i)*dx[j]) - 3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j])*dx[k] + 2*δ(i,j)*(x[k]*(ε(k,l,1)*dx[1] + ε(k,l,2)*dx[2] + ε(k,l,3)*dx[3]) + (ε(k,l,1)*x[1] +
ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*dx[k]))*d2x[l] + (2*δ(i,j)*(ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3]) - 3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j]))*x[k]*d3x[l])/6.

Sijkl5(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, i::Int64, j::Int64, k::Int64, l::Int64)::Float64 = (10*(2*δ(i,j)*(3*dx[k]*(ε(k,l,1)*d2x[1] +
ε(k,l,2)*d2x[2] + ε(k,l,3)*d2x[3]) + 3*ε(k,l,1)*dx[1]*d2x[k] + 3*ε(k,l,2)*dx[2]*d2x[k] + 3*ε(k,l,3)*dx[3]*d2x[k] + ε(k,l,1)*x[k]*d3x[1] + ε(k,l,2)*x[k]*d3x[2] + ε(k,l,3)*x[k]*d3x[3] + (ε(k,l,1)*x[1] + ε(k,l,2)*x[2] +
ε(k,l,3)*x[3])*d3x[k]) - 3*(3*dx[k]*(ε(k,l,j)*d2x[i] + ε(k,l,i)*d2x[j]) + 3*ε(k,l,j)*dx[i]*d2x[k] + 3*ε(k,l,i)*dx[j]*d2x[k] + ε(k,l,j)*x[k]*d3x[i] + ε(k,l,i)*x[k]*d3x[j] + (ε(k,l,j)*x[i] +
ε(k,l,i)*x[j])*d3x[k]))*d3x[l] + 5*d2x[l]*(12*δ(i,j)*(ε(k,l,1)*d2x[1] + ε(k,l,2)*d2x[2] + ε(k,l,3)*d2x[3])*d2x[k] - 18*(ε(k,l,j)*d2x[i] + ε(k,l,i)*d2x[j])*d2x[k] + 8*δ(i,j)*dx[k]*(ε(k,l,1)*d3x[1] +
ε(k,l,2)*d3x[2] + ε(k,l,3)*d3x[3]) - 12*dx[k]*(ε(k,l,j)*d3x[i] + ε(k,l,i)*d3x[j]) + 8*δ(i,j)*(ε(k,l,1)*dx[1] + ε(k,l,2)*dx[2] + ε(k,l,3)*dx[3])*d3x[k] - 12*(ε(k,l,j)*dx[i] + ε(k,l,i)*dx[j])*d3x[k] +
2*δ(i,j)*x[k]*(ε(k,l,1)*d4x[1] + ε(k,l,2)*d4x[2] + ε(k,l,3)*d4x[3]) - 3*x[k]*(ε(k,l,j)*d4x[i] + ε(k,l,i)*d4x[j]) + 2*δ(i,j)*(ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*d4x[k] - 3*(ε(k,l,j)*x[i] +
ε(k,l,i)*x[j])*d4x[k]) + 10*(2*δ(i,j)*(2*ε(k,l,1)*dx[1]*dx[k] + 2*ε(k,l,2)*dx[2]*dx[k] + 2*ε(k,l,3)*dx[3]*dx[k] + ε(k,l,1)*x[k]*d2x[1] + ε(k,l,2)*x[k]*d2x[2] + ε(k,l,3)*x[k]*d2x[3] + (ε(k,l,1)*x[1] +
ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*d2x[k]) - 3*(2*ε(k,l,j)*dx[i]*dx[k] + 2*ε(k,l,i)*dx[j]*dx[k] + ε(k,l,j)*x[k]*d2x[i] + ε(k,l,i)*x[k]*d2x[j] + (ε(k,l,j)*x[i] + ε(k,l,i)*x[j])*d2x[k]))*d4x[l] +
dx[l]*(2*δ(i,j)*(10*d2x[k]*(ε(k,l,1)*d3x[1] + ε(k,l,2)*d3x[2] + ε(k,l,3)*d3x[3]) + 10*ε(k,l,1)*d2x[1]*d3x[k] + 10*ε(k,l,2)*d2x[2]*d3x[k] + 10*ε(k,l,3)*d2x[3]*d3x[k] + 5*ε(k,l,1)*dx[k]*d4x[1] +
5*ε(k,l,2)*dx[k]*d4x[2] + 5*ε(k,l,3)*dx[k]*d4x[3] + 5*ε(k,l,1)*dx[1]*d4x[k] + 5*ε(k,l,2)*dx[2]*d4x[k] + 5*ε(k,l,3)*dx[3]*d4x[k] + ε(k,l,1)*x[k]*d5x[1] + ε(k,l,2)*x[k]*d5x[2] + ε(k,l,3)*x[k]*d5x[3] +
(ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*d5x[k]) - 3*(10*d2x[k]*(ε(k,l,j)*d3x[i] + ε(k,l,i)*d3x[j]) + 10*ε(k,l,j)*d2x[i]*d3x[k] + ε(k,l,j)*(5*dx[k]*d4x[i] + 5*dx[i]*d4x[k] + x[k]*d5x[i] + x[i]*d5x[k]) +
ε(k,l,i)*(10*d2x[j]*d3x[k] + 5*dx[k]*d4x[j] + 5*dx[j]*d4x[k] + x[k]*d5x[j] + x[j]*d5x[k]))) + 5*(-3*x[k]*(ε(k,l,j)*dx[i] + ε(k,l,i)*dx[j]) - 3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j])*dx[k] + 2*δ(i,j)*(x[k]*(ε(k,l,1)*dx[1] +
ε(k,l,2)*dx[2] + ε(k,l,3)*dx[3]) + (ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*dx[k]))*d5x[l] + (2*δ(i,j)*(ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3]) - 3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j]))*x[k]*d6x[l])/6.

Sijkl6(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, i::Int64, j::Int64,
k::Int64, l::Int64)::Float64 = (15*d3x[l]*(12*δ(i,j)*(ε(k,l,1)*d2x[1] + ε(k,l,2)*d2x[2] + ε(k,l,3)*d2x[3])*d2x[k] - 18*(ε(k,l,j)*d2x[i] + ε(k,l,i)*d2x[j])*d2x[k] + 8*δ(i,j)*dx[k]*(ε(k,l,1)*d3x[1] + ε(k,l,2)*d3x[2] +
ε(k,l,3)*d3x[3]) - 12*dx[k]*(ε(k,l,j)*d3x[i] + ε(k,l,i)*d3x[j]) + 8*δ(i,j)*(ε(k,l,1)*dx[1] + ε(k,l,2)*dx[2] + ε(k,l,3)*dx[3])*d3x[k] - 12*(ε(k,l,j)*dx[i] + ε(k,l,i)*dx[j])*d3x[k] + 2*δ(i,j)*x[k]*(ε(k,l,1)*d4x[1] +
ε(k,l,2)*d4x[2] + ε(k,l,3)*d4x[3]) - 3*x[k]*(ε(k,l,j)*d4x[i] + ε(k,l,i)*d4x[j]) + 2*δ(i,j)*(ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*d4x[k] - 3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j])*d4x[k]) +
20*(2*δ(i,j)*(3*dx[k]*(ε(k,l,1)*d2x[1] + ε(k,l,2)*d2x[2] + ε(k,l,3)*d2x[3]) + 3*ε(k,l,1)*dx[1]*d2x[k] + 3*ε(k,l,2)*dx[2]*d2x[k] + 3*ε(k,l,3)*dx[3]*d2x[k] + ε(k,l,1)*x[k]*d3x[1] + ε(k,l,2)*x[k]*d3x[2] +
ε(k,l,3)*x[k]*d3x[3] + (ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*d3x[k]) - 3*(3*dx[k]*(ε(k,l,j)*d2x[i] + ε(k,l,i)*d2x[j]) + 3*ε(k,l,j)*dx[i]*d2x[k] + 3*ε(k,l,i)*dx[j]*d2x[k] + ε(k,l,j)*x[k]*d3x[i] +
ε(k,l,i)*x[k]*d3x[j] + (ε(k,l,j)*x[i] + ε(k,l,i)*x[j])*d3x[k]))*d4x[l] + 6*d2x[l]*(10*d2x[k]*(2*δ(i,j)*(ε(k,l,1)*d3x[1] + ε(k,l,2)*d3x[2] + ε(k,l,3)*d3x[3]) - 3*(ε(k,l,j)*d3x[i] + ε(k,l,i)*d3x[j])) +
10*(2*δ(i,j)*(ε(k,l,1)*d2x[1] + ε(k,l,2)*d2x[2] + ε(k,l,3)*d2x[3]) - 3*(ε(k,l,j)*d2x[i] + ε(k,l,i)*d2x[j]))*d3x[k] + 5*dx[k]*(2*δ(i,j)*(ε(k,l,1)*d4x[1] + ε(k,l,2)*d4x[2] + ε(k,l,3)*d4x[3]) - 3*(ε(k,l,j)*d4x[i] +
ε(k,l,i)*d4x[j])) + 5*(2*δ(i,j)*(ε(k,l,1)*dx[1] + ε(k,l,2)*dx[2] + ε(k,l,3)*dx[3]) - 3*(ε(k,l,j)*dx[i] + ε(k,l,i)*dx[j]))*d4x[k] + x[k]*(2*δ(i,j)*(ε(k,l,1)*d5x[1] + ε(k,l,2)*d5x[2] + ε(k,l,3)*d5x[3]) -
3*(ε(k,l,j)*d5x[i] + ε(k,l,i)*d5x[j])) + (2*δ(i,j)*(ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3]) - 3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j]))*d5x[k]) + 15*(2*δ(i,j)*(2*ε(k,l,1)*dx[1]*dx[k] + 2*ε(k,l,2)*dx[2]*dx[k] +
2*ε(k,l,3)*dx[3]*dx[k] + ε(k,l,1)*x[k]*d2x[1] + ε(k,l,2)*x[k]*d2x[2] + ε(k,l,3)*x[k]*d2x[3] + (ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*d2x[k]) - 3*(2*ε(k,l,j)*dx[i]*dx[k] + 2*ε(k,l,i)*dx[j]*dx[k] +
ε(k,l,j)*x[k]*d2x[i] + ε(k,l,i)*x[k]*d2x[j] + (ε(k,l,j)*x[i] + ε(k,l,i)*x[j])*d2x[k]))*d5x[l] + dx[l]*(40*δ(i,j)*(ε(k,l,1)*d3x[1] + ε(k,l,2)*d3x[2] + ε(k,l,3)*d3x[3])*d3x[k] - 60*(ε(k,l,j)*d3x[i] +
ε(k,l,i)*d3x[j])*d3x[k] + 30*δ(i,j)*d2x[k]*(ε(k,l,1)*d4x[1] + ε(k,l,2)*d4x[2] + ε(k,l,3)*d4x[3]) - 45*d2x[k]*(ε(k,l,j)*d4x[i] + ε(k,l,i)*d4x[j]) + 30*δ(i,j)*(ε(k,l,1)*d2x[1] + ε(k,l,2)*d2x[2] +
ε(k,l,3)*d2x[3])*d4x[k] - 45*(ε(k,l,j)*d2x[i] + ε(k,l,i)*d2x[j])*d4x[k] + 12*δ(i,j)*dx[k]*(ε(k,l,1)*d5x[1] + ε(k,l,2)*d5x[2] + ε(k,l,3)*d5x[3]) - 18*dx[k]*(ε(k,l,j)*d5x[i] + ε(k,l,i)*d5x[j]) +
12*δ(i,j)*(ε(k,l,1)*dx[1] + ε(k,l,2)*dx[2] + ε(k,l,3)*dx[3])*d5x[k] - 18*(ε(k,l,j)*dx[i] + ε(k,l,i)*dx[j])*d5x[k] + 2*δ(i,j)*x[k]*(ε(k,l,1)*d6x[1] + ε(k,l,2)*d6x[2] + ε(k,l,3)*d6x[3]) -
3*x[k]*(ε(k,l,j)*d6x[i] + ε(k,l,i)*d6x[j]) + 2*δ(i,j)*(ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*d6x[k] - 3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j])*d6x[k]) + 6*(-3*x[k]*(ε(k,l,j)*dx[i] + ε(k,l,i)*dx[j]) -
3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j])*dx[k] + 2*δ(i,j)*(x[k]*(ε(k,l,1)*dx[1] + ε(k,l,2)*dx[2] + ε(k,l,3)*dx[3]) + (ε(k,l,1)*x[1] + ε(k,l,2)*x[2] + ε(k,l,3)*x[3])*dx[k]))*d6x[l] + (2*δ(i,j)*(ε(k,l,1)*x[1] +
ε(k,l,2)*x[2] + ε(k,l,3)*x[3]) - 3*(ε(k,l,j)*x[i] + ε(k,l,i)*x[j]))*x[k]*d7x[l])/6.

Sijklm3(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, i::Int64, j::Int64, k::Int64, l::Int64, m::Int64)::Float64 = (3*(2*(-2*δ(j,k)*(ε(l,m,i)*x[1]*dx[1] +
ε(l,m,1)*x[i]*dx[1] + ε(l,m,i)*x[2]*dx[2] + ε(l,m,2)*x[i]*dx[2] + ε(l,m,i)*x[3]*dx[3] + ε(l,m,3)*x[i]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[i]) - 2*δ(i,k)*(ε(l,m,j)*x[1]*dx[1] +
ε(l,m,1)*x[j]*dx[1] + ε(l,m,j)*x[2]*dx[2] + ε(l,m,2)*x[j]*dx[2] + ε(l,m,j)*x[3]*dx[3] + ε(l,m,3)*x[j]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[j]) - 2*δ(i,j)*(ε(l,m,k)*x[1]*dx[1] +
ε(l,m,1)*x[k]*dx[1] + ε(l,m,k)*x[2]*dx[2] + ε(l,m,2)*x[k]*dx[2] + ε(l,m,k)*x[3]*dx[3] + ε(l,m,3)*x[k]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[k]) + 5*(ε(l,m,k)*x[j]*dx[i] + ε(l,m,j)*x[k]*dx[i] +
ε(l,m,k)*x[i]*dx[j] + ε(l,m,i)*x[k]*dx[j] + (ε(l,m,j)*x[i] + ε(l,m,i)*x[j])*dx[k]))*dx[l] + x[l]*(-2*δ(j,k)*(ε(l,m,i)*dx[1]^2 + ε(l,m,i)*dx[2]^2 + ε(l,m,i)*dx[3]^2 + 2*ε(l,m,1)*dx[1]*dx[i] + 2*ε(l,m,2)*dx[2]*dx[i] +
2*ε(l,m,3)*dx[3]*dx[i] + ε(l,m,i)*x[1]*d2x[1] + ε(l,m,1)*x[i]*d2x[1] + ε(l,m,i)*x[2]*d2x[2] + ε(l,m,2)*x[i]*d2x[2] + ε(l,m,i)*x[3]*d2x[3] + ε(l,m,3)*x[i]*d2x[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] +
ε(l,m,3)*x[3])*d2x[i]) - 2*δ(i,k)*(ε(l,m,j)*dx[1]^2 + ε(l,m,j)*dx[2]^2 + ε(l,m,j)*dx[3]^2 + 2*ε(l,m,1)*dx[1]*dx[j] + 2*ε(l,m,2)*dx[2]*dx[j] + 2*ε(l,m,3)*dx[3]*dx[j] + ε(l,m,j)*x[1]*d2x[1] + ε(l,m,1)*x[j]*d2x[1] +
ε(l,m,j)*x[2]*d2x[2] + ε(l,m,2)*x[j]*d2x[2] + ε(l,m,j)*x[3]*d2x[3] + ε(l,m,3)*x[j]*d2x[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*d2x[j]) - 2*δ(i,j)*(ε(l,m,k)*dx[1]^2 + ε(l,m,k)*dx[2]^2 +
ε(l,m,k)*dx[3]^2 + 2*ε(l,m,1)*dx[1]*dx[k] + 2*ε(l,m,2)*dx[2]*dx[k] + 2*ε(l,m,3)*dx[3]*dx[k] + ε(l,m,k)*x[1]*d2x[1] + ε(l,m,1)*x[k]*d2x[1] + ε(l,m,k)*x[2]*d2x[2] + ε(l,m,2)*x[k]*d2x[2] + ε(l,m,k)*x[3]*d2x[3] +
ε(l,m,3)*x[k]*d2x[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*d2x[k]) + 5*(2*ε(l,m,i)*dx[j]*dx[k] + 2*dx[i]*(ε(l,m,k)*dx[j] + ε(l,m,j)*dx[k]) + ε(l,m,k)*x[j]*d2x[i] + ε(l,m,j)*x[k]*d2x[i] +
ε(l,m,k)*x[i]*d2x[j] + ε(l,m,i)*x[k]*d2x[j] + (ε(l,m,j)*x[i] + ε(l,m,i)*x[j])*d2x[k])) + (-(δ(j,k)*(ε(l,m,i)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[i])) -
δ(i,k)*(ε(l,m,j)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[j]) - δ(i,j)*(ε(l,m,k)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[k]) +
5*(ε(l,m,i)*x[j]*x[k] + x[i]*(ε(l,m,k)*x[j] + ε(l,m,j)*x[k])))*d2x[l])*d2x[m] + dx[m]*(-6*δ(j,k)*dx[l]*(ε(l,m,i)*dx[1]^2 + ε(l,m,i)*dx[2]^2 + ε(l,m,i)*dx[3]^2 + 2*ε(l,m,1)*dx[1]*dx[i] +
2*ε(l,m,2)*dx[2]*dx[i] + 2*ε(l,m,3)*dx[3]*dx[i] + ε(l,m,i)*x[1]*d2x[1] + ε(l,m,1)*x[i]*d2x[1] + ε(l,m,i)*x[2]*d2x[2] + ε(l,m,2)*x[i]*d2x[2] + ε(l,m,i)*x[3]*d2x[3] + ε(l,m,3)*x[i]*d2x[3] + (ε(l,m,1)*x[1] +
ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*d2x[i]) - 6*δ(i,k)*dx[l]*(ε(l,m,j)*dx[1]^2 + ε(l,m,j)*dx[2]^2 + ε(l,m,j)*dx[3]^2 + 2*ε(l,m,1)*dx[1]*dx[j] + 2*ε(l,m,2)*dx[2]*dx[j] + 2*ε(l,m,3)*dx[3]*dx[j] + ε(l,m,j)*x[1]*d2x[1] +
ε(l,m,1)*x[j]*d2x[1] + ε(l,m,j)*x[2]*d2x[2] + ε(l,m,2)*x[j]*d2x[2] + ε(l,m,j)*x[3]*d2x[3] + ε(l,m,3)*x[j]*d2x[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*d2x[j]) - 6*δ(i,j)*dx[l]*(ε(l,m,k)*dx[1]^2 +
ε(l,m,k)*dx[2]^2 + ε(l,m,k)*dx[3]^2 + 2*ε(l,m,1)*dx[1]*dx[k] + 2*ε(l,m,2)*dx[2]*dx[k] + 2*ε(l,m,3)*dx[3]*dx[k] + ε(l,m,k)*x[1]*d2x[1] + ε(l,m,1)*x[k]*d2x[1] + ε(l,m,k)*x[2]*d2x[2] + ε(l,m,2)*x[k]*d2x[2] +
ε(l,m,k)*x[3]*d2x[3] + ε(l,m,3)*x[k]*d2x[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*d2x[k]) + 15*dx[l]*(2*ε(l,m,i)*dx[j]*dx[k] + 2*dx[i]*(ε(l,m,k)*dx[j] + ε(l,m,j)*dx[k]) + ε(l,m,k)*x[j]*d2x[i] +
ε(l,m,j)*x[k]*d2x[i] + ε(l,m,k)*x[i]*d2x[j] + ε(l,m,i)*x[k]*d2x[j] + (ε(l,m,j)*x[i] + ε(l,m,i)*x[j])*d2x[k]) - 6*δ(j,k)*(ε(l,m,i)*x[1]*dx[1] + ε(l,m,1)*x[i]*dx[1] + ε(l,m,i)*x[2]*dx[2] + ε(l,m,2)*x[i]*dx[2] +
ε(l,m,i)*x[3]*dx[3] + ε(l,m,3)*x[i]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[i])*d2x[l] - 6*δ(i,k)*(ε(l,m,j)*x[1]*dx[1] + ε(l,m,1)*x[j]*dx[1] + ε(l,m,j)*x[2]*dx[2] + ε(l,m,2)*x[j]*dx[2] +
ε(l,m,j)*x[3]*dx[3] + ε(l,m,3)*x[j]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[j])*d2x[l] - 6*δ(i,j)*(ε(l,m,k)*x[1]*dx[1] + ε(l,m,1)*x[k]*dx[1] + ε(l,m,k)*x[2]*dx[2] + ε(l,m,2)*x[k]*dx[2] +
ε(l,m,k)*x[3]*dx[3] + ε(l,m,3)*x[k]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[k])*d2x[l] + 15*(ε(l,m,k)*x[j]*dx[i] + ε(l,m,j)*x[k]*dx[i] + ε(l,m,k)*x[i]*dx[j] + ε(l,m,i)*x[k]*dx[j] +
(ε(l,m,j)*x[i] + ε(l,m,i)*x[j])*dx[k])*d2x[l] - 2*δ(j,k)*x[l]*(3*dx[i]*(ε(l,m,1)*d2x[1] + ε(l,m,2)*d2x[2] + ε(l,m,3)*d2x[3]) + 3*ε(l,m,2)*dx[2]*d2x[i] + 3*ε(l,m,3)*dx[3]*d2x[i] + 3*dx[1]*(ε(l,m,i)*d2x[1] +
ε(l,m,1)*d2x[i]) + ε(l,m,1)*x[i]*d3x[1] + ε(l,m,2)*x[i]*d3x[2] + ε(l,m,3)*x[i]*d3x[3] + ε(l,m,i)*(3*dx[2]*d2x[2] + 3*dx[3]*d2x[3] + x[1]*d3x[1] + x[2]*d3x[2] + x[3]*d3x[3]) + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] +
ε(l,m,3)*x[3])*d3x[i]) - 2*δ(i,k)*x[l]*(3*dx[j]*(ε(l,m,1)*d2x[1] + ε(l,m,2)*d2x[2] + ε(l,m,3)*d2x[3]) + 3*ε(l,m,2)*dx[2]*d2x[j] + 3*ε(l,m,3)*dx[3]*d2x[j] + 3*dx[1]*(ε(l,m,j)*d2x[1] + ε(l,m,1)*d2x[j]) +
ε(l,m,1)*x[j]*d3x[1] + ε(l,m,2)*x[j]*d3x[2] + ε(l,m,3)*x[j]*d3x[3] + ε(l,m,j)*(3*dx[2]*d2x[2] + 3*dx[3]*d2x[3] + x[1]*d3x[1] + x[2]*d3x[2] + x[3]*d3x[3]) + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*d3x[j]) -
2*δ(i,j)*x[l]*(3*dx[k]*(ε(l,m,1)*d2x[1] + ε(l,m,2)*d2x[2] + ε(l,m,3)*d2x[3]) + 3*ε(l,m,2)*dx[2]*d2x[k] + 3*ε(l,m,3)*dx[3]*d2x[k] + 3*dx[1]*(ε(l,m,k)*d2x[1] + ε(l,m,1)*d2x[k]) + ε(l,m,1)*x[k]*d3x[1] +
ε(l,m,2)*x[k]*d3x[2] + ε(l,m,3)*x[k]*d3x[3] + ε(l,m,k)*(3*dx[2]*d2x[2] + 3*dx[3]*d2x[3] + x[1]*d3x[1] + x[2]*d3x[2] + x[3]*d3x[3]) + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*d3x[k]) +
5*x[l]*(3*ε(l,m,k)*dx[i]*d2x[j] + 3*dx[k]*(ε(l,m,j)*d2x[i] + ε(l,m,i)*d2x[j]) + 3*ε(l,m,j)*dx[i]*d2x[k] + 3*dx[j]*(ε(l,m,k)*d2x[i] + ε(l,m,i)*d2x[k]) + ε(l,m,k)*x[j]*d3x[i] + ε(l,m,j)*x[k]*d3x[i] +
ε(l,m,k)*x[i]*d3x[j] + ε(l,m,i)*x[k]*d3x[j] + (ε(l,m,j)*x[i] + ε(l,m,i)*x[j])*d3x[k]) - δ(j,k)*(ε(l,m,i)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[i])*d3x[l] -
δ(i,k)*(ε(l,m,j)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[j])*d3x[l] - δ(i,j)*(ε(l,m,k)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] +
ε(l,m,3)*x[3])*x[k])*d3x[l] + 5*(ε(l,m,i)*x[j]*x[k] + x[i]*(ε(l,m,k)*x[j] + ε(l,m,j)*x[k]))*d3x[l]) + 3*(x[l]*(-2*δ(j,k)*(ε(l,m,i)*x[1]*dx[1] + ε(l,m,1)*x[i]*dx[1] + ε(l,m,i)*x[2]*dx[2] + ε(l,m,2)*x[i]*dx[2] +
ε(l,m,i)*x[3]*dx[3] + ε(l,m,3)*x[i]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[i]) - 2*δ(i,k)*(ε(l,m,j)*x[1]*dx[1] + ε(l,m,1)*x[j]*dx[1] + ε(l,m,j)*x[2]*dx[2] + ε(l,m,2)*x[j]*dx[2] +
ε(l,m,j)*x[3]*dx[3] + ε(l,m,3)*x[j]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[j]) - 2*δ(i,j)*(ε(l,m,k)*x[1]*dx[1] + ε(l,m,1)*x[k]*dx[1] + ε(l,m,k)*x[2]*dx[2] + ε(l,m,2)*x[k]*dx[2] +
ε(l,m,k)*x[3]*dx[3] + ε(l,m,3)*x[k]*dx[3] + (ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*dx[k]) + 5*(ε(l,m,k)*x[j]*dx[i] + ε(l,m,j)*x[k]*dx[i] + ε(l,m,k)*x[i]*dx[j] + ε(l,m,i)*x[k]*dx[j] + (ε(l,m,j)*x[i] +
ε(l,m,i)*x[j])*dx[k])) + (-(δ(j,k)*(ε(l,m,i)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[i])) - δ(i,k)*(ε(l,m,j)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] +
ε(l,m,3)*x[3])*x[j]) - δ(i,j)*(ε(l,m,k)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[k]) + 5*(ε(l,m,i)*x[j]*x[k] + x[i]*(ε(l,m,k)*x[j] + ε(l,m,j)*x[k])))*dx[l])*d3x[m] +
(-(δ(j,k)*(ε(l,m,i)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[i])) - δ(i,k)*(ε(l,m,j)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[j]) -
δ(i,j)*(ε(l,m,k)*(x[1]^2 + x[2]^2 + x[3]^2) + 2*(ε(l,m,1)*x[1] + ε(l,m,2)*x[2] + ε(l,m,3)*x[3])*x[k]) + 5*(ε(l,m,i)*x[j]*x[k] + x[i]*(ε(l,m,k)*x[j] + ε(l,m,j)*x[k])))*x[l]*d4x[m])/15.

Sij2(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = Sijkl2(x, dx, d2x, d3x, i, j, 1, 1) + Sijkl2(x, dx, d2x, d3x, i, j, 1, 2) +
Sijkl2(x, dx, d2x, d3x, i, j, 1, 3) + Sijkl2(x, dx, d2x, d3x, i, j, 2, 1) + Sijkl2(x, dx, d2x, d3x, i, j, 2, 2) + Sijkl2(x, dx, d2x, d3x, i, j, 2, 3) + Sijkl2(x, dx, d2x, d3x, i, j, 3, 1) +
Sijkl2(x, dx, d2x, d3x, i, j, 3, 2) + Sijkl2(x, dx, d2x, d3x, i, j, 3, 3)

Sij5(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 1, 1) +
Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 1, 2) + Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 1, 3) + Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 2, 1) + Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 2, 2) +
Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 2, 3) + Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 3, 1) + Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 3, 2) + Sijkl5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j, 3, 3)

Sij6(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, i::Int64, j::Int64)::Float64 = Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 1, 1) +
Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 1, 2) + Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 1, 3) + Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 2, 1) +
Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 2, 2) + Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 2, 3) + Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 3, 1) +
Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 3, 2) + Sijkl6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, 3, 3)

Sijk3(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, i::Int64, j::Int64, k::Int64)::Float64 = Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 1, 1) +
Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 1, 2) + Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 1, 3) + Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 2, 1) + Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 2, 2) +
Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 2, 3) + Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 3, 1) + Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 3, 2) + Sijklm3(x, dx, d2x, d3x, d4x, i, j, k, 3, 3)


Mij2(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Mass_quad_prefactor(q) * Mij2(x, dx, d2x, i, j) 

Mij3(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Mass_quad_prefactor(q) * Mij3(x, dx, d2x, d3x, i, j)

Mij4(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Mass_quad_prefactor(q) * Mij4(x, dx, d2x, d3x, d4x, i, j)

Mij5(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Mass_quad_prefactor(q) * Mij5(x, dx, d2x, d3x, d4x, d5x, i, j) 

Mij6(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Mass_quad_prefactor(q) * Mij6(x, dx, d2x, d3x, d4x, d5x, d6x, i, j) 

Mij7(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Mass_quad_prefactor(q) * Mij7(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j) 

Mij8(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, d8x::AbstractVector{Float64}, q::Float64, i::Int64,
j::Int64)::Float64 = Mass_quad_prefactor(q) * Mij8(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, d8x, i, j) 

Mijk3(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64, k::Int64)::Float64 = Mass_oct_prefactor(q) * Mijk3(x, dx, d2x, d3x, i, j, k)

Mijk7(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64,
k::Int64)::Float64 = Mass_oct_prefactor(q) * Mijk7(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j, k) 

Mijk8(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, d8x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64,
k::Int64)::Float64 = Mass_oct_prefactor(q) * Mijk8(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, d8x, i, j, k) 

Mijkl(x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64, k::Int64, l::Int64)::Float64 = Mass_hex_prefactor(q) * Mijkl(x, i, j, k, l)

Mijkl4(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64, k::Int64, l::Int64)::Float64 = Mass_hex_prefactor(q) * Mijkl4(x, dx, d2x, d3x, d4x, i, j, k, l)

Sij2(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Current_quad_prefactor(q) * Sij2(x, dx, d2x, d3x, i, j) 

Sij5(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Current_quad_prefactor(q) * Sij5(x, dx, d2x, d3x, d4x, d5x, d6x, i, j)

Sij6(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, d5x::AbstractVector{Float64}, d6x::AbstractVector{Float64}, d7x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64)::Float64 = Current_quad_prefactor(q) * Sij6(x, dx, d2x, d3x, d4x, d5x, d6x, d7x, i, j) 

Sijk3(x::AbstractVector{Float64}, dx::AbstractVector{Float64}, d2x::AbstractVector{Float64}, d3x::AbstractVector{Float64}, d4x::AbstractVector{Float64}, q::Float64, i::Int64, j::Int64, k::Int64)::Float64 = Current_oct_prefactor(q) * Sijk3(x, dx, d2x, d3x, d4x, i, j, k)

# computes the mulitpole derivatives necessary for computing the self-force
@views function AnalyticMultipoleDerivs_SF!(x::AbstractVector{Float64}, dx_dt::AbstractVector{Float64}, d2x_dt::AbstractVector{Float64}, d3x_dt::AbstractVector{Float64}, d4x_dt::AbstractVector{Float64}, d5x_dt::AbstractVector{Float64}, d6x_dt::AbstractVector{Float64}, d7x_dt::AbstractVector{Float64}, d8x_dt::AbstractVector{Float64}, q::Float64, Mij5::AbstractArray, Mij6::AbstractArray, Mij7::AbstractArray, Mij8::AbstractArray, Mijk7::AbstractArray, Mijk8::AbstractArray, Sij5::AbstractArray, Sij6::AbstractArray, lmax_mass::Int64, lmax_current::Int64)
    @inbounds for i=1:3, j=1:3
        if lmax_mass >= 2
            Mij5[i, j] = AnalyticMultipoleDerivs.Mij5(x, dx_dt, d2x_dt, d3x_dt, d4x_dt, d5x_dt, q, i, j)
            Mij6[i, j] = AnalyticMultipoleDerivs.Mij6(x, dx_dt, d2x_dt, d3x_dt, d4x_dt, d5x_dt, d6x_dt, q, i, j)
            Mij7[i, j] = AnalyticMultipoleDerivs.Mij7(x, dx_dt, d2x_dt, d3x_dt, d4x_dt, d5x_dt, d6x_dt, d7x_dt, q, i, j)
            Mij8[i, j] = AnalyticMultipoleDerivs.Mij8(x, dx_dt, d2x_dt, d3x_dt, d4x_dt, d5x_dt, d6x_dt, d7x_dt, d8x_dt, q, i, j)
        end
        if lmax_current >= 2
            Sij5[i, j] = AnalyticMultipoleDerivs.Sij5(x, dx_dt, d2x_dt, d3x_dt, d4x_dt, d5x_dt, d6x_dt, q, i, j)
            Sij6[i, j] = AnalyticMultipoleDerivs.Sij6(x, dx_dt, d2x_dt, d3x_dt, d4x_dt, d5x_dt, d6x_dt, d7x_dt, q, i, j)
        end

        if lmax_mass >= 3
            @inbounds for k=1:3
                Mijk7[i, j, k] = AnalyticMultipoleDerivs.Mijk7(x, dx_dt, d2x_dt, d3x_dt, d4x_dt, d5x_dt, d6x_dt, d7x_dt, q, i, j, k)
                Mijk8[i, j, k] = AnalyticMultipoleDerivs.Mijk8(x, dx_dt, d2x_dt, d3x_dt, d4x_dt, d5x_dt, d6x_dt, d7x_dt, d8x_dt, q, i, j, k)
            end
        end
    end
end


# computes the mulitpole derivatives necessary for computing the waveform
@views function AnalyticMultipoleDerivs_WF!(r::Float64, θ::Float64, ϕ::Float64, r_dot::Float64, θ_dot::Float64, ϕ_dot::Float64, r_ddot::Float64, θ_ddot::Float64, ϕ_ddot::Float64, Mij2::AbstractArray, Mijk3::AbstractArray, Mijkl4::AbstractArray, Sij2::AbstractArray, Sijk3::AbstractArray, a::Float64, q::Float64, E::Float64, L::Float64, C::Float64, lmax_mass::Int64, lmax_current::Int64, xBL::AbstractVector{Float64}, vBL::AbstractVector{Float64}, aBL::AbstractVector{Float64}, dxBL_dt::AbstractVector{Float64}, d2xBL_dt::AbstractVector{Float64}, d3xBL_dt::AbstractVector{Float64}, d4xBL_dt::AbstractVector{Float64}, d5xBL_dt::AbstractVector{Float64}, d6xBL_dt::AbstractVector{Float64}, d7xBL_dt::AbstractVector{Float64}, d8xBL_dt::AbstractVector{Float64},dx_dλ::AbstractVector{Float64}, d2x_dλ::AbstractVector{Float64}, d3x_dλ::AbstractVector{Float64}, d4x_dλ::AbstractVector{Float64}, d5x_dλ::AbstractVector{Float64}, d6x_dλ::AbstractVector{Float64},d7x_dλ::AbstractVector{Float64}, d8x_dλ::AbstractVector{Float64},xH::AbstractVector{Float64}, dxH_dt::AbstractVector{Float64}, d2xH_dt::AbstractVector{Float64},d3xH_dt::AbstractVector{Float64}, d4xH_dt::AbstractVector{Float64}, d5xH_dt::AbstractVector{Float64},d6xH_dt::AbstractVector{Float64}, d7xH_dt::AbstractVector{Float64}, d8xH_dt::AbstractVector{Float64}) 
    ### COMPUTE BL COORDINATE DERIVATIVES ###
    xBL[1] = r; xBL[2] = θ; xBL[3] = ϕ;
    vBL[1] = r_dot; vBL[2] = θ_dot; vBL[3] = ϕ_dot;
    aBL[1] = r_ddot; aBL[2] = θ_ddot; aBL[3] = ϕ_ddot;

    AnalyticCoordinateDerivs.ComputeDerivs!(xBL, sign(vBL[1]), sign(vBL[2]), dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,
    dx_dλ, d2x_dλ, d3x_dλ, d4x_dλ, d5x_dλ, d6x_dλ, d7x_dλ, d8x_dλ, a, E, L, C)

    ### COMPUTE HARMONIC COORDINATE DERIVATIVES ###
    HarmonicCoordDerivs.compute_harmonic_derivs!(xBL, dxBL_dt, d2xBL_dt, d3xBL_dt, d4xBL_dt, d5xBL_dt, d6xBL_dt, d7xBL_dt, d8xBL_dt,
    xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, d5xH_dt, d6xH_dt, d7xH_dt, d8xH_dt, a)
    
    @inbounds for i=1:3, j=1:3
        if lmax_mass >= 2
            Mij2[i, j] = AnalyticMultipoleDerivs.Mij2(xH, dxH_dt, d2xH_dt, q, i, j)
        end

        if lmax_current >= 2
            Sij2[i, j] = AnalyticMultipoleDerivs.Sij2(xH, dxH_dt, d2xH_dt, d3xH_dt, q, i, j)
        end

        @inbounds for k=1:3
            if lmax_mass >= 3
                Mijk3[i, j, k] = AnalyticMultipoleDerivs.Mijk3(xH, dxH_dt, d2xH_dt, d3xH_dt, q, i, j, k)
            end

            if lmax_current == 3
                Sijk3[i, j, k] = AnalyticMultipoleDerivs.Sijk3(xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, q, i, j, k)
            end

            if lmax_mass == 4
                @inbounds for l=1:3
                    Mijkl4[i, j, k, l] = AnalyticMultipoleDerivs.Mijkl4(xH, dxH_dt, d2xH_dt, d3xH_dt, d4xH_dt, q, i, j, k, l)
                end
            end
        end
    end
end

end