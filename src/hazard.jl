"""
    ConstantHazard <: AbstractHazard

Implements a ConstantHazard. This implements a constant 1/λ chance of a changepoint occuring

arguments:
    `λ`<:Real

"""
struct ConstantHazard <: AbstractHazard
    λ::Real
end

function (h::ConstantHazard)(x::AbstractArray)
    1/h.λ * ones(length(x))
end
