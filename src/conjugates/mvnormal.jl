# MvNormal distribution with unknown mean and variance
# Normal Inverse-Wishart Distribution as Conjugate Prior
# Student T Distribution and predictive posterior
# See https://en.wikipedia.org/wiki/Conjugate_prior
mutable struct ConjugateModel{MvNormal, T} <: AbstractConjugateModel
    μ0::AbstractVector{T}
    Ψ0::AbstractMatrix{T}
    ν0::T
    κ0::T
    μ::Array{<:AbstractVector{T}}
    Ψ::Array{<:AbstractMatrix{T}}
    ν::AbstractArray{T}
    κ::AbstractArray{T}
end

function ConjugateModel{MvNormal, T}(
    μ::AbstractVector{T},
    Ψ::AbstractMatrix{T},
    ν::T,
    κ::T,
    ) where {T <: Real}
    ConjugateModel{MvNormal, T}(μ, Ψ, ν, κ, [μ], [Ψ], [ν], [κ])
end


# Convenience constructor using directly the ConjugatePrior
# pri = NormalInverseWishart(mu0, kappa0, T0, nu0)
function ConjugateModel{MvNormal, T}(
    prior::NormalInverseWishart
    ) where {T}

    μ = prior.mu
    κ = prior.kappa
    ν = prior.nu
    Ψ = prior.Lamchol
    ConjugateModel{MvNormal, T}(μ, Matrix(Ψ), ν, κ)
end

function update_theta!(d::ConjugateModel{MvNormal, T}, x::AbstractVector) where {T}

    # We want to get the update but also apply the update to each
    N = 1
    L = length(d.μ)

    μ = Array{Vector{T}}(undef, L)
    κ = Array{T}(undef, L)
    ν = Array{T}(undef, L)
    Ψ = Array{Matrix{T}}(undef, L)

    # Rather than using broadcasting write as a map as it's a clearer
    map(1:length(d.μ)) do IDX
        μ[IDX] = ((d.κ[IDX] .* d.μ[IDX]) + N.*x) ./ (d.κ[IDX] .+ N)
        κ[IDX] = d.κ[IDX] .+ N
        ν[IDX] = d.ν[IDX] .+ N
        Ψ[IDX] = d.Ψ[IDX] .+ (x .- d.μ[IDX])*(x .- d.μ[IDX])' * (N * d.κ[IDX] / (d.κ[IDX] .+ N))
    end

    d.μ = vcat([d.μ0], μ)
    d.κ = vcat([d.κ0], κ)
    d.ν = vcat([d.ν0], ν)
    d.Ψ = vcat([d.Ψ0], Ψ)
end

function posterior_predictive(d::ConjugateModel{MvNormal, T}) where {T}
    D = length(d.μ0)
    _prefactor = (d.κ .+ 1) ./ (d.κ .* (d.ν .- D .+ 1))
    Ψ = _prefactor .* d.Ψ
    Ψ = Symmetric.(condition.(Ψ))
    @assert all(isposdef.(Ψ))
    GenericMvTDist.(d.ν .- D .+ 1, d.μ, PDMat.(Ψ))
end
