module BOCP

using ConjugatePriors
using Distributions
using Distributions: GenericMvTDist
using LinearAlgebra
using LinearAlgebra: diagm
using Models
using NamedDims
using PDMats
using StatsBase: uweights

include("changepoint_models.jl")
include("util.jl")
include("hazard.jl")
include("./conjugates/conjugates.jl")
include("./conjugates/mvnormal.jl")
include("cp.jl")

export
    AbstractHazard,
    AbstractConjugateModel,
    ConjugateModel,
    ChangePointTemplate,
    ChangePointModel
end
