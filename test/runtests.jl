using BOCP
using Test

using BOCP: ChangePointModel, ChangePointTemplate
using BOCP: ConjugateModel
using ConjugatePriors
using BOCP: ConstantHazard, predprob, online_changepoint_detection
using BOCP: fit, predict
using BOCP: condition
using Distributions
using LinearAlgebra
using NamedDims
using Random
using Models
using Models.TestUtils: test_interface

Random.seed!(1)

include("util.jl")
include("hazard.jl")
include("cp.jl")
