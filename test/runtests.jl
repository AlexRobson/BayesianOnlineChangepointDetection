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
using FillArrays
using NamedDims
using StableRNGs
using Models
using Models.TestUtils: test_interface

StableRNGs.seed!(1)

include("util.jl")
include("hazard.jl")
include("cp.jl")
