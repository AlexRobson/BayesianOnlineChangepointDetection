"""
    prior_type(::D) where {D<:Sampleable, T}

Returns the natural prior for the given expontential family likelihood Distribution D.
See https://en.wikipedia.org/wiki/Conjugate_prior

"""
function prior_type end
# This is potentially type piracy as we are extending a function from ConjugatePriors
# Also there are multiple conjugate priors for the same distribution
# These just serve as a convenience for building conjugate models
prior_type(::Type{MvNormal}) = NormalInverseWishart
prior_type(::Type{MvNormalKnownCov}) = MvNormal
#prior_type(::Type{MvNormalKnownMean}) = InverseWishart # Not defined in Distributions
#prior_type(::Type{Normal}) = NormalInverseGamma
#prior_type(::Type{NormalKnownVar}) = Normal
#prior_type(::Type{NormalKnownMean}) = InverseGamma
