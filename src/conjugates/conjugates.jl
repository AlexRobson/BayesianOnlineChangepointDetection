"""
    prior_type(::D) where {D<:Sampleable, T}

Returns the natural prior for the given expontential family likelihood Distribution D.
See https://en.wikipedia.org/wiki/Conjugate_prior

"""
function prior_type end

prior_type(::Type{MvNormal}) = NormalInverseWishart
prior_type(::Type{MvNormalKnownCov}) = MvNormal
# prior_type(::Type{MvNormalKnownMean}) = InverseWishart # Not defined in Distributions
