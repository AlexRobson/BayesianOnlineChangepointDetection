"""
    abstract type AbstractConjugateModel end

"""
abstract type AbstractConjugateModel end

"""
    abstract type AbstractHazard end

"""
abstract type AbstractHazard end

"""
    posterior_predictive(
        d::ConjugateModel
    )

Computes the posterior predictive of the Conjugate Model `d`

"""
function posterior_predictive end

"""
    update_theta!(
        d::ConjugateModel
        x::AbstractVector
    )

Updates a conjugate model `d` with data `x`.

"""
function update_theta! end
