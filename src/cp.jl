"""
    ChangePointTemplate{D<:Sampleable}

    Specifies a ChangePoint Template for a given likelihood Distribution D.
"""
struct ChangePointTemplate{L} <: Template
    hazard_func::AbstractHazard
    prior::Sampleable
    function ChangePointTemplate{L}(h, p) where {L}
        if typeof(p) <: prior_type(L)
            return new{L}(h, p)
        else
            error("Prior distribution $p is not conjugate to $L")
        end
    end
end

hazard(m::ChangePointTemplate) = m.hazard_func
likelihood(m::ChangePointTemplate{L}) where {L} = L
prior(m::ChangePointTemplate) = m.prior

struct ChangePointModel{L} <: Model
    model::L
    hazard_func::AbstractHazard
    R::AbstractArray # Run length Matrix
    N::Int # Number of training data seen
end

# https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf # Eqn 258
# https://en.wikipedia.org/wiki/Conjugate_prior#cite_note-ppredNt-9
predprob(d::AbstractConjugateModel, x::AbstractVector) = map(m -> pdf(m, x), posterior_predictive(d))
logpredprob(d::AbstractConjugateModel, x::AbstractVector) = map(m -> logpdf(m, x), posterior_predictive(d))

Models.estimate_type(::Type{<:ChangePointTemplate}) = Models.DistributionEstimate
function Models.output_type(
        ::Type{<:ChangePointTemplate{L}}
    ) where {L<:Sampleable{M, V}} where {M, V}
    if M == Distributions.Multivariate
        Models.MultiOutput
    elseif M == Distributions.Univariate
        Models.SingleOutput
    else
        error("Unable to parse $M")
    end
end

Models.estimate_type(::Type{<:ChangePointModel}) = Models.DistributionEstimate
function Models.output_type(
        ::Type{<:ChangePointModel{ConjugateModel{L, T}}}
    ) where {L<:Sampleable{M, V}, T} where {M, V}
    if M == Distributions.Multivariate
        Models.MultiOutput
    elseif M == Distributions.Univariate
        Models.SingleOutput
    else
        error("Unable to parse $M")
    end
end

function Models.fit(
    m::ChangePointTemplate{L},
    outputs::AbstractMatrix{T},
    inputs::AbstractMatrix{T},
    weights=uweights(T, size(outputs, 2))
    ) where {T<:Real, L}

    outputs = NamedDimsArray{(:variates, :observations)}(outputs)
    inputs = NamedDimsArray{(:features, :observations)}(inputs)

    if !(size(outputs, :variates) == m.prior.dim)
        error("Incorrect dimensions in outputs. Expected $(m.prior.dim), got $(size(outputs, :variates))")
    end

    out = offline_changepoint_detection(outputs', hazard(m), ConjugateModel{L, T}(m.prior))
    R, maxes, obsmodel, prob, preddist, N = out
    return ChangePointModel(obsmodel, hazard(m), R, N)

end

function Models.predict(
    m::ChangePointModel,
    inputs::AbstractMatrix;
    inc::Bool=true
    )

    inputs = NamedDimsArray{(:features, :observations)}(inputs)

    L_train = m.N
    L_test = size(inputs, :observations)

    # Extend R
    _R = deepcopy(m.R)
    R = zeros(L_train + L_test + 1, L_train + L_test + 1)
    R[1:L_train + 1, 1:L_train + 1] = _R

    obsmodel = deepcopy(m.model)
    hazard_func = m.hazard_func

    # Build ensemble
    W = R[1:L_train + 1, L_train + 1] # Ensemble weights
    rundists = MixtureModel(posterior_predictive(obsmodel), W)
    T = size(inputs, :observations)

    # TODO: Return rundists first, then sample the following
    if inc == true
        preddist= map(L_train + 1: T + L_train) do t
            x_t = rand(rundists)
            predprobs = _update_with_data!(x_t, t, obsmodel, hazard_func, R)
            W = R[1:t+1, t + 1] # Ensemble weights
            _preddist = MixtureModel(posterior_predictive(obsmodel), W)

            return _preddist
        end
        return preddist
    else
        @assert T==1
        return [rundists]
    end

end

# TODO: Impplement max_run_length
function offline_changepoint_detection(
    data,
    hazard_func,
    obsmodel::ConjugateModel;
    max_run_length = 1000
    )

    L = size(data, 1)
    D = size(data, 2)

    max_rl = min(max_run_length, L)

    maxes = Array{Int}(zeros(L + 1))
    observation_likelihood = deepcopy(obsmodel)
    R = zeros(max_rl + 1, max_rl + 1)
    pred = zeros(L)
    D = Array{Sampleable}([posterior_predictive(obsmodel)[1]]) # Draw from the unconditional prior

    R[1, 1] = 1

    for (t, x) in enumerate(eachslice(data; dims=1))

        predprobs = _update_with_data!(x, t, observation_likelihood, hazard_func, R)

        # Build ensemble
        W = R[1:t+1, t + 1] # Ensemble weights
        if !isprobvec(W)
            error("$W is not a probability vector")
        end
        rundists = MixtureModel(posterior_predictive(observation_likelihood), W)

        # Store the probability of this datum
        pred[t] = predprobs[end]

        # Record the predictive distribution
        push!(D, rundists)

        # Record the maxes
        maxes[t] = findmax(R[:, t])[2]
    end

    return R, maxes, observation_likelihood, pred, Array{Sampleable}(D), L
end

function _update_with_data!(x, t, obsmodel, hazard_func, R)

    # Evaluate the predictive distribution for the new datum under each of
    # the parameters.  This is the standard thing from Bayesian inference.
    predprobs = logpredprob(obsmodel, x)

    # Evaluate the hazard function for this interval
    #H = hazard_func(1:(t+1))
    H = hazard_func(1:t)

    # Evaluate the growth probabilities - shift the probabilities down and to
    # the right, scaled by the hazard function and the predictive
    # probabilities.
    R[2:t+1, t+1] = R[1:t, t] .* exp.(predprobs) .* (1 .- H) .+ eps(0.0)

    # Evaluate the probability that there *was* a changepoint and we're
    # accumulating the mass back down at r = 0.
    R[1, t+1] = sum( R[1:t, t] .* exp.(predprobs) .* H) .+ eps(0.0)# + (t<2 ? eps(0.0) : 0.0)

    # Determine run length distribution
    R[:, t+1] = R[:, t+1] / sum(R[:, t+1])

    # Update the parameter sets for each possible run length.
    update_theta!(obsmodel, x)

    return predprobs

end
