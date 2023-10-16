using Plots
using Distributions
using Distributions: AffineDistribution
using Random
using ConjugatePriors

t1 = (1:50) ./ (2 * pi)
t2 = (51:100) ./ (2 * pi)
t3 = (101:150) ./ (2 * pi)
t = [t1; t2; t3]

μ1 = 0.
μ2 = 1.

σ1 = 1.
σ2 = 2.

d1 = Normal(μ1, σ1)
d2 = Normal(μ2, σ2)

y1 = rand(d1, length(t1))
y2 = rand(d2, length(t2))
y3 = rand(d1, length(t3))
y = [y1; y2; y3]


# Want to identify the changepoint 

μ_0 = -5
σ_0 = 10.0

# Normal with unknown mean and variance
pri = NormalInverseChisq(μ_0, σ_0, 100., 100.)

x = reduce(hcat, [collect(rand(convert(NormalInverseGamma, pri))) for _ in 1:1000])

abstract type AbstractHazard end

struct ConstantHazard <: AbstractHazard
    λ::Real
end

function (h::ConstantHazard)(x::AbstractArray)
    1/h.λ * ones(length(x))
end

function predictive_probability(nics::NormalInverseChisq, x::Real)
    μ, κ, ν, σ2 = nics.μ, nics.κ, nics.ν, nics.σ2
    scale = sqrt(σ2 * (1 + 1/κ))
    td = AffineDistribution(μ, scale, TDist(ν))
    return pdf(td, x)
end

# Bayesian Online Change Point Detection
function bocpd(data, Hazard::AbstractHazard)
    n = length(data)
    R = zeros(n + 1, n + 1)
    R[1,1] = 1
    
    # Initialize with prior parameters for NormalInverseChisq: mean=0, kappa=1, nu=1, sigma2=1
    prior = NormalInverseChisq(0.0, 1.0, 100.0, 100.0)
    params = [prior]

    max_runlength = 0

    for t in 1:n
        # Evaluate predictive distribution
        predprobs = [predictive_probability(p, data[t]) for p in params]
        H = Hazard(1:t)
        
        # Calculate growth probabilities (no changepoint)
        R[2:t+1, t+1] = R[1:t, t] .* predprobs .* (1 .- H)
            
        # Calculate change point probabilities (changepoint)
        R[1, t+1] = sum(R[1:t, t] .* predprobs .* H)
            
            
        # Renormalize
        R[:, t+1] /= sum(R[:, t+1])

        # Update run length distribution and parameters
        max_runlength = t
        newparams = []
        for r in 0:max_runlength
            if R[r+1, t+1] > 1e-6
                if r == 0
                    push!(newparams, prior)
                else
                    sstats = suffstats(Normal, data[t-r+1:t])
                    post = posterior(params[r], sstats)
                    push!(newparams, post)
                end
            else
                push!(newparams, prior)
            end
        end
        params = newparams
    end

    return R
end


hazard = ConstantHazard(100)

R = bocpd(y, hazard)

# Create the key plots

 
p0 = plot(t, y, color = :blue)
plot!(p0, t1, repeat([μ1], length(t1)), ribbon = σ1, linestyle = :dash, color = :lightblue)
plot!(p0, t2, repeat([μ2], length(t2)), ribbon = σ2, linestyle = :dash, color = :lightblue)
plot!(p0, t3, repeat([μ1], length(t3)), ribbon = σ1, linestyle = :dash, color = :lightblue)

p1 = heatmap(R)

p2 = plot(mapslices(argmax, R, dims = 1)')

f = plot(p0, p1, p2,layout = (3,1))

savefig(f, "bocp.png")


