"""
    condition(M::AbstractMatrix; ϵ = 1e-4)

    Adds ϵ to the diagonal
"""
condition(M::AbstractMatrix; ϵ = 1e-4) = Symmetric(M .+ ϵ.*diagm(ones(size(M, 1))))

# Formulate predictions
function samplemoments(D::Array{Sampleable}; N = 1000)
    samples = permutedims(cat(rand.(D, N)...; dims = 3), [3, 1, 2])
    samples = samples[1:end, :, :]
    μ_s = mean(samples; dims = 3)[1:end - 1, :, 1]
    Σ_s = map(1:size(samples, 1)) do t cov(samples[t, :, :]; dims = 2) end
    Σ_s = permutedims(cat(Σ_s...; dims = 3), [3, 1, 2])

    r_xy = mapslices(x -> cor(x; dims = 2), samples; dims = [2,3])

    return μ_s, Σ_s, r_xy
end

function samplemoments(D::Sampleable; N = 1000)
    samples = rand(D, N)
    μ_s = mean(samples; dims = 2)[:]
    Σ_s = cov(samples; dims = 2)
    r_xy = cor(samples; dims = 2)
    return μ_s, Σ_s, r_xy
end
