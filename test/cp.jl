@testset "CP Functions" begin

    DIM = 2
    TLENGTH = 10
    data = rand(TLENGTH, DIM)
    μ = zeros(DIM)
    Σ = Matrix{Float64}(I, 2, 2)

    @testset "ConjugateModel" begin

        # Test the Underlying probability model can be instantiated
        @test ConjugateModel{MvNormal, Float64}(μ, Σ, 1.0, 1.0, [μ], [Σ], [1.0], [1.0]) isa AbstractConjugateModel
        @test ConjugateModel{MvNormal, Float64}(μ, Σ, 1.0, 1.0) isa AbstractConjugateModel

        # Test this returns a value
        m = ConjugateModel{MvNormal, Float64}(μ, Σ, 2.0, 1.0)
        p = predprob(m, zeros(2))
        @test isapprox(predprob(m, zeros(2))[1], 0.080; atol = 0.01)

    end

    @testset "online_changepoint_detection" begin

        R, maxes, obslikelihood, pred, dists, L = online_changepoint_detection(
            data,
            ConstantHazard(2.0),
            ConjugateModel{MvNormal, Float64}(μ, Σ, 2.0, 2.0),
        )

        @test R isa Matrix
        @test maxes isa Array{Int}
        @test obslikelihood isa BOCP.ConjugateModel
        @test pred isa Array{Float64}
        @test dists isa Array{<:Sampleable}
        @test L isa Int

        @test isapprox(exp(pred[end]), 0.3105, atol=0.001)

    end
end

@testset "Test Models Integration" begin

    f(x) = 10 + 10 * x + 5 * x^2 + 10 * sin(2π*x)
    g(x1, x2, x3, x4, x5) = [f(x1), 2*f(x2), 3*f(x3)]

    X_train = NamedDimsArray{(:features, :observations)}(randn(5, 100))
    y_train = mapslices(x -> g(x...), X_train, dims=:features)
    y_train = NamedDimsArray{(:variates, :observations)}(y_train.data)

    function test_pipeline(template, X_test; kwargs...)
      fitted_model = fit(template, y_train, X_train)
      p = predict(fitted_model, X_test; kwargs...)
      @test p isa Array{<:Sampleable}
      @test length(p) == size(X_test, :observations)
      @test length(p[1].components[1]) == size(y_train, :variates) # Get the dim for MixtureModel
      return fitted_model
    end

    mu0 = rand(3)
    kappa0 = 3.
    nu0 = 4.
    T0 = Matrix(1.0I, 3, 3)
    T0[1,2] = T0[2,1] = .5

    prior = NormalInverseWishart(mu0, kappa0, T0, nu0)
    template = ChangePointTemplate{MvNormal}(
        ConstantHazard(10),
        prior
    )

    test_pipeline(template, X_train)
    test_interface(
        template;
        inputs=rand(3, 5),
        outputs=rand(3,5),
        distribution_inputs = [MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(m ^ 2, 3))) for m in 1:5]
    )

end
