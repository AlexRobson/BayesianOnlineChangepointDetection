@testset "util" begin

    @testset "condition" begin 
        r = rand(5,5)
        Σ = r' * r

        Σ_cond = condition(Σ)
        @test Σ_cond isa Symmetric
    end

end
