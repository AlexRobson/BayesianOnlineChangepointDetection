@testset "Hazard" begin

    @testset "Constant Hazard" begin

        h = ConstantHazard(2.0)

        @test h isa AbstractHazard
        @test h(1:5) == 0.5 .* ones(5,)

    end


end
