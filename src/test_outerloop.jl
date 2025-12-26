using Test
using UnicodePlots
using SparseArrays
include("import_inner_loop.jl")
include("outer_loop.jl")

@testset "Testing Outer Loop" begin 
    
    function instantiation(
    )::Tuple{IAPGOuterLoopRunner, Vector{Float64}}
        m = 1024
        n = 1024
        l = 1024
        A = diagm(ones(n))
        b = zeros(m)
        C = randn(l, n)
        f = ResidualNormSquared(C, b)
        ω = OneNormFunction(0.01)
        global OuterLoop = IAPGOuterLoopRunner(f, ω, A)

        return OuterLoop, randn(n)

    end
    
    function test_instantioation()::Bool
        @info "Testing Instantiation. "
        OuterLoop, x = instantiation()
        return true
    end

    function test_one_iteration()::Bool
        @test "Test if one iteration is ok for the outerloop. "
    end


    @test test_instantioation()

end