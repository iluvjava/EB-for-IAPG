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
        OuterLoop = IAPGOuterLoopRunner(f, ω, A)

        return OuterLoop, randn(n)

    end
    
    function test_instantioation()::Bool
        @info "Testing Instantiation. "
        OuterLoop, x = instantiation()
        return true
    end

    function test_iterations()::Bool
        @info "Test if iteration go ok for the outerloop on simple problem. "
        # Make the test instance 
        m = 8
        n = 8
        l = 8
        A = diagm(randn(n))
        b = zeros(m)
        C = randn(l, n)
        f = ResidualNormSquared(C, b)
        ω = OneNormFunction(0.01)
        OuterLoop = IAPGOuterLoopRunner(
            f, ω, A, error_scale=1, rho=1
        )
        x0 = 10*ones(n)
        Results = run_outerloop_for!(
            OuterLoop, x0, 1e-10, max_itr=65536
        )
        @info "Reporting Results. "

        return true
    end


    @test test_instantioation()
    @test test_iterations()

end