using Test
using UnicodePlots
using SparseArrays

include("fxn_properties.jl")
include("actual_fxn.jl")

@testset "Testing ResidualNormSquared function" begin 

    function make_instance(
        A::AbstractVecOrMat, 
        b::AbstractVecOrMat, 
        alpha=1
    )::ResidualNormSquared
        return ResidualNormSquared(A, b, alpha)
    end

    function test_instantiation(
        A::Matrix, 
        b::Vector
    )
        @info "Testing function instantiation and evaluation. The Vector case. "
        α = 1/100
        f = make_instance(A, b, α)
        result = f(zeros(3))
        return isapprox(α*norm(b)^2/2, result)
    end

    function gradient_evaluation()
        @info "Testing function gradient evaluation. "
        α = 1/10
        A = diagm([1, 2, 3])
        b = rand(3)
        x = zeros(3)
        f = make_instance(A, b, α)
        grad = zeros(3)
        # gradient at x
        grad_and_fxnval!(f, grad, x)
        grad_shouldbe = α*A*(A*x - b)
        return isapprox(grad_shouldbe, grad)
    end
    
    function glipz_test()
        A = rand(3,3); b = rand(3)
        f = make_instance(A, b)
        L = glipz(f)
        return isapprox(norm(A'*A), L)
    end

    @test test_instantiation(rand(3,3), rand(3))
    @test gradient_evaluation()
    @test glipz_test()

end


