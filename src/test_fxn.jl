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


@testset "Testing CubeDistanceSquaredAffine function" begin

    function make_instance(A, b, r)
        return CubeDistanceSquaredAffine(A, b, r)
    end

    function test_instantiation()
        @info "Testing CubeDistanceSquaredAffine instantiation."
        A = randn(4, 6)
        b = randn(4)
        f = make_instance(A, b, 1.5)
        return f isa CubeDistanceSquaredAffine
    end

    function test_fval_inside_box()
        @info "Testing f=0 when Ax-b is strictly inside the box."
        # A=I, b=0, r=2: x=[1,1,1] -> Ax-b=[1,1,1] inside [-2,2]^3 -> prox=0 -> f=0
        A = Matrix{Float64}(I, 3, 3)
        b = zeros(3)
        f = make_instance(A, b, 2.0)
        grad = zeros(3)
        fval = grad_and_fxnval!(f, grad, ones(3))
        return isapprox(fval, 0.0) && isapprox(grad, zeros(3))
    end

    function test_fval_and_grad_known()
        @info "Testing known function value and gradient."
        # A=I, b=0, r=1: x=[2,0] -> Ax-b=[2,0] -> prox=[1,0] -> f=0.5, grad=[1,0]
        A = Matrix{Float64}(I, 2, 2)
        b = zeros(2)
        f = make_instance(A, b, 1.0)
        grad = zeros(2)
        fval = grad_and_fxnval!(f, grad, [2.0, 0.0])
        return isapprox(fval, 0.5) && isapprox(grad, [1.0, 0.0])
    end

    function test_gradient_finitediff()
        @info "Testing gradient via finite differences."
        m, n = 4, 5
        A = randn(m, n); b = randn(m)
        f = make_instance(A, b, 0.5)
        x = randn(n)
        grad = zeros(n)
        grad_and_fxnval!(f, grad, x)
        ε = 1e-6
        grad_fd = zeros(n)

        fval(xp) = begin
            q = A*xp - b
            sum(abs2, @. max(abs(q) - 0.5, 0) * sign(q)) / 2
        end
        
        for i in 1:n
            xp = copy(x); xp[i] += ε
            xm = copy(x); xm[i] -= ε
            grad_fd[i] = (fval(xp) - fval(xm)) / (2ε)
        end
        return isapprox(grad, grad_fd; atol=1e-5)
    end

    function test_fval_callable()
        @info "Testing function value via callable interface."
        # A=I, b=0, r=1, x=[2,0] -> Ax-b=[2,0] -> prox=[1,0] -> f=0.5
        A = Matrix{Float64}(I, 2, 2)
        b = zeros(2)
        f = make_instance(A, b, 1.0)
        return isapprox(f([2.0, 0.0]), 0.5)
    end

    function test_fval_callable_nonzero_b()
        @info "Testing callable with nonzero b."
        # A=I, b=[1,0], r=1, x=[3,0] -> Ax-b=[2,0] -> prox=[1,0] -> f=0.5
        A = Matrix{Float64}(I, 2, 2)
        b = [1.0, 0.0]
        f = make_instance(A, b, 1.0)
        return isapprox(f([3.0, 0.0]), 0.5)
    end

    @test test_instantiation()
    @test test_fval_inside_box()
    @test test_fval_and_grad_known()
    @test test_gradient_finitediff()
    @test test_fval_callable()
    @test test_fval_callable_nonzero_b()

end


