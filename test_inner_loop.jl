using Test
using UnicodePlots
using SparseArrays

@testset "TESTING THE FUNCTION ‖⋅‖_1 " begin
    include("import_inner_loop.jl")

    # Test subjects and function now follows. 
    λ = 1.1
    n = 3
    f = OneNormFunction(λ) 
    y = randn(n)

    function test_function_value()
        @info "Testing function value evaluation. "
        z = f(y)
        @assert isapprox(z, λ*norm(y, 1))
        return true
    end

    function test_resolvent_identity()
        @info "Testing the basic case of the Resolvent identity of proximal operator. "
        zp = similar(y)
        zd = similar(y)
        dprox!(f, zd, y, 1) # zd mutated 
        prox!(f, zp, y, 1)
        @assert isapprox(zp + zd, y)
        return true
    end

    function dval_dprox()
        @info "Testing if dprox returns dual feasible value.  "
        zd = similar(y)
        dprox!(f, zd, y, 1/2) # zd mutated 
        d_val = dval(f, zd)

        return d_val < Inf64
    end

    @test test_function_value()
    @test test_resolvent_identity()
    @test dval_dprox()

end


@testset "TESTING THE INNER LOOP SOLVER" begin
    m = 2^5
    n = 2^5
    A = randn(m, n)
    ω = OneNormFunction(10)
    y = randn(n)
    λ = 0.001
    global InxProx = nothing

    function create_instance()
        InxProx = InexactProximalPoint(A, ω)
        return true
    end

    function try_running_it()
        itr_max = 2^16
        global v = Vector{Float64}()
        @time j = do_ista_iteration!(
            InxProx, y, λ, itr_max=itr_max, epsilon=1e-3, 
            duality_gaps=v, backtracking=true
        )
        p = eval_primal_objective_at_current_point(InxProx, y, λ)
        q = eval_dual_objective_at_current_point(InxProx, y, λ)
        @info "Total number of ISTA iteration to is: $j"
        @info "Primal Objective is: $p"
        @info "Dual Objective is: $q"
        @info "Final Duality Gap = p + q = $(p + q)"
        p = lineplot(1:length(v), v.|>log2, title="Duality Gaps")
        p|>print
        return j < itr_max
    end

    @test create_instance()
    @test try_running_it()

end

