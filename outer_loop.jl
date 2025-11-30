
"""
Collect the results produced by the IAPGOuterLoopRunner. 
"""
struct ResultsCollector

end



struct IAPGOuterLoopRunner
    f::ClCnvxFxn                # differentiable. 
    iprox::InexactProximalPoint # inexact prox operator. 
    collector::ResultsCollector # a collector for collecting results. 

    # Constant parameters. 
    p::Number       # error shrinkage power. 
    E::Number       # initial error constant 
    rho::Number     # A simple constant for relative error. 

    # These will mutate. 
    y::Vector{Float64}
    x::Vector{Float64}
    v::Vector{Float64}
    y_next::Vector{Float64}
    x_next::Vector{Float64}
    v_next::Vector{Float64}
    # inner loop primal dual iterates. 
    z::Vector{Float64}
    v::Vector{Float64} 

    function IAPGOuterLoopRunner(
        f, omega, A, p=2, error_scale=1
    )
        @assert p > 1 
        @assert error_scale > 0

    end



end


### Implementations plans 
### 1. Define all relevant parameters
### 2. Prototype one step of iteration successfully at least. 
### 3. Optimize it while testing it continuously. 

"""
Perform one iteration of inexact proximal gradient method, with line search. 
It mutates parameters instead of re-assigning them or changing them.

The algorithm updates by (y, x, v) ↦ (y⁺, x⁺, v⁺). 
This operator is not inplace hence new memory will be used. 



"""
function _eval_prox_grad!(
    this::IAPGOuterLoopRunner,
    y⁺::Vector{Float64},    # Will mutate 
    z::Vector{Float64},     # Will mutate
    v::Vector{Float64},     # Will mutate
    ∇fy::Vector{Float64},    # Will reference
    y::Vector{Float64},     # Will reference
    fx::Number,             # Will reference
    L::Number,              # Will reference
    ρ::Number,              # Will reference
    ϵk::Number,             # Will reference
)
    iprox = this.iprox
    x⁺ .-= (1/(L + ρ))*∇fy
    do_ista_iteration!(
        iprox
    )

    return nothing
end