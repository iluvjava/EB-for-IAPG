
### ============================================================================
### FUNCTION TYPE
### ============================================================================

using LinearAlgebra

"""
λ‖⋅‖_1 
"""
struct OneNormFunction <:ClCnvxFxn
    lambda::Float64

    function OneNormFunction(lambda::Number)
        if lambda < 0 
            error("lambda constant for OneNormFunction can be negative. ")
        end
        new(convert(Float64, lambda))
    end
end

function (this::OneNormFunction)(x::AbstractVecOrMat)
    return (this.lambda)*norm(x, 1)
end

function (this::OneNormFunction)(x::Vector{Float64})
    return (this.lambda)*norm(x, 1)
end

# Traits assignment and implementations for this type

function prox_trait_assign(::OneNormFunction)
    return Proxable()
end

function dval_trait_assigner(::OneNormFunction)
    return HasFenchelDual()
end


"""
Proximal operator of the one-norm is the soft thresholding operator. 
It computes: 
prox[x ↦ λ‖ηx‖_1, ρ](y)
"""
function prox!(
    ::Proxable, 
    this::OneNormFunction, 
    y::Union{Array{Float64}, Float64}, 
    y_out::Union{Array{Float64}, Float64},
    rho::Number, # prox regularization parameter
    eta::Number #  this multiplies onto input x. 
)::Nothing
    λ = rho*(this.lambda)
    y_out .= @. max(abs(eta*y) - λ, 0)*sign(eta*y)
    return nothing
end

"""
The Fenchel dual of λ‖⋅‖_1 would be indicator of {x: ‖x‖_∞ ≤ λ}. 
Hence the prox is projecting onto hyper box: [-λ, λ]^n
"""
function dprox!(
    this::OneNormFunction, 
    y_out::FiniteEuclideanSpace,
    y::FiniteEuclideanSpace, 
    rho::Number=1
)::Nothing
    λ = this.lambda
    y_out .= @. min(max(y, -λ), λ)
    return nothing
end

"""
Evaluate the dual of z ↦ λ‖z‖_1, which is the indicator of set λ{x: ‖x‖_∞ ≤ 1}.
The indicator function of rescaled norm ball of the dual norm. 
"""
function dval(
    ::HasFenchelDual, 
    this::OneNormFunction, 
    x::FiniteEuclideanSpace
)::Number
    if norm(x, Inf) > this.lambda
        return Inf64
    end
    return 0.0
end


# ------------------------------------------------------------------------------

"""
It's a class made to compute: 
x ↦ (α/2)‖Ax - b‖^2. 


"""
struct ResidualNormSquared <: ClCnvxFxn
    a::Number
    A::AbstractMatrix
    b::AbstractMatrix

end

function differentiable_trait_assigner(
    ::ResidualNormSquared
)::TraitsOfClCnvxFxn
    return Differentiable()
end


"""
Compute the gradient together with the function value at a point, 
mutate the vector to get the gradient, and then return the numerical values 
of the function evaluated at that point. 
"""
function grad_and_fxnval!(
    ::Differentiable,
    this::ResidualNormSquared, 
    x::FiniteEuclideanSpace,
    x_out::FiniteEuclideanSpace
)::Number
    # TODO: IMPLEMENT THIS ONE HERE. 
    error("I haven't implemented this yet. ")
end