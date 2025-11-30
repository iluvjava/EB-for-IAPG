
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
f = x ↦ (α/2)‖Ax - b‖_F^2. 
∇f = x ↦ αAᵀ(Ax - b)


"""
struct ResidualNormSquared <: ClCnvxFxn
    # parameters for the function. 
    a::Number
    A::AbstractMatrix{Float64}
    AT::AbstractMatrix{Float64}
    b::AbstractVecOrMat{Float64}
    # For computing
    "p is the same shape as x "
    p::AbstractVecOrMat{Float64}
    "q is the same shape as Ax "
    q::AbstractVecOrMat{Float64}



    function ResidualNormSquared(
        A::AbstractMatrix{Float64}, 
        b::AbstractVecOrMat{Float64}, 
        alpha::Number=1
    )
        @assert alpha >= 0 "alpha in ResidualNormSquare type must non-negative."
        # Initialize intermediate storage of A.         
        _, n = size(A)
        p = ndims(b) == 1 ? zeros(n) : zeros(n, size(b, 2))
        Aᵀ = transpose(A)
        q = A*p
        return new(alpha, A, Aᵀ, b, p, q)
    end

end

"""
Assign differentiable trait to ResidualNormSquared Type
"""
function differentiable_trait_assigner(
    ::ResidualNormSquared
)::TraitsOfClCnvxFxn
    return Differentiable()
end

function (this::ResidualNormSquared)(x::AbstractVecOrMat)::Number
    A = this.A
    B = this.b
    α = this.alpha
    q = this.q
    mul!(q, A, x)
    q .-= B
    return (α/2)*dot(q, q)
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
    A = this.A
    Aᵀ = this.AT
    B = this.b
    α = this.alpha
    q = this.q
    
    mul!(q, A, x)
    q .-= B 
    mul!(p, Aᵀ, q)
    x_out .= @. α*p
    return (α/2)*dot(q, q)
end


# ------------------------------------------------------------------------------

"""
It's a class made to model the function: 
X ↦ (α/2)‖A*X*C - B‖_F^2

Here, A, C are both matrices. 
This function can represent more advanced image bluring tasks. 

"""
struct MatrixResidualNormSquared <: ClCnvxFxn
    
end