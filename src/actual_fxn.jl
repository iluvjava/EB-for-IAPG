
### ============================================================================
### FUNCTION TYPE
### ============================================================================

using LinearAlgebra

"""
# FUNCTION: λ‖⋅‖_1 


"""
struct OneNormFunction <:ClCnvxFxn
    eta::Float64

    function OneNormFunction(lambda::Number)
        if lambda < 0 
            error("lambda constant for OneNormFunction can be negative. ")
        end
        new(convert(Float64, lambda))
    end
end

function (this::OneNormFunction)(x::AbstractVecOrMat)
    return (this.eta)*norm(x, 1)
end

function (this::OneNormFunction)(x::Vector{Float64})
    return (this.eta)*norm(x, 1)
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
    λ = rho*(this.eta)
    y_out .= @. max(abs(eta*y) - λ, 0)*sign(eta*y)
    return nothing
end

"""
The Fenchel dual of η‖⋅‖_1 would be indicator of {x: ‖x‖_∞ ≤ η}. 
Hence the prox is projecting onto hyper box: [-η, η]^n, it has nothing 
to do with the prox regularization parameter. 
"""
function dprox!(
    ::Proxable,
    this::OneNormFunction, 
    y_out::FiniteEuclideanSpace,
    y::FiniteEuclideanSpace, 
    rho::Number=1 
)::Nothing
    λ = this.eta
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
    # Loosen the criteria numerical computations issues. 
    # Adds some slack and make use of Catastrophic cancellation errors from prox. 
    if norm(x, Inf) <= (1 + eps(Float32))*this.eta
        return 0.0
    end
    return Inf
end


# ------------------------------------------------------------------------------

"""
# FUNCTION: (α/2)‖Ax - b‖_F^2

It's a class made to compute: 
f = x ↦ (α/2)‖Ax - b‖_F^2. 
∇f = x ↦ αAᵀ(Ax - b)

- `b` is a matrix or a vector. Cannot be a number. 
- `A` is a matrix. 
- `x` is a matrix or a vector, depends on what `A, b` are. 


"""
struct ResidualNormSquared <: ClCnvxFxn
    # parameters for the function. 
    alpha::Number
    A::AbstractMatrix{Float64}
    AT::AbstractMatrix{Float64}
    b::Vector{Float64}
    # For computing
    "p is the same shape as x "
    p::Vector{Float64}
    "q is the same shape as Ax "
    q::Vector{Float64}

    function ResidualNormSquared(
        A::AbstractMatrix, 
        b::Vector, 
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
    # Return the trait type: Differentiable for the differentiable
    # interface. 
    return Differentiable()
end

function (this::ResidualNormSquared)(x::Vector{Float64})::Number
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
    p = this.p
    
    mul!(q, A, x)       # q <- Ax
    q .-= B             # q <- Ax - b
    mul!(p, Aᵀ, q)      # p <- Aᵀ(Ax - b)
    x_out .= @. α*p     # x_out <- α*Aᵀ(Ax - b)

    # Return function value. 
    return (α/2)*dot(q, q)
end

function glipz(
    ::Differentiable, 
    this::ResidualNormSquared
)
    α = this.alpha
    A = this.A
    A⁺ = this.AT
    return α*norm(A⁺*A)
end


# ------------------------------------------------------------------------------


"""
It's a class made to model the function: 
X ↦ (α/2)‖A*X*C - B‖_F^2

#LATER: This struct is not yet finished. 

Here, A, C are both matrices. 
This function can represent more advanced image bluring tasks. 
But it's still obliged to operate on vector instead of, just matrix. 

"""
struct MatrixResidualNormSquared <: ClCnvxFxn
    # parameters for the function. 
    alpha::Number
    A::AbstractMatrix{Float64}
    AT::AbstractMatrix{Float64}
    B::AbstractMatrix{Float64}
    # For computing
    "p is the same shape as x "
    P::AbstractMatrix{Float64}
    "q is the same shape as Ax "
    Q::AbstractMatrix{Float64}

end


# ------------------------------------------------------------------------------

"""
x |-> (1/2)dist(Ax - b | [-r, r]^n)^2
#PLANNED: Implemented this type, and then perform TV L2 experiment 
using this objective function instead. 

"""
struct CubeDistanceSquaredAffine <: ClCnvxFxn
    A::AbstractMatrix{Float64}
    b::AbstractVector{Float64}
    r::Number

    AT::AbstractMatrix{Float64}
    "Allocated memory for primal space.  "
    p::AbstractVector{Float64}

    function CubeDistanceSquaredAffine(
        A::AbstractMatrix{Float64}, 
        b::AbstractVector{Float64}, 
        r::Number
    )
        @assert r >= 0 "r must be greater than 0 but we have r=$r."
        p = zeros(size(A, 1)) # to store Ax
        return new(A, b, r, transpose(A), p)
    end
end



"""
Assign differentiable trait to ResidualNormSquared Type. 
"""
function differentiable_trait_assigner(
    ::CubeDistanceSquaredAffine
)::TraitsOfClCnvxFxn
    return Differentiable()
end

"""
Compute the gradient and the function value for type CubeDistanceSquaredAffine. 
The function value is by: f(x) = (1/2)dist(Ax - b | [-λ, λ]^n)^2. 
There are ways to compute the function value and its gradient. 
The calculation shows: 
1. f(x) = (1/2)‖ prox(λ‖⋅‖_1 @ Ax - b) ‖^2
2. ∇f(x) = Aᵀ prox(λ‖⋅‖_1 @ Ax - b)

Where prox(λ‖⋅‖_1 @ x) = max.(|x| - λ, 0)*sign.(x). 
Here are the intermediate values we would need to allocate memory: 
1. Ax - b, then prox(λ‖⋅‖_1 @ Ax - b)
2. The out put of Aᵀx, which is of a different dimension. 

"""
function grad_and_fxnval!(
    ::Differentiable,
    this::CubeDistanceSquaredAffine, 
    x::FiniteEuclideanSpace,
    x_out::FiniteEuclideanSpace
)::Number
    p = this.p; r = this.r; b = this.b
    A = this.A; Aᵀ = this.AT
    mul!(p, A, x)
    p .= @. max(abs(p - b) - r, 0)*sign(p - b)
    # Function value ready. 
    fxnval = dot(p, p)/2
    # Assign the gradient. 
    mul!(x_out, Aᵀ, p)
    return fxnval
end

function (this::CubeDistanceSquaredAffine)(x::AbstractVector{Float64})::Number
    p = this.p; r = this.r; b = this.b
    A = this.A
    mul!(p, A, x)
    p .= @. max(abs(p - b) - r, 0)*sign(p - b)
    return dot(p, p)/2
end

"""
The Lipschitz smoothness constant for function: (1/2)dist(Ax - b | [-λ, λ]^n)^2. 
Recall that the gradient is given by: 
    Aᵀ prox(λ‖⋅‖_1 @ Ax - b)
The Lipschitz smoothness is the same as the Lipschitz continuity modulo for the 
gradient, which is 1 for prox(λ‖⋅‖_1 @ x). 
Therefore, the Lipschitz smoothness constant would be ‖A‖_2^2. 
It is the spectral norm squared, and its upper bound is Frobenius norm of A
squared. 

"""
function glipz(
    ::Differentiable, 
    this::CubeDistanceSquaredAffine
)::Number
    return norm(this.A)^2
end



struct BallDistanceSquaredMatrix <: ClCnvxFxn
    r::Int
    A::AbstractMatrix{Float64}
    b::AbstractVector{Float64}
end

"""
Assign differentiable trait to ResidualNormSquared Type
"""
function differentiable_trait_assigner(
    ::BallDistanceSquaredMatrix
)::TraitsOfClCnvxFxn
    # Return the trait type: Differentiable for the differentiable
    # interface. 

    return Differentiable()
end