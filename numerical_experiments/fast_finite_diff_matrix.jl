import LinearAlgebra: adjoint, transpose, mul!, size, norm
import Base: *, display


abstract type FiniteDifferenceMatrix <: AbstractMatrix{Float64}
end


function size(this::FiniteDifferenceMatrix)::Tuple{Int, Int}
return (this.n, this.n) end

function display(this::FiniteDifferenceMatrix)::Text
    return "$(typeof(this))"
end

function norm(this::FiniteDifferenceMatrix, p::Int)::Float64
    @assert p == 2 "Sorry, we only implemented the Frobenius Norm for "*
    "abstract type FiniteDifferenceMatrix"
    return sqrt(2*this.n)
end


"""
Implements a first order first derivative forward finite difference 
matrix with periodic period directly. 
"""
struct PeriodicFastFiniteDiffMatrix <: FiniteDifferenceMatrix
    n::Int
    function PeriodicFastFiniteDiffMatrix(n::Int) return new(n) end
end



function (*)(
    this::PeriodicFastFiniteDiffMatrix, 
    x::AbstractVector{T}
)::AbstractVector{T} where {T<:Number}
    y = similar(x)
    mul!(y, this, x)
    return y
end


function mul!(
    y::AbstractVector{T}, 
    ::PeriodicFastFiniteDiffMatrix, 
    x::AbstractVector{T}
)::AbstractVector{T} where T <: Number
    n = length(x)
    @simd for i in eachindex(x)
        @inbounds y[i] = x[i] - x[mod(i + 1, n) + 1]
    end
    return y
end


"""
Implements the transpose of the first order, first derivative
forward finite difference matrix with periodic period directly. 
"""
struct PeriodicFastFiniteDiffMatrixTransposed <: FiniteDifferenceMatrix
    n::Int
    function PeriodicFastFiniteDiffMatrixTransposed(n::Int) return new(n) end
end

function adjoint(::PeriodicFastFiniteDiffMatrixTransposed)::AbstractMatrix
    return PeriodicFastFiniteDiffMatrix(this.n)
end

transpose(
    this::PeriodicFastFiniteDiffMatrixTransposed
)::AbstractMatrix = adjoint(this)

function adjoint(this::PeriodicFastFiniteDiffMatrix)::AbstractMatrix
    return PeriodicFastFiniteDiffMatrixTransposed(this.n)
end
transpose(
    this::PeriodicFastFiniteDiffMatrix
)::AbstractMatrix = adjoint(this)

function (*)(
    this::PeriodicFastFiniteDiffMatrixTransposed, 
    x::AbstractVector{T}
)::AbstractVector{T} where {T <: Number}
    y = similar(x)
    mul!(y, this, x)
    return y
end

function mul!(
    y::AbstractVector{T}, 
    ::PeriodicFastFiniteDiffMatrixTransposed, 
    x::AbstractVector{T}
)::AbstractVector{T} where T <: Number
    n = length(x)
    @simd for i in eachindex(x)
        @inbounds y[i] = x[i] - x[mod(i - 2, n) + 1]
    end
    return y
end


