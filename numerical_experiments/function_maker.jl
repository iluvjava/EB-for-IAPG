# Make function for specifictype of numerical experiment problems. 

import SparseArrays:sparse

"""
Finite Differences, First order, forward, or backward differences, 1D. 

## Options: 
1. `bnd_type::Int=1`: Periodic. 
2. `bnd_type::Int=2`: Mirrored. 
3. `bnd_type::Int=0`: No boundary, truncate it. 


"""
function make_fd_matrix(n::Number, bnd_type::Int=1)::AbstractMatrix
    @assert bnd_type == 0 || bnd_type == 1
    c = Vector{Int}()
    r = Vector{Int}()
    v = Vector{Float64}()
    if bnd_type == 1
        for i in 1:n
            # Diagonal ones
            push!(r, i); push!(c, i); push!(v, 1.0)
        end
        # Lower diagonal -1, periodic. 
        for i in 1:n
            push!(r, i); push!(c, mod1(i - 1, n)); push!(v, -1.0)
        end
    # None periodic condition here. 
    else
        for i in 1:n - 1
            # upper diagonal 1
            push!(r, i); push!(c, i + 1); push!(v, 1.0)
        end
        # diagonal -1, 
        for i in 1:n - 1
            push!(r, i); push!(c, i); push!(v, -1.0)
        end
    end
    
    return sparse(r, c, v)
end


"""
1D convolution using a box kernel. Always periodic. 
The box kernel: [1, ..., 1]/((2*width - 1) + 1)


"""
function box_kernel_averaging(
    n::Number, width::Int=3
)::AbstractMatrix
    # Tolerate and handle weird inputs. 
    width = max(2, width)
    l = 2*(width - 1) + 1

    r = Vector{Int}()
    c = Vector{Int}()
    v = Vector{Float64}()

    for i = 0:(n - 1)
        s = max((i - div(l, 2)), 0)
        t = min((i + div(l, 2)), n - 1)
        for j = s:t
            push!(r, i); push!(c, mod(j, n)); push!(v, 1/(t - s + 1))
        end
    end
    r .+= 1; c .+= 1
    return sparse(r, c, v)
end


"""
A matrix that does down sampling. 
Give a binary filter vector, it will make the down sampling matrix. 
Which is just a diagonal I matrix with zeros on the diagonal. 
"""
function ignore_elements_matrix(
    filter::Vector{T}
)::AbstractMatrix where {T <: Number} return spdiagm(filter) end


"""
It's like anti-aliasing
Suppose we have [1, 2, 3, 4]. 
Donw sample by 2 it makes: 
    [(1 + 2)/2, (3 + 4)/2]=[1.5, 3.5]
"""
function down_sample_inhalf(n)

    r = Vector{Int}()
    c = Vector{Int}()
    v = Vector{Float64}()



    return 
end