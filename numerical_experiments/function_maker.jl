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
    @assert bnd_type == 0 || bnd_type == 2 || bnd_type == 1
    c = Vector{Int}()
    r = Vector{Int}()
    v = Vector{Float64}()
    h = 1
    for i in 0:(n - 1)
        # Diagonal ones
        push!(r, i); push!(c, i); push!(v, 1/h)
        if bnd_type == 0 && i == n - 1
            continue
        end
        push!(r, i)
        if bnd_type == 1
            push!(c, mod(i + 1, n))
        else
            push!(c, i == n - 1 ? n - 2 : i + 1)
        end
        push!(v, -1/h)
    end
    r .+= 1
    c .+= 1
    return sparse(r, c, v)
end


"""
Finite Differences, second order, central difference at center, 
backwards/forward differences at boundaries. 1D
"""
function make_secord_fd_matrix(n::Number)
    c = Vector{Int}()
    r = Vector{Int}()
    v = Vector{Float64}()
    h = 1/n
    bd = fd = [2, -5, 4, -1]/h
    cd = [1, -2, 1]/h
    # first row, forward differences. 


    # last row, backwards differences. 
    r .+= 1
    c .+= 1
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
    k = 1/l

    r = Vector{Int}()
    c = Vector{Int}()
    v = Vector{Float64}()

    for i = 0:(n - 1)
        for j = (i - div(l, 2)):(i + div(l, 2))
            push!(r, i); push!(c, mod(j, n)); push!(v, k)
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