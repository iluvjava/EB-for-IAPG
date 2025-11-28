# Algorithms for evaluating the proximal operator inexactly, up to a given accuracy. 


"""

This struct is designed to inexactly evaluate proximal problem of the form: 
```
    1/(2λ)‖x - y‖^2 + ω(A*x). 
```

"""
struct InexactProximalPoint
    
    # Fieds that remains constants and shouldn't be changed follow: 
    A::AbstractMatrix
    A_adj::AbstractMatrix
    "omega as a type of `ClCnvxFxn` must have trait `Proxable`"
    omega::ClCnvxFxn
    """
        Inverted stepsize for ISTA evaluation. Best choice is the spectral norm of AᵀA
    """
    t::Number
    
    # Fields that will mutate when running now follows.
    z::AbstractVector
    "Intermediate step for Au, for primal objective computations."
    z1::AbstractVector
    v::AbstractVector
    "Intermediate: Aᵀv"
    v1::AbstractVector
    "Intermediate: AAᵀv"
    v2::AbstractVector
    "Intermediate step for computing prox of ω⋆"
    v3::AbstractVector


    function InexactProximalPoint(
        A::AbstractMatrix, 
        A_adj::AbstractMatrix,
        z::AbstractVector, 
        v::AbstractVector,
        omega::ClCnvxFxn, 
    )
        # parameter assignments.
        t = norm(A, 2)^2
        # memory allocations. 
        z1 = similar(v)  # Az
        v1 = similar(z)  # Aᵀv
        v2 = similar(v)  # AAᵀv
        v3 = similar(v)  # proximal gradient on dual objective
        return new(
            A, A_adj, omega, t, 
            z, z1, v, v1, v2, v3
        )
    end

    function InexactProximalPoint(
        A::AbstractMatrix, 
        omega::ClCnvxFxn, 
    )
        # parameter assignments.
        A_adj = transpose(A)
        # memory allocations. 
        (m, n) = size(A)
        z = zeros(n)
        v = zeros(m)
        return InexactProximalPoint(
            A, A_adj, z, v, omega
        )
    end
end


# TODO: Here is a list of things to do for this struct. 
# - [ ]: a function to evaluate the objective value of the primal, given λ. 
# - [ ]: a function to eval the objetive of the dual, given λ. 
# - [ ]: The duality gap for terminations. 
# - [ ]: Given a point and prox problem regularization parameter λ, accuracy ϵ,
#        a point y, it evaluates the proximal. 

"""
Returns the value of : `ω(Az) + 1/(2λ)‖u - v‖²`. 
"""
function eval_primal_objective_at_current_point(
    this::InexactProximalPoint, 
    y::AbstractVector, 
    lambda::Number
)::Number
    ω = this.omega
    A = this.A
    z = this.z
    λ = lambda
    return ω(A*z) + dot(z - y, z - y)/(2λ)
end


function eval_dual_objective_at_current_point(
    this::InexactProximalPoint, 
    y::AbstractVector,
    lambda::Number
)::Number
    # TODO: IMPLEMENT THIS ONE HERE. 
    v = this.v
    Aᵀv = (this.A_adj)*v
    λ = lambda 
    ω = this.omega
    return (λ/2)*dot(Aᵀv, Aᵀv) - dot(Aᵀv, y) + dval(ω, v)
end




"""
Do one step of primal dual update, using a fixed stepsize which should be valid. 
Update by: 
```
v⁺ = prox[ω⋆](v - (1/τ)(AAᵀv - y))
z⁺ = y - λAᵀv
```

The above is just Chambolle Pock of the proximal problem. 

To elimiate garbage collector time, implementations requires the following 
list of intermediate results to be stored after every 
matrix vector multiplication, so we need memory assigned to the heap for 
variables: AAᵀv, Aᵀv, v, z, v⁺, z⁺. 
And we will mutate them. 


"""
function _update_dual!(
    this::InexactProximalPoint,     # will mutate, specifically, t
    v⁺::AbstractVector,             # will mutate.
    AAᵀv::AbstractVector,           # will Mutate. 
    ∇::AbstractVector,              # will mutate. 
    v::AbstractVector,              # will reference. 
    Ay::AbstractVector,             # will reference. 
    Aᵀv::AbstractVector,            # will ref
    λ::Number,                      # will ref
    τ::Number,                      # will ref
    backtracking::Bool=true
)::Number
    # Referencing. 
    A = this.A
    Aᵀ = this.A_adj
    ω = this.omega
    # Mutate AᵀAv, no need for Aᵀv anymore. 
    mul!(AAᵀv, A, Aᵀv)

    while true
        ∇ .= @. v - (1/τ)*(λ*AAᵀv - Ay)
        # Mutae v⁺
        dprox!(
            ω, 
            v⁺,     # mutates
            ∇, 1/τ  # no mutate. 
        )
        if !backtracking 
            break # we are done here. 
        end
        ∇ .= @. v⁺ - v
        d = (τ/2)*dot(∇, ∇)
        mul!(Aᵀv, Aᵀ, ∇)
        if τ < Inf64 && (λ/2)*dot(Aᵀv, Aᵀv) <= d
            # good! shrink τ to speed up future iteration. 
            τ /= 2^(1/2048)
            break
        else
            # not good, increase τ, and do again. 
            τ *= 2
        end
    end
    return τ
end



"""
Returns the number of iterations used to achieve the assigned accuracies. 
It mutates the given vectors. 

It returns the total number of iterations experienced. 
If the number is -1, it means max iteration reached and the duality gap
tolerance is not satisfied. 
"""
function do_ista_iteration!(
    this::InexactProximalPoint,     # will mutate
    v_out::AbstractVector,          # will mutate
    z_out::AbstractVector,          # will mutate
    y::AbstractVector,              # will reference
    lambda::Number; 
    epsilon::Number=1e-6,
    itr_max::Int=8000, 
    duality_gaps::Union{Vector, Nothing}=nothing, # will mutate
    backtracking::Bool=true
)::Number
    # check dimensions of inputs. 
    @assert size(this.v) == size(v_out)
    @assert size(this.z) == size(z_out)
    @assert epsilon > 0
    @assert lambda > 0

    # Referenced Parameters: 
    λ = lambda
    ϵ = epsilon
    τ = (this.t)*(λ)   # step size
    ω = this.omega
    z = this.z
    v = this.v
    A = this.A
    Aᵀ = this.A_adj
    # Mutating running parameters: 
    Aᵀv = this.v1
    AAᵀv = this.v2
    Az = this.z1
    z⁺ = z_out  
    v⁺ = v_out
    # Ends
    Ay = A*y
    # Starting the forloop, with feasible (z, v) primal dual initial guesses. 
    z .= y
    dprox!(ω, v, Ay)
    mul!(Aᵀv, Aᵀ, v)
    j = 0
    while j < itr_max
        # update duality gap optimality condition, on (z, v)
        mul!(Az, A, z)
        z .= @. z - y 
        p = ω(Az) + dot(z, z)/(2λ)
        z .= @. z + y
        q = (λ/2)*dot(Aᵀv, Aᵀv) - dot(Aᵀv, y) + dval(ω, v)
        if !isnothing(duality_gaps)
            push!(duality_gaps, p + q)
        end
        if p + q <= ϵ
            # (z⁺, v⁺) from previous iteration satisfies duality gap. 
            break
        end
        # perform iteration
        j += 1
        τ = _update_dual!(
            this, 
            v⁺, AAᵀv, this.v3,      # will mutate
            v, Ay, Aᵀv, λ, τ,       # no mutate
            backtracking,
        )
        # update reference (z, v) to (z⁺, v⁺)
        mul!(Aᵀv, Aᵀ, v)
        z⁺ .= @. y - λ*Aᵀv
        z .= z⁺
        v .= v⁺
    end
    return j
end


function do_ista_iteration!(
    this::InexactProximalPoint,
    y::AbstractVector,     # will reference
    lambda::Number;
    epsilon::Number=1e-6,
    itr_max::Int=8000,
    duality_gaps::Union{Vector, Nothing}=nothing,
    backtracking::Bool=true
)::Number 

return do_ista_iteration!(
        this, 
        similar(this.v), 
        similar(this.z), 
        y, 
        lambda, 
        epsilon=epsilon, 
        itr_max=itr_max, 
        duality_gaps=duality_gaps,
        backtracking=backtracking
    )
end