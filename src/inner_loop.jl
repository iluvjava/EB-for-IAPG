# Algorithms for evaluating the proximal operator inexactly, up to a given accuracy. 


"""

This struct is designed to inexactly evaluate proximal problem of the form: 
```
    1/(2λ)‖x - y‖^2 + ω(A*x). 
```

"""
struct InexactProximalPoint
    
    # Fieds that remains constants and shouldn't be changed follow: 
    A::AbstractMatrix{Float64}
    AT::AbstractMatrix{Float64}
    "omega as a type of `ClCnvxFxn` must have trait `Proxable`"
    omega::ClCnvxFxn
    """
        Inverted stepsize for ISTA evaluation. Best choice is the spectral
        norm of AᵀA
    """
    t::Number
    
    # Fields that will mutate when running now follows.
    z::Vector{Float64}
    v::Vector{Float64}
    "Intermediate: Aᵀv. "
    z1::Vector{Float64}
    "Intermediate: z - y. "
    z2::Vector{Float64}
    "Intermediate: Az. "
    v1::Vector{Float64}
    "Intermediate: AAᵀv. "
    v2::Vector{Float64}
    "Intermediate variable for computing prox of ω⋆"
    v3::Vector{Float64}
    


    function InexactProximalPoint(
        A::AbstractMatrix{Float64}, 
        A_adj::AbstractMatrix{Float64},
        z::Vector{Float64}, 
        v::Vector{Float64},
        omega::ClCnvxFxn, 
    )
        # parameter assignments.
        t = norm(A, 2)^2
        # memory allocations. 
        v1 = similar(v)  # Az
        z1 = similar(z)  # Aᵀv
        v2 = similar(v)  # AAᵀv
        v3 = similar(v)  # Proximal gradient on ω⋆
        z2 = similar(z)  # z - y
        return new(
            A, A_adj, omega, t, 
            z, v, z1, z2, v1, v2, v3
        )
    end

    function InexactProximalPoint(
        A::AbstractMatrix{Float64}, 
        omega::ClCnvxFxn, 
    )
        # parameter assignments.
        A_adj = transpose(A)
        # memory allocations. 
        (m, n) = size(A)
        z = zeros(n)               # Initial guess z. 
        v = zeros(m)
        dprox!(omega, v, zeros(m)) # Initial guess v. 
        return InexactProximalPoint(
            A, A_adj, z, v, omega
        )
    end
end



"""
Returns the value of : `ω(Az) + 1/(2λ)‖u - v‖²`. 
"""
function eval_primal_objective_at_current_point(
    this::InexactProximalPoint, 
    y::Vector{Float64}, 
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
    y::Vector{Float64},
    lambda::Number
)::Number
    v = this.v
    Aᵀv = (this.AT)*v
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
    this::InexactProximalPoint,     # will mutate. 
    v⁺::Vector{Float64},            # will mutate.
    AAᵀv::Vector{Float64},          # will Mutate. 
    ∇::Vector{Float64},             # will mutate. 
    v::Vector{Float64},             # will reference. 
    Ay::Vector{Float64},            # will reference. 
    Aᵀv::Vector{Float64},           # will Mutate
    λ::Number,                      # will ref
    τ::Number,                      # will ref
    backtracking::Bool, 
    s::Int
)::Tuple
    # Referencing. 
    A = this.A
    Aᵀ = this.AT
    ω = this.omega
    # Mutate AᵀAv, no need for Aᵀv anymore. 
    mul!(AAᵀv, A, Aᵀv)

    while true
        ∇ .= @. v - (1/τ)*(λ*AAᵀv - Ay)
        # v⁺ <- prox[ω](v - (1/τ)*(λ*AAᵀv - Ay))
        dprox!(
            ω,      # mutate
            v⁺,     # mutates
            ∇, 1/τ  # ref. 
        )
        if !backtracking 
            break   # we are done here. 
        end
        ∇ .= @. v⁺ - v 
        # Aᵀv <- Aᵀ(v⁺ - v) 
        mul!(Aᵀv, Aᵀ, ∇) 
        if τ < Inf64 && λ*dot(Aᵀv, Aᵀv) <= τ*dot(∇, ∇)
            τ /= 2^(1/s)            # Shrink τ. Backtracking.  
            break
        else
            τ *= 2                                  # Increase τ, Line Search. 
            s *= 2                  # double shrinkage rate. 
        end
    end
    return τ, s
end



"""
Performs one the entire ISTA iteration. 
It returns the total number of iterations experienced, if the number is negative, 
then it's edge cases

1. `-1`, it means max iteration reached and the duality gap tolerance 
is not satisfied. 
2. `-2`, it means backtracking line search in the dual problem failed. 
"""
@inline function do_pgd_iteration!( 
    this::InexactProximalPoint,      # will mutate
    v_out::Vector{Float64},          # will mutate
    z_out::Vector{Float64},          # will mutate
    y::Vector{Float64},              # will reference
    lambda::Number;
    epsilon::Number=1e-6,
    rho::Number=0, 
    itr_max::Int=8000, 
    duality_gaps::Union{Vector, Nothing}=nothing, # will mutate
    backtracking::Bool=true,
    bcktrck_shrinkage::Int=2048,
    relerr_anchor::Vector{Float64}=y
)::Number
    # check dimensions of inputs. 
    @assert size(this.v) == size(v_out)
    @assert size(this.z) == size(z_out)
    @assert epsilon >= 0 "Expect ϵ >= 0, but got ϵ=$epsilon"
    @assert lambda > 0 "Expect λ > 0, but we had λ=$lambda. Catastrophic error."

    # Referenced Parameters: 
    λ = lambda
    ϵ = epsilon
    ρ = rho
    τ = (this.t)*(λ)   # step size
    s = bcktrck_shrinkage
    ω = this.omega
    z = this.z
    v = this.v
    A = this.A
    Aᵀ = this.AT
    Ay = A*y
    # Mutating running parameters: 
    Az = this.v1
    Aᵀv = this.z1
    AAᵀv = this.v2
    zy = this.z2
    z⁺ = z_out
    v⁺ = v_out    
    # Initial guess z update. 
    z .= y
    # Initial guess of v update. 
    mul!(Aᵀv, Aᵀ, v)
    j = 0
    while true
        # compute primal dual objective p, q
        mul!(Az, A, z)
        zy .= @. z - y 
        p = ω(Az) + dot(zy, zy)/(2λ)  
        q = (λ/2)*dot(Aᵀv, Aᵀv) - dot(Aᵀv, y) + dval(ω, v)
        if !isnothing(duality_gaps)  
            push!(duality_gaps, p + q)
        end
        zy .= @. z - relerr_anchor
        if p + q <= ϵ + (ρ/2)*dot(zy, zy) && j >= 1
            # EXITS. (z⁺, v⁺) duality gap reached. 
            break
        end
        j += 1; if j > itr_max  
            # EXITS. Max Iteration. 
            j = -1
            break
        end
        τ, s = _update_dual!(
            this, 
            v⁺, AAᵀv, this.v3,      # will mutate
            v, Ay,                  # Will reference
            Aᵀv,                    # Will mutate 
            λ, τ,
            backtracking,
            s
        )
        if isinf(τ)  
            # EXITS. Line/Backtracking failed. 
            j = -2
            break
        end
        # UPDATES. All Iterates. 
        v  .= v⁺
        mul!(Aᵀv, Aᵀ, v)
        z .= @. y - λ*Aᵀv
        
    end
    z⁺  .= z
    return j
end


function do_pgd_iteration!(
    this::InexactProximalPoint,
    y::Vector{Float64},     # will reference
    lambda::Number;
    epsilon::Number=1e-6,
    rho::Number=0, 
    itr_max::Int=8000,
    duality_gaps::Union{Vector, Nothing}=nothing,
    backtracking::Bool=true, 
    relerr_anchor::Vector{Float64}=y
)::Number 


return do_pgd_iteration!(
        this, 
        similar(this.v),
        similar(this.z),
        y,
        lambda, 
        epsilon=epsilon, 
        rho=rho,
        itr_max=itr_max, 
        duality_gaps=duality_gaps,
        backtracking=backtracking,
        relerr_anchor=relerr_anchor
    )
end