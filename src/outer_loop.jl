
"""
Collect the results produced by the IAPGOuterLoopRunner. 

It has the following
"""
mutable struct ResultsCollector

    j::Vector{Int}
    "Error schedule used for each iteration of the inner loop. "
    epsilon::Vector{Float64}
    "The norm of the gradient mapping. "
    dy::Vector{Float64}
    "Stepsize used, 1/(B + L). "
    ss::Vector{Float64}
    "Current Iterates. "
    x::Vector{Float64}

    function ResultsCollector()

        return new(
            Vector{Int}(), 
            Vector{Float64}(), 
            Vector{Float64}(),
            Vector{Float64}(), 
            Vector{Float64}() 
        )
    end

end



"""
Add the intermediate convergence metric computed in the outer loop 
into the fields of this struct. 
"""
function register!(
    this::ResultsCollector,
    j::Int,
    ϵk::Float64, 
    pg::Float64, 
    ss::Float64
)::Nothing
    push!(this.j, j)
    push!(this.epsilon, ϵk)
    push!(this.dy, pg)
    push!(this.ss, ss)
    return nothing
end


""""
The inner loop has several parameters that can be adjusted. 
These parameters wil be referred here for the outer loop to manage. 
"""
struct InnerLoopSettings 

end



struct IAPGOuterLoopRunner
    # Important Objects
    f::ClCnvxFxn                # differentiable. 
    omega::ClCnvxFxn            # Proxable. 
    A::AbstractMatrix           
    ipp::InexactProximalPoint   # inexact prox operator. 
    collector::ResultsCollector # a collector for collecting results. 

    # Constant parameters. 
    p::Number                   # error shrinkage power. 
    E::Number                   # initial error constant 
    rho::Number                 # A simple constant for relative error. 

    # Primary iterates that mutate. 
    yk::Vector{Float64}
    xk::Vector{Float64}
    vk::Vector{Float64}
    y_next::Vector{Float64}
    x_next::Vector{Float64}
    v_next::Vector{Float64}
    
    # auxillary intermediate iterates that mutate. 
    "An intermediate vector ∇fy "
    y1::Vector{Float64}
    "An intermediate vector y - 1/(B + ρ)∇fy"
    y2::Vector{Float64}
    "An intermediate: y - z, z is the output from the inner loop. "
    y3::Vector{Float64}

    # Inner loop primal dual iterates. 
    z::Vector{Float64}
    v::Vector{Float64} 

    function IAPGOuterLoopRunner(
        f::ClCnvxFxn, omega::ClCnvxFxn, A::AbstractMatrix;
        p::Number=2,
        error_scale::Number=1, 
        rho::Number=1
    )
        @assert p > 1 
        @assert error_scale > 0
        @assert rho >= 0
        
        # Assign. 
        E = error_scale
        
        # Instantiate
        m, n = size(A)
        yk = zeros(n); xk = similar(yk); vk = similar(yk)
        y_next = similar(yk); x_next = similar(yk); v_next = similar(yk)
        y1 = similar(yk); y2 = similar(yk); y3 = similar(yk)

        v = zeros(m)
        z = similar(yk)
        ipp = InexactProximalPoint(
            A, omega
        )
        collector = ResultsCollector()

        # Return the instance. 
        return new(
            f, omega, A, ipp, collector,
            p, E, rho, 
            xk, yk, vk, y_next, x_next, v_next, y1, y2, y3, 
            z, v
        )
    end


end


### Implementations plans 
### 1. Define all relevant parameters
### 2. Prototype one step of iteration successfully at least. 
### 3. Optimize it while testing it continuously. 

"""
Perform one iteration of inexact proximal gradient method operator and that is. 

### Inputs

These will get mutated: 
- `y⁺`: Temporary of `y - (1/L)*∇f(y)`. 
- `y⁺⁺`: Temporary of `prox[g/(B + ρ)](y⁺)`
- `v`: Temporary mutating vector for the dual iterates of the inexact proximal
proximal gradient method. 



"""
function _ipg!(
    this::IAPGOuterLoopRunner,
    y⁺::Vector{Float64},                        # Will mutate
    y⁺⁺::Vector{Float64},                       # Will mutate
    v::Vector{Float64},                         # Will mutate
    y::Vector{Float64},                         # Will reference
    ∇fy::Vector{Float64},                       # Will reference
    B::Number,                                  # Will reference
    ϵ::Number,                                  # Will reference
    ρ::Number;                                  # Will reference
    inner_loop_itr_max::Number=4096             # Will reference
)::Number
    ipp = this.ipp
    y⁺ .= @. y - (1/(B + ρ))*∇fy
    j = do_pgd_iteration!(
        ipp, v, y⁺⁺, y⁺,                        # will mutate
        1/(B + ρ),                              # ref only
        itr_max=inner_loop_itr_max,             # ref only
        epsilon=ϵ,
        rho=ρ
    )
    @assert !any(isnan, y⁺⁺) "Nans in y⁺⁺ from the inner loop. "
    if j < 0
        # Something failed in the inner loop. 
        return j
    end
    return j
end


"""
Performs a step of proximal gradient and then do line search if asked for it. 


"""
function _ipg_ls!(
    this::IAPGOuterLoopRunner,      
    y⁺::Vector{Float64},            # Will mutate
    y⁺⁺::Vector{Float64},           # Will mutate
    v::Vector{Float64},             # Will mutate
    δy::Vector{Float64},            # Will mutate
    # ---------------------------------------------------
    ∇fy::Vector{Float64},           # Will reference
    y::Vector{Float64},             # Will reference
    fy::Float64,                    # will reference
    B::Number,                      # Will reference
    # ---------------------------------------------------
    ϵk::Number;                     # Will reference
    ls::Bool=false,                 # Will reference
    lsbtrk::Bool=false,              # Will reference
    lsbtrk_shrinkby::Number=1024
)::Tuple{Int, Float64}
    
    # Reference these constant variables: 
    f = this.f; ρ = this.rho;

    # y⁺⁺ <- iprox[g/L](y - (1/L)∇f(y)). 
    j = _ipg!(
        this, y⁺, y⁺⁺, v,   # will mutate. 
        y, ∇fy,
        B, ϵk, ρ
    )
    δy .= @. y⁺⁺ - y

    if j < 0  
        # RETURN. Inner loop failed. 
        return j, B
    end
    
    if ls
        while true
            LineSearchOk = f(y⁺⁺) - fy - dot(∇fy, δy) <= (B/2)*dot(δy, δy)
            if LineSearchOk 
                # EXITS. Line search good. 
                break 
            else
                δy .= @. y⁺⁺ - y
                j⁺ = _ipg!(
                    this, y⁺, y⁺⁺, v, 
                    y, ∇fy, B, ρ, ϵk
                )
                if j⁺ < 0
                    # EXITS. Inner loop failed. 
                    return j, B
                else
                    j += j⁺
                end
                B *= 2
                ϵk *= 2
                if isinf(B) 
                    # EXITS. Outer loop line search failed. 
                    return j, B 
                end
            end
        end
        if lsbtrk 
            B /= 2^(1/lsbtrk_shrinkby)
            ϵk /= 2^(1/lsbtrk_shrinkby)
        end
    end

    return j, B
end


"""
Perform one iteration of the outerloop. 
Keep the field updated too. 
Returns (j, Bk, fy)
1. `j` the number of iteration by the inner loop. 
2. `Bk` the Lipschitz constant line search. 
3. `fy` f(y), scalar value. 

"""
function _iterate(
    this::IAPGOuterLoopRunner,
    yk⁺::Vector{Float64},     # Will mutate 
    xk⁺::Vector{Float64},     # Will mutate 
    vk⁺::Vector{Float64},     # Will mutate 
    v::Vector{Float64},       # Will mutate
    ∇fy::Vector{Float64},     # Will mutate
    y⁺::Vector{Float64},      # Will mutate
    y⁺⁺::Vector{Float64},     # Will mutate
    δy::Vector{Float64},      # Will mutate
    xk::Vector{Float64},      # Will ref
    vk::Vector{Float64},      # Will ref
    k::Number,
    αk::Number,
    B0::Number, 
    Bk::Number;
    ls::Bool=false,
    lsbtrk::Bool=false
)::Tuple{Int, Float64, Float64, Float64}
    # Reference the constants. 
    f = this.f; ρ = this.rho; E = this.E; p = this.p

    yk⁺ .= @. αk*vk + (1 - αk)*xk
    fy = grad_and_fxnval!(f, ∇fy, yk⁺)
    L0 = B0 + ρ; Lk = Bk + ρ
    ϵk = k >= 1 ? (E*Lk/L0)/(k^p) : E
    j, Bk⁺ = _ipg_ls!(
        this, y⁺, y⁺⁺, v, δy,   # Will mutate. 
        ∇fy, yk⁺, fy, Bk, ϵk,
        ls=ls,
        lsbtrk=lsbtrk           # Will reference
    )
    xk⁺ .= y⁺⁺
    vk⁺ .= @. xk + (1/αk)*(xk⁺ - xk)
    Lk⁺ = Bk⁺ + ρ
    # FORMULA HERE INCORRECT. 
    α⁺ = (1/2)*(Lk/Lk⁺)*(-αk^2 + sqrt(αk^4 + (4*αk^2)*(Lk⁺/Lk)))
    return j, Bk⁺, α⁺, ϵk
end


"""
Run outerloop for a given amount of iterations, or until termination condition 
is satisfied. 

"""
function run_outerloop_for!(
    this::IAPGOuterLoopRunner, 
    v0::Vector{Float64},
    delta::Number;
    max_itr::Int=512
)::ResultsCollector
    @assert length(v0) == size(this.A, 2)
    k = 0
    α = 1
    f = this.f
    ρ = this.rho
    Bk = B0 = glipz(f)
    xk = this.xk; vk = this.vk; 
    xk .= v0; vk .= v0
    vk⁺ = this.v_next; yk⁺ = this.y_next; xk⁺ = this.x_next
    ∇fy = this.y1
    y⁺ = this.y2
    y⁺⁺ = this.z
    δy = this.y3
    rstlcllctr = this.collector

    while true
        j, Bk, α, ϵk = _iterate(
            # All of these mutates. 
            this, yk⁺, xk⁺, vk⁺, this.v, ∇fy, y⁺, y⁺⁺, δy,
            xk, vk, k, α, B0, Bk
        )
        vk .= vk⁺
        xk .= xk⁺
        register!(
            rstlcllctr, j, ϵk, norm(δy), 1/(Bk + ρ)
        )

        if norm(δy) < delta
            # EXITS. Optimality reached.
            break
        end
        k += 1; if k > max_itr 
            # EXITS. Maximum iteration reached. 
            break 
        end

    end


    return rstlcllctr
end