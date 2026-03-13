"""
A super type for all types representing the exist conditions. 
"""
abstract type AbstractExitFlag

end

"""
Nothing strange happened, algorithm terminated because 
stationary condition achieved. 
"""
struct SUCCESSFUL <: AbstractExitFlag end

"""
Inner Loop Line Search Failed. 
"""
struct INNERLOOP_LINESEARCH_FAILED <: AbstractExitFlag end

"""
Inner Loop Max Iteration Reached. 
"""
struct INNERLOOP_MAX_ITERATION_REACHED <: AbstractExitFlag end

"""
Outer Loop Line Search Failed. 
"""
struct OUTERLOOP_LINESEARCH_FAILED <: AbstractExitFlag end

"""
Outer Loop Max Iteration Reached. 
"""
struct OUTERLOOP_MAX_ITERATION_REACHED <: AbstractExitFlag end



"""
Collect the results produced by the IAPGOuterLoopRunner. 

It has the following
"""
mutable struct ResultsCollector
    "Iteration by the inner loop. "
    j::Vector{Int}
    "Error schedule used for each iteration of the inner loop. "
    epsilon::Vector{Float64}
    "‖xk - yk‖ "
    dy::Vector{Float64}
    "Stepsize used, 1/(B + L). "
    ss::Vector{Float64}
    "Current Iterates. "
    x::Vector{Float64}
    "Objective function value"
    fxn_vals::Vector{Float64}
    
    "Intermediate storage for A*x, when computing objective value. "
    Ax::Vector{Float64}
    "Boolean whether to store function objective value. "
    store_fxn_vals::Bool
    "A flag for noting the exiting conditions of the algorithm. "
    flag::AbstractExitFlag
    

    function ResultsCollector(;store_fxn_vals::Bool=false)

        return new(
            Vector{Int}(), 
            Vector{Float64}(), 
            Vector{Float64}(),
            Vector{Float64}(), 
            Vector{Float64}(), 
            Vector{Float64}(), 
            Vector{Float64}(),
            store_fxn_vals, 
            SUCCESSFUL()
        )
    end

end

"""
Store initial guess, and funtion value, depending on the boolean value. 
"""
function initialize!(
    this::ResultsCollector, 
    x::Vector{Float64}, 
    f::ClCnvxFxn, 
    A::AbstractMatrix,
    omega::ClCnvxFxn
)
    # Store Solution
    this.x = similar(x)
    this.x .= x

    # Store Function value
    if this.store_fxn_vals    
        this.Ax = A*x
        push!(this.fxn_vals, f(x) + omega(this.Ax))
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
    ss::Float64, 
    x::Vector{Float64}, 
    f::ClCnvxFxn, 
    A::AbstractMatrix,
    omega::ClCnvxFxn
)::Nothing
    push!(this.j, j)
    push!(this.epsilon, ϵk)
    push!(this.dy, pg)
    push!(this.ss, ss)
    # Store Solution. 
    if length(this.x) == 0
        error(
            "Can you `initialize!` "*
            "`ResultsCollector` before `register!` the iterates? "
        )
    end
    this.x .= x
    # Store objective function value. 
    if this.store_fxn_vals
        mul!(this.Ax, A, x)
        push!(this.fxn_vals, f(x) + omega(this.Ax))
    end
    return nothing
end




""""
The inner loop has several parameters that can be adjusted. 
These parameters wil be referred here for the outer loop to manage. 
Note. If it's set for one outer loop, then it's there fixed for that outer loop. 
"""
struct InnerLoopCommunicator 
    
    itr_max::Int
    backtracking::Bool
    bcktrck_shrinkage::Int

    # More fields? : 
    # 0. The progress meter. 
    # 1. Last Inner Loop Total Iterations Counts. 
    # 2. Outer Loop stoping conditions over tolerance. 
    # 3. Current iteration of the inner loop.

    function InnerLoopCommunicator(
        itr_max::Int=4096, 
        backtracking::Bool=true, 
        bcktrck_shrinkage::Int=2048
    )
        return new(itr_max, backtracking, bcktrck_shrinkage)
    end
end






struct IAPGOuterLoopRunner
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
        rho::Number=1, 
        store_fxn_vals::Bool=false
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
        collector = ResultsCollector(store_fxn_vals=store_fxn_vals)

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
@inline function _ipg!(
    this::IAPGOuterLoopRunner,
    y⁺::Vector{Float64},                        # Will mutate
    y⁺⁺::Vector{Float64},                       # Will mutate
    v::Vector{Float64},                         # Will mutate
    y::Vector{Float64},                         # Will reference
    ∇fy::Vector{Float64},                       # Will reference
    B::Number,                                  # Will reference
    ϵ::Number,                                  # Will reference
    ρ::Number,                                  # Will reference
    inner_loop_settings::InnerLoopCommunicator      # Will reference
)::Number
    ipp = this.ipp
    L = (1 + ρ)*B
    y⁺ .= @. y - (1/L)*∇fy
    j = do_pgd_iteration!(
        ipp, v, y⁺⁺, y⁺,                        # will mutate
        1/L,                                    # ref only
        epsilon=ϵ,
        rho=ρ*B, 
        itr_max=inner_loop_settings.itr_max,    # ref only
        backtracking=inner_loop_settings.backtracking,
        relerr_anchor=y
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
@inline function _ipg_ls!(
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
    ϵk::Number,                     # Will reference
    ls::Bool,                       # Will reference
    lsbtrk::Bool,                   # Will reference
    lsbtrk_shrinkby::Number, 
    inner_loop_settings::InnerLoopCommunicator
)::Tuple{Int, Float64}
    
    # Reference these constant variables: 
    f = this.f; ρ = this.rho;

    # y⁺⁺ <- iprox[g/L](y - (1/L)∇f(y)). 
    j = _ipg!(
        this, y⁺, y⁺⁺, v,   # will mutate. 
        y, ∇fy,
        B, ϵk, ρ, inner_loop_settings
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
                    y, ∇fy, B, ρ, ϵk, 
                    inner_loop_settings
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
            B  /= 2^(1/lsbtrk_shrinkby)
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
    Bk::Number,
    ls::Bool,
    lsbtrk::Bool, 
    lsbtrk_shrinkby::Number, 
    inner_loop_settings::InnerLoopCommunicator
)::Tuple{Int, Float64, Float64, Float64}
    # Reference the constants. 
    f = this.f; ρ = this.rho; E = this.E; p = this.p

    yk⁺ .= @. αk*vk + (1 - αk)*xk
    fy = grad_and_fxnval!(f, ∇fy, yk⁺)
    L0 = (1 + ρ)*B0; Lk = (1 + ρ)*Bk
    ϵk = k >= 1 ? (Lk/L0)*((αk)^2)*E/(k^p) : E
    j, Bk⁺ = _ipg_ls!(
        this, y⁺, y⁺⁺, v, δy,   # Will mutate. 
        ∇fy, yk⁺, fy, Bk, ϵk,
        ls,
        lsbtrk,           # Will reference
        lsbtrk_shrinkby, 
        inner_loop_settings
    )
    xk⁺ .= y⁺⁺
    vk⁺ .= @. xk + (1/αk)*(xk⁺ - xk)
    Lk⁺ = (1 + ρ)Bk⁺
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
    tol::Number;
    max_itr::Int=512, 
    ls::Bool=true,
    lsbtrk::Bool=true, 
    lsbtrk_shrinkby::Number=1024,
    show_progress::Bool=true,
    inner_loop_settings::InnerLoopCommunicator=InnerLoopCommunicator()
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
    
    initialize!(rstlcllctr, xk, f, this.A, this.omega)
    ProgMeter = ProgressThresh(tol; desc="‖x_k - y_k‖:", dt=0.1)
    for k = 0:max_itr
        j, Bk, α, ϵk = _iterate(
            # All of these mutates. 
            this, yk⁺, xk⁺, vk⁺, this.v, ∇fy, y⁺, y⁺⁺, δy,
            xk, vk, k, α, B0, Bk, 
            # These will reference. 
            ls, lsbtrk, lsbtrk_shrinkby, inner_loop_settings
        )
        # STORE. The iterates. 
        vk .= vk⁺
        xk .= xk⁺
        δynorm = norm(δy)
        register!(
            rstlcllctr, j, ϵk, δynorm, 1/(Bk*(1 + ρ)), xk, 
            f, this.A, this.omega
        )
        if show_progress 
            update!(
                ProgMeter, 
                δynorm, 
                showvalues=[("k",k), ("Last Inner Loop Iterated for",j)]
            ) 
        end
        # CHECK. If rrrors occured. 
        if j < 0 || isinf(Bk)
            if j == -1
                rstlcllctr.flag = INNERLOOP_MAX_ITERATION_REACHED()
                break        
            end
            if j == -2 
                rstlcllctr.flag = INNERLOOP_LINESEARCH_FAILED()
                break
            end
            rstlcllctr.flag = OUTERLOOP_LINESEARCH_FAILED()
        end
        if δynorm < tol
            # EXITS. Optimality reached.
            break
        end
        if k >= max_itr 
            # EXITS. Maximum iteration reached. 
            rstlcllctr.flag = OUTERLOOP_MAX_ITERATION_REACHED()
            break 
        end
    end

    return rstlcllctr
end