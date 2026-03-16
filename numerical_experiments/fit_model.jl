"""
Make a reference line for data. The reference line formula is:
y = c * max(1, log(max(gamma, x))^alpha) / max(gamma, x)^beta
- Alto
— Claude Sonnet 4.6
"""
mutable struct StrangeTwoPhaseLogLogModel
    best_c::Number
    best_alpha::Number
    best_beta::Number
    """Phase transition threshold"""
    gamma::Number

    alpha_bounds::Tuple{Number, Number}
    beta_bounds::Tuple{Number, Number}
    n_points::Number
    data_range::Tuple{Number, Number}

    function StrangeTwoPhaseLogLogModel(
        alpha_bounds::Tuple{Number, Number}=(1.0, 8.0),
        beta_bounds::Tuple{Number, Number}=(0.8, 4.0),
        n_points::Int=300
    )
        alpha_bounds[1] >= alpha_bounds[2] && error(
            "alpha_bounds must satisfy lower < upper, got $alpha_bounds")
        beta_bounds[1] >= beta_bounds[2] && error(
            "beta_bounds must satisfy lower < upper, got $beta_bounds")
        (alpha_bounds[1] <= 0 || beta_bounds[1] <= 0) && error(
            "bounds must be positive; got alpha_bounds=$alpha_bounds, beta_bounds=$beta_bounds")
        n_points < 2 && error("n_points must be ≥ 2, got $n_points")
        return new(
            NaN, NaN, NaN, NaN, alpha_bounds, beta_bounds, n_points, (NaN, NaN)
        )
    end

    

end

"""
Fit the model and find the best parameters.
- Alto
— Claude Sonnet 4.6
- Alto: Fixed a small bug regarding step() function
"""
function fit_model!(
    this::StrangeTwoPhaseLogLogModel, 
    j_values::Vector{<:Real},
    residuals::Vector{<:Real},
    n_alpha_points::Int=20, 
    n_beta_points::Int=20
)::Nothing

    jΣ = Float64.(accumulate(+, j_values))
    # gamma: J_Summed at the first index where Residuals <= 1. 
    idx_transition = findfirst(r -> r <= 1.0, residuals)
    gamma = this.gamma = 
        isnothing(idx_transition) ? jΣ[1] : jΣ[idx_transition]
    model_shape(x, alpha, beta) = 
        max(1.0, log(max(gamma, x))^alpha)/max(gamma, x)^beta
    
    # THE LOSS FUNCTION
    # Take a look at the log log scale. 
    # ln(y) = ln(c) + ln(max(1, ln(x_masked)^alpha)) - beta*ln(x_masked)
    #       = ln(c) + max(0, alpha*lnln(x_masked)) - beta*ln(x_masked)
    # Therefore: 
    # ln(y) - ln(c) = alpha*max(0, lnln(x_masked)) - beta*ln(x_masked)
    # This is a linear regression problem of two variables: alpha, beta, 
    # and the bias is `ln(c)`. 
    function fit_c_mse(alpha, beta)
        mask = jΣ .> gamma
        !any(mask) && return (NaN, Inf)
        lnmasked = log.(model_shape.(jΣ[mask], alpha, beta))
        lny      = log.(residuals[mask])
        ln_c     = mean(lny .- lnmasked)
        mse      = mean((lny .- ln_c .- lnmasked).^2)
        return exp(ln_c), mse
    end


    # coarse grid search
    loss = Inf
    alphas = LinRange(
        this.alpha_bounds[1], 
        this.alpha_bounds[2], 
        n_alpha_points
    )
    betas = LinRange(
        this.beta_bounds[1], 
        this.beta_bounds[2], 
        n_beta_points
    )
    α⁺  = first(alphas)
    β⁺  = first(betas)
    c⁺  = 1.0

    for α in alphas, β in betas
        c, mse = fit_c_mse(α, β)
        isnan(mse) && continue
        if mse < loss
            loss   = mse
            α⁺     = α
            β⁺     = β
            c⁺     = c
        end
    end

    # fine grid search around coarse best
    fine_alphas = (α⁺ - step(alphas)) : step(alphas)/n_alpha_points : (α⁺ + step(alphas))
    fine_betas  = (β⁺ - step(betas)) : step(betas)/n_beta_points : (β⁺  + step(betas))

    for alpha in fine_alphas, beta in fine_betas
        c, mse = fit_c_mse(alpha, beta)
        isnan(mse) && continue
        if mse < loss
            loss   = mse
            α⁺     = alpha
            β⁺     = beta
            c⁺     = c
        end
    end
    this.best_alpha = α⁺
    this.best_beta  = β⁺
    this.best_c     = c⁺
    this.data_range = (minimum(jΣ[jΣ .> 0]), maximum(jΣ))
    return nothing
end



"""
Predict on a log-uniform grid spanning the fitted data range.
Returns (x_grid, y_ref).
— Claude Sonnet 4.6
"""
function ref_line(this::StrangeTwoPhaseLogLogModel)::Tuple{Vector, Vector}
    model_shape(x, alpha, beta) =
        max(1.0, log(max(this.gamma, x))^alpha)/max(this.gamma, x)^beta
    x_grid = exp.(LinRange(log(this.data_range[1]),
        log(this.data_range[2]),
        Int(this.n_points)
    ))
    y_ref  = @. this.best_c * model_shape(x_grid, this.best_alpha, this.best_beta)
    return x_grid, y_ref
end