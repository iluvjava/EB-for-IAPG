"""
Fit reference line to convergence data (Js, Residuals) using the model:

    y = c * max(1, log(max(gamma, x))^alpha) / max(gamma, x)^beta

where x = cumsum(Js). gamma is fixed as the first cumulative iteration
count at which Residuals drops at or below 1 (the transition from the
transient regime into the asymptotic convergence regime). Grid search
is then over (alpha, beta) only; for each pair the optimal c is found
analytically by minimising log-log MSE over the asymptotic region x > gamma.

Returns (x_grid, y_ref): a log-uniform grid spanning the full range of
cumsum(Js) and the model predictions on that grid.

Keyword arguments:
  alphas — range of alpha values for coarse search
  betas  — range of beta values for coarse search
  n_pts  — number of points in the output x_grid

— Claude Sonnet 4.6
"""
function fit_ref_line(
    Js::Vector, Residuals::Vector;
    alphas = 1.0:0.5:8.0,
    betas  = 0.8:0.2:4.0,
    n_pts  = 300
)
    J_Summed = Float64.(accumulate(+, Js))

    # gamma: J_Summed at the first index where Residuals <= 1. 
    idx_transition = findfirst(r -> r <= 1.0, Residuals)
    gamma = isnothing(idx_transition) ? J_Summed[1] : J_Summed[idx_transition]

    model_shape(x, alpha, beta) =
        max(1.0, log(max(gamma, x))^alpha)/max(gamma, x)^beta
    
    # - Alto: 
    # Take a look at the log log scale. 
    # ln(y) = ln(c) + ln(max(1, ln(x_masked)^alpha)) - beta*ln(x_masked)
    #       = ln(c) + max(0, alpha*lnln(x_masked)) - beta*ln(x_masked)
    # Therefore: 
    # ln(y) - ln(c) = alpha*max(0, lnln(x_masked)) - beta*ln(x_masked)
    # This is a linear regression problem of two variables: alpha, beta, 
    # and the bias is `ln(c)`. 
    function fit_c_mse(alpha, beta)
        mask = J_Summed .> gamma
        !any(mask) && return (NaN, Inf)
        lnmasked = log.(model_shape.(J_Summed[mask], alpha, beta))
        lny      = log.(Residuals[mask])
        ln_c     = mean(lny .- lnmasked)
        mse      = mean((lny .- ln_c .- lnmasked).^2)
        return exp(ln_c), mse
    end

    # coarse grid search
    best_mse   = Inf
    best_alpha = first(alphas)
    best_beta  = first(betas)
    best_c     = 1.0

    for alpha in alphas, beta in betas
        c, mse = fit_c_mse(alpha, beta)
        isnan(mse) && continue
        if mse < best_mse
            best_mse   = mse
            best_alpha = alpha
            best_beta  = beta
            best_c     = c
        end
    end

    # fine grid search around coarse best
    fine_alphas = (best_alpha - step(alphas)) : step(alphas)/5 : (best_alpha + step(alphas))
    fine_betas  = (best_beta  - step(betas))  : step(betas)/5  : (best_beta  + step(betas))

    for alpha in fine_alphas, beta in fine_betas
        c, mse = fit_c_mse(alpha, beta)
        isnan(mse) && continue
        if mse < best_mse
            best_mse   = mse
            best_alpha = alpha
            best_beta  = beta
            best_c     = c
        end
    end

    x_grid = exp.(LinRange(log(minimum(J_Summed[J_Summed .> 0])),
                           log(maximum(J_Summed)), n_pts))
    y_ref  = @. best_c * model_shape(x_grid, best_alpha, best_beta)


    return x_grid, y_ref
end
