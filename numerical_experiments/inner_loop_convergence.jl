# Demonstrate the linear convergence of the inner loop. 
# Plot out how the duality gap converges, given a specific error. 

using Plots, SparseArrays, ProgressMeter, Statistics, StatsPlots, 
    DataFrames, Latexify, LaTeXStrings, Random
include("../src/import_inner_loop.jl")

function setup_parameters(
    ;m=1024,
    n=2048, 
    λ=1, 
    η=2
)
    A = sprand(m, n, 1/sqrt(m*n)) + I
    ω = OneNormFunction(η)

    return λ, A, ω
end


"""
Run once, random y, bounded norm. 
Same initial guess for the dual problem. 
"""
function run_once(
    z_out, v_out, y, A, ω, λ, ϵ; itr_max=2^20
)::Vector{Float64}
    iprox = InexactProximalPoint(A, ω)
    gaps = Vector{Float64}()
    do_pgd_iteration!(
        iprox, v_out, z_out, y, λ, epsilon=ϵ, itr_max=itr_max, 
        backtracking=true, 
        duality_gaps=gaps
    )
    return gaps
end

function visualize_trajectories(
    trajectories::Vector{Vector}
)::Nothing
    
    p = plot(
        title="Algorithm Convergence Trajectories",
        xlabel="Iteration",
        ylabel="Convergence Metric (log₂ scale)",
        yscale=:log2,
        legend=false,
        size=(800, 400),
        margin=5Plots.mm
    )
    
    # Plot each trajectory with transparency
    for (_, trajectory) in enumerate(trajectories)
        iterations = 1:length(trajectory)
        trajectory = map((x)-> max(x, eps(Float64)), trajectory)
        # trajectory ./= maximum(trajectory)
        plot!(
            p,
            iterations,
            trajectory,
            label="",
            alpha=0.3,
            linewidth=2,
            color=:gray
        )
    end
    
    display(p)
    return nothing

end

m = 128
n = 128
η = 2
λ, A, ω = setup_parameters(n=n, m=m, η=η)
Trajectories = Vector{Vector}()
repetitions = 30
v_out = zeros(m)
z_out = zeros(n)

@showprogress for _ in 1:repetitions    
    y = η*sign.(randn(n))
    # y .*= 10*η*rand(n)
    ϵ = 2^(-32)
    gaps = run_once(z_out, v_out, y, A, ω, λ, ϵ)
    push!(Trajectories, gaps)
end


visualize_trajectories(Trajectories)

