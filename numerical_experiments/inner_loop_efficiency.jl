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
)::Number
    iprox = InexactProximalPoint(A, ω)
    j = do_pgd_iteration!(
        iprox, v_out, z_out, y, λ, epsilon=ϵ, itr_max=itr_max, 
        backtracking=true
    )
    return j
end


function column_quantiles(
    A::Matrix
)
    mins = minimum(A, dims=1)
    q1 = mapslices(col -> quantile(col, 0.25), A, dims=1)
    mid = median(A, dims=1)
    q3 = mapslices(col -> quantile(col, 0.75), A, dims=1)
    maxs = maximum(A, dims=1)
    return mins[:], q1[:], mid[:], q3[:], maxs[:]
end

function plot_ribbon(
    x::Vector,
    results::Matrix;
    xlabel="X LABEL",
    ylabel="Y LABEL",
    title="TITLE",
    show_range=true,
    show_iqr=true,
    median_style::Symbol=:path,
    plot_size::Tuple{Int, Int}=(800, 600),
    dpi::Int=330
)
    min_vals, Q1_vals, median_vals, Q3_vals, max_vals = 
    column_quantiles(results)
    n_points = size(results, 2)

    @assert n_points == length(x) 
    
    p = plot(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=:topright,  
        # yaxis=:log2, 
        xaxis=:log2, 
        size=plot_size,
        dpi=dpi, 
        
    )    
    plot!(
        p, x, median_vals, 
        label="Median", linewidth=2.5, color=:darkblue, 
        seriestype=median_style, 
        marker=:diamond, 
        markersize=3, 
    )
    if show_iqr # IQR
        plot!(p, x, Q3_vals, fillrange=Q1_vals,
              label="IQR", alpha=0.4, color=:cyan, linewidth=0
        )
    end
    if show_range # MIN, Max Range. 
        plot!(p, x, max_vals, fillrange=min_vals,
              label="Full Range", alpha=0.2, color=:gray, linewidth=0
        )
    end

    return p
end



function results_matrix_to_latex(
    A::Matrix;
    row_names::Union{Vector{String}, Nothing}=nothing,
    col_names::Vector{String}=["Min", "Q1", "Median", "Q3", "Max"],
    digits::Int=5
)
    
    mins, q1, mid, q3, maxs = column_quantiles(A)
    if isnothing(row_names)
        row_names = ["Col$(i)" for i in 1:length(mins)]
    end
    @assert row_names |> length == mins |> length
    stats_matrix = hcat(mins, q1, mid, q3, maxs)
    df = DataFrame(stats_matrix, col_names)
    insertcols!(df, 1, :Column => row_names)
    for col in col_names
        df[!, col] = round.(df[!, col], digits=digits)
    end
    return latexify(df; env=:table, booktabs=true, latex=false)
end


# EFFICIENCIES TEST. 

m = 128
n = 128
η = 2
repetition = 100
exponents = -(LinRange(16, 32, 65)|>collect)
# radius = 100
λ, A, ω = setup_parameters(n=n, m=m, η=η)

results = zeros(repetition, exponents |> length)
Threads.@threads for r = 1:repetition
    v_out = zeros(m)
    z_out = zeros(n)
    for (k, ϵ) in pairs(2.0 .^ exponents)
        y = η*sign.(randn(n))
        y .*= η*rand(n)
        results[r, k] = run_once(z_out, v_out, y, A, ω, λ, ϵ)
    end
end

let # THIS BLOCK IS FOR PLOTTING.  
    global p = plot_ribbon(
        2.0 .^ exponents, 
        results, 
        plot_size=(800, 400), 
        xlabel="\nInner Loop Primal Dual Gap: "*L"ϵ_k"*"\n",
        ylabel="\nTotal Number of Inner loop Iterations",
        title="5-Point Summary of Inner loop iterations varying "
            *L"\epsilon_k^\circ=2^{-k}", 
        median_style=:scatterpath, 
    )
    p|>display


    @info "The latex of the results matrix as a table: "
    latex_table = results_matrix_to_latex(
        results, 
        row_names=["ϵ = 2^($(round(k,digits=4)))" for k in exponents]
    )
    latex_table |> println
    savefig(p, "5p-inner-loop-j$(exponents[1])$(exponents[end]).png")

end


