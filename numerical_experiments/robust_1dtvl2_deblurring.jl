using Plots, SparseArrays, ProgressMeter, Statistics, StatsPlots,
    DataFrames, Latexify, LaTeXStrings, JLD2

include("../src/import_inner_loop.jl")
include("../src/outer_loop.jl")
include("function_maker.jl")
include("fast_finite_diff_matrix.jl")
include("fit_model.jl")


n = 2048
WORKSPACE_DIR = "saved_workspace_n=$n"
WORKSPACE_FILE = "$WORKSPACE_DIR/workspace_N=$n.jld2"
OVERWRITE_WORKSPACE = false

# Setup the problems
if !OVERWRITE_WORKSPACE && isfile(WORKSPACE_FILE)
    # Load back all the variables from previous sessions. 
    @load WORKSPACE_FILE x y Blurred_Signal NoisyBlurred_Signal Results
    println("Loaded workspace from $WORKSPACE_FILE")
else

    let

        global x = LinRange(-2, 2, n)
        global y = @. ((2pi*x) |> sin |> sign)
        global B = box_kernel_averaging(n, div(n, 16))

        # x: Time domain of the signal.
        # y: The true signal.
        # b: Observed signal blurred by A^3 and corrupted.
        # C: box kernel plus subsampling

        # Corrupt the signal
        global Blurred_Signal = B*y
        noise = 3e-1*randn(length(Blurred_Signal))
        global NoisyBlurred_Signal = noise + Blurred_Signal
        global C = B
        global b = NoisyBlurred_Signal

    end

    # Setup the cost functions of the optimizations problem.
    # f = ResidualNormSquared(C, b)
    f = CubeDistanceSquaredAffine(C, b, 0.3)
    ω = OneNormFunction(2.0)
    # A = make_fd_matrix(n, 0)
    A = FastFiniteDiffMatrix(n)
    rho = 1
    p = 2

    # Make the outer loop.
    OuterLoop = IAPGOuterLoopRunner(
        f, ω, A, error_scale=64, p=2, rho=rho, store_fxn_vals=true
    )

    x0 = zeros(n)
    tol = 1e-8

    @time global Results = run_outerloop_for!(
        OuterLoop, x0, tol,
        max_itr=1024, lsbtrk=true, show_progress=true,
        inner_loop_settings=InnerLoopCommunicator(65536*16, true, 4096)
    )

end  # if/else workspace

# ==============================================================================
# VISUALIZING THE SIGNALS 
# ==============================================================================

# Observed VS The denoised signal. 
# - x: The grid. 
# - y: The original signal 
# - Results.x: The reconstructed signal.  
p1 = scatter(
    x, NoisyBlurred_Signal, 
    title="Corrupted VS Recovered", 
    color=:gray, 
    label="Corrupted Signal", 
    marker=:x, 
    markerstrokewidth=3, 
    markersize=5, 
    size=(800, 600),
    dpi=330
)
plot!(
    p1, x, Results.x, 
    color=:blue, alpha=0.5, linewidth=3, label="Recovered"
)
p1|>display
savefig(p1, "Corrupted VS Recovered N=$n.png")

# Denoised VS Original Signal
p2 = scatter( 
    x, y, 
    title="Ground Truth VS Recovered", 
    color=:gray, 
    label="Ground Truth", 
    marker=:x, 
    markerstrokewidth=3, 
    markersize=5, 
    size=(800, 600),
    dpi=330
)
plot!(
    p2, x, Results.x, 
    color=:blue, alpha=0.5, linewidth=3, label="Recovered"
)
p2 |> display
savefig(p2, "Ground Truth VS Recovered Signal N=$n=n.png")

# ==============================================================================
# ABSOLUTE TOLERANCE AND INNER LOOP ITERATIONS
# ==============================================================================
# Plotting out the values of ϵ_k vs J_k. 

AbsTols = Results.abstols
Jk = Results.j
p3 = scatter(
    AbsTols, @.(Jk), xscale=:log2, # yscale=:log2, 
    ylabel=L"J_k", 
    xlabel=L"\epsilon^\circ_k", 
    title=L"J_k "*" vs "*L" \epsilon_k^\circ",
    label="Data",
    legend=:bottomleft,
    markersize=3,
    marker=:x, 
    markerstrokewidth=1.5,
    grid=true, 
    gridstyle=:dash, 
    gridlinewidth=0.5,
    gridalpha=0.5, 
    minorgrid=true,
    minorticks=2,
    minorgridalpha=0.2,
    minorgridstyle=:dot,
    tickfontsize=6, titlefontsize=8, legendfontsize=6, 
    size=(400, 300), dpi=430
)

let
    # The assumed relationship between input x: AbsTols, and output y: 1/Jk
    # is that they have a linear relation under log log plot, i.e.,
    # `ln(1/Jk) = -ln(Jk) = log(a) + b*log(x)`.
    # Implying that the underlying relation between x, y is polynomial:
    # `1/Jk = a*x^b`.
    # Log-log linear regression: ln(1/Jk) = ln(a) + b*ln(x)
    # Robustified: only fit on y values within [10th, 90th] percentile.
    # - Alto: Commanded Claude for implementations.
    # — Claude Sonnet 4.6
    x = Float64.(AbsTols)
    y = Float64.(Jk)
    lo, hi = quantile(y, 0.10), quantile(y, 0.90)
    mask = (y .>= lo) .& (y .<= hi)
    X = hcat(ones(sum(mask)), log.(x[mask]))
    a, b = X \ y[mask]
    x_ref = exp.(LinRange(log(minimum(x)), log(maximum(x)), 300))
    plot!(
        p3, x_ref, (@. a + b*log(x_ref)),
        label=L"a + b\ln(ϵ_k^∘)", 
        color=:red, 
        linewidth=2
    )
    @info "Model Jk = a + b ln(ϵ_k^∘) fitted with a=$a, b=$b"  
end

p3 |> display
savefig(p3, "Epsilonk vs Jk N=$n.png")


# ==============================================================================
# CONVERGENCE TO STATIONARITY CONDITION RELATIVE TO TOTAL INNER LOOP ITERATIONS
# ==============================================================================
# Log log plot of: 
# 1. X-Axis is the array of the total number of iteration by the inner loop
#    for every outer loop. 
# 2. Y-Axis is the value of the residual, measured at that outer loop. 
#   Overall we expect a O(ε^(-1)\ln(1/ε)) relation between the quantities, 
#   We would need a reference plot, with the formulat `c(ε^(-1)\ln(1/ε))` 
#   for some constant c. 


Residuals = Results.dy
J_Summed = accumulate(+, Results.j)

"""
Fit reference line y = c1·ref(x) + c0 to data (x_data, y_data).
ref(u) = ln(u)/u, valid for u > 1 (holds since J_Summed >> 1).
c1 is the tightest upper-bound constant (reference touches data from above).
c0 defaults to 0 (no vertical offset).
Returns (x_range, y_ref) on an n_pts-point log-uniform grid.
— Claude Sonnet 4.6
"""
function ref_line(x_data::Vector, y_data::Vector; c0=0, n_pts=300)
    ref(u) = log(u)/u
    c1 = maximum(@. (y_data - c0)/ref(x_data))
    x_range = exp.(LinRange(log(minimum(x_data)), log(maximum(x_data)), n_pts))
    y_ref = @. c1*ref(x_range) + c0
    return x_range, y_ref
end

x_range, y_ref = ref_line(J_Summed, Residuals)

p4 = scatter(
    J_Summed, Residuals,
    xscale=:log2, yscale=:log2,
    xlabel=L"\sum_{i=0}^k J_{i}",
    ylabel=L"\Vert x_k - y_k\Vert",
    title="Residual vs \$J_k\$ Summed",
    label="Data",
    grid=true, 
    gridstyle=:dash, 
    gridlinewidth=0.5,
    gridalpha=0.5, 
    minorgrid=true,
    minorticks=2,
    minorgridalpha=0.2,
    minorgridstyle=:dot,
    mark=:x, 
    markersize=3,
    tickfontsize=6, titlefontsize=8, legendfontsize=5, 
    size=(400, 300), dpi=430
)


(!@isdefined GuessedModel) && (GuessedModel = StrangeTwoPhaseLogLogModel())
fit_model!(GuessedModel, Results.j, Results.dy)
x_grid, y_ref = ref_line(GuessedModel)
@info "Model fitted"
GuessedModel|>display

plot!(
    p4, x_grid, y_ref,
    label=L"c \cdot \ln(1, \max(c_1, \Sigma_{i = 0}^k J_i))^a / \max(c_1, \Sigma_{i = 0}^k J_i)^b",
    color=:red, linewidth=1, 
    legend=:bottomleft
)
p4 |> display
savefig(p4, "Summed Inner Loop Itr vs Residual N=$n.png")


# ==============================================================================
# OUTERLOOP ITERATION VS INNER LOOP ITERATION
# ==============================================================================
p5 = scatter(
    Results.j, xscale=:log2,
    label="Data",
    xlabel="k",
    ylabel=L"J_k",
    title="\$J_k\$ vs \$k\$",
    legend=:bottomright,
    markersize=3, 
    marker=:x,
    grid=true, 
    gridstyle=:dash, 
    gridlinewidth=0.5, 
    gridalpha=0.5, 
    minorgrid=true, 
    minorticks=2, 
    minorgridalpha=0.2,
    minorgridstyle=:dot,
    tickfontsize=6, titlefontsize=8,
    size=(400, 300), dpi=430
)

# Log-linear regression: Jk = a + b*ln(k)  =>  ref line: exp(a)*k^b
# — Claude Sonnet 4.6
# - Alto
let
    # Robustified: only fit on y values within [10th, 90th] percentile.
    # — Claude Sonnet 4.6
    k = Float64.(1:length(Results.j))
    y = Float64.(Results.j)
    lo, hi = quantile(y, 0.10), quantile(y, 0.90)
    mask = (y .>= lo) .& (y .<= hi)
    X = hcat(ones(sum(mask)), log.(k[mask]))
    a, b = X \ y[mask]
    k_ref = exp.(LinRange(1.0, log(length(Results.j)), 300))
    @info "The ref line parameters of Outer Loop "*
    "Iterations Vs Inner Loop Iterations has: \n a=$a, b=$b."
    plot!(
        p5, k_ref, @.(a + b*log(k_ref)),
        label=L" a + b\ln(k)", color=:red, linewidth=2
    )
end

p5 |> display
savefig(p5, "Jk vs k N=$n.png")

# ==============================================================================
# THE RELATIVE VS ABSOLUTE TOLERANCE
# ==============================================================================
# How does the absolute tolerance, and the relative tolerance by the 
# inner loop looks like compared to each other? 

RelTols = Results.reltols
AbsTols = Results.abstols
TolsSummed = @. RelTols + AbsTols

p6 = plot(
    AbsTols, 
    yscale=:log2, xscale=:log2, 
    label=L"\epsilon_k^∘",
    xlabel="k",
    # ylabel="Inner loop Tolerances",
    title="Inner loop relative and absolute tolerance",
    legend=:bottomleft,
    linestyle=:dot,
    markersize=5, 
    grid=true, 
    gridstyle=:dash, 
    gridlinewidth=0.5, 
    gridalpha=0.5, 
    minorgrid=true, 
    minorticks=2, 
    minorgridalpha=0.2,
    minorgridstyle=:dot,
    tickfontsize=6, titlefontsize=8,labelfontsize=8, 
    size=(400, 300), dpi=430
)
plot!(
    p6, RelTols,
    label=L"\rho_k/\Vert x_k - y_k\Vert"
)
p6|>display
savefig(p6, "abs vs rel tols N=$n.png")



# ==============================================================================
# SAVE WORKSPACE
# ==============================================================================
mkpath(WORKSPACE_DIR)
jldsave(WORKSPACE_FILE;
    n, x, y, Blurred_Signal, NoisyBlurred_Signal, Results
)