using Plots, SparseArrays, ProgressMeter, Statistics, StatsPlots, 
    DataFrames, Latexify, LaTeXStrings

include("../src/import_inner_loop.jl")
include("../src/outer_loop.jl")
include("function_maker.jl")
include("fast_finite_diff_matrix.jl")


n = 1024

# Setup the problems
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
f = ResidualNormSquared(C, b)
ω = OneNormFunction(10.0)
# A = make_fd_matrix(n, 0)
A = FastFiniteDiffMatrix(n)
rho = 1e-5

# Make the outer loop. 
OuterLoop = IAPGOuterLoopRunner(
    f, ω, A, error_scale=64, rho=rho, store_fxn_vals=true
)

x0 = zeros(n)
tol = 1e-8

@time global Results = run_outerloop_for!(
    OuterLoop, x0, tol, 
    max_itr=1024, lsbtrk=true, show_progress=true,
    inner_loop_settings=InnerLoopCommunicator(65536*16, true, 4096)
)

# ==============================================================================
# VISUALIZING THE SIGNALS 
# ==============================================================================

# Observed VS The denoised signal. 
# - x: The grid. 
# - y: The original signal 
# - Results.x: The reconstructed signal.  
p1 = scatter(
    x, NoisyBlurred_Signal, 
    title="Corrupted VS Recovered Signal", 
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
savefig(p1, "Corrupted VS Recovered Signal N=$n.png")

# Denoised VS Original Signal
p2 = scatter(
    x, y, 
    title="Ground Truth VS Recovered Signal", 
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
# TOTAL INNER LOOP ITERATIVE COMPLEXITY
# ==============================================================================

InnerLoop_ItrJ_Cum = accumulate(+, Results.j[1:end - 1]) # prevent last one is -1.
ks = 1:(length(Results.j) - 1 )
p3 = plot(
    ks, 
    (@. InnerLoop_ItrJ_Cum/ks),
    title="Illustrating if: \$k^{-1}"*
    "\\left(\\sum_{i = 1}^kJ^{(i)}\\right)\\propto \\log_2(k)\$",
    label="Accmulated Inner Loop Iterations over k", 
    xscale=:log2, xlabel="\$k\$: Iteration of the Outerloop", 
    ylabel="\n\$k^{-1}\\left(\\sum_{i = 1}^kJ^{(i)}\\right)\$\n", 
    color=:gray, linewidth=4,
    size=(800, 600), 
    dpi=330
)

p3 |> display
savefig(p3, "Cum Inner Loop Itr Per Outer Loop N=$n.png")

# ==============================================================================
# CONVERGENCE TO STATIONARITY CONDITION RELATIVE TO TOTAL INNER LOOP ITERATIONS
# ==============================================================================
# Log log plot of: 
# 1. X-Axis is the array of the total number of iteration by the inner loop 
#    for every outer loop. 
# 2. Y-Axis is the value of the residual, measured at that outer loop. 
# Overall we expect a O(ε^(-1)\ln(1/ε)) relation between the quantities, 
# We would need a reference plot for that yep. 


Residuals = Results.dy
Js = Results.j
J_Max = accumulate(max, Js)
J_Max_Weighted = J_Max.*(1: (Js |> length))
J_Max_Weighted_Residuals = J_Max_Weighted.*Residuals


p4 = scatter(
    Residuals.^(-1),
    J_Max_Weighted_Residuals,
    xscale=:log2, 
    #yscale=:log2,
    minorgrid=true, minorticks=4,
    xlabel=L"\frac{1}{\left\Vert x_t - y_t \right\Vert}",
    ylabel="\n"*L"(t + 1)\Vert x_t - y_t \Vert\max_{i = 1, \ldots, t}J_i",
    # title="Total Inner Loop Iterations VS Residual \$\\Vert x_k - y_k\\Vert\$", 
    size=(800, 600), dpi=330, 
    markershape=:+, 
    markersize=5
)
plot!(
    p4, 
    Residual_Max.^(-1),
    J_Summed_UpperBound.*Residual_Max,
)
p4|>display