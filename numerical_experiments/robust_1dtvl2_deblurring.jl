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
    global B = box_kernel_averaging(n, 8)

    # x: Time domain of the signal. 
    # y: The true signal. 
    # b: Observed signal blurred by A^3 and corrupted. 
    # C: box kernel plus subsampling
    
    # Corrupt the signal
    global Blurred_Signal = B*y
    global Corrupted_Signal  = 
        [rand() > 1/sqrt(sqrt(n)) ? NaN : x for x in Blurred_Signal] + 
            1e-1*randn(n)
    F = ignore_elements_matrix([isnan(i) ? 0.0 : 1.0 for i in Corrupted_Signal])
    global C = F*B
    global b = [isnan(i) ? 0.0 : i for i in Corrupted_Signal]
    

end

# Setup the cost functions of the optimizations problem. 
f = ResidualNormSquared(C, b)
ω = OneNormFunction(0.5)
A = make_fd_matrix(n)
# A = PeriodicFastFiniteDiffMatrix(n)
rho = 0.5

# Make the outer loop. 
OuterLoop = IAPGOuterLoopRunner(
    f, ω, A, error_scale=64, rho=rho, store_fxn_vals=true
)

x0 = ones(n)

@time global Results = run_outerloop_for!(
    OuterLoop, x0, 1e-5, 
    max_itr=1024, lsbtrk=true, show_progress=true,
    inner_loop_settings=InnerLoopCommunicator(65536*16, true, 4096)
)

# PLOTTING OUT THE SIGNALS =====================================================
# Observed VS The denoised signal. 
# x: The grid. 
# y: The original signal 
# Results.x: The deblurred signal.  
p1 = scatter(
    x, Corrupted_Signal, 
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
# INSIGHTS INTO INNER LOOP ITERATION WRT TO OTHER VARIABLES. 
# ==============================================================================

InnerLoop_ItrJ_Cum = accumulate(+, Results.j[1:end - 1]) # prevent last one is -1.
ks = 1:(length(Results.j) - 1 )
p3 = plot(
    ks, 
    (@. InnerLoop_ItrJ_Cum/ks),
    title="Illustrating if: \$k^{-1}"*
    "\\left(\\sum_{i = 1}^kJ^{(i)}\\right)\\propto \\log_2(k)\$",
    label="Accmulated Inner Loop Iterations over k", 
    xscale=:log2, xlabel="\$\\log_2(k)\$, k: Iteration of the Outerloop", 
    ylabel="\n\$k^{-1}\\left(\\sum_{i = 1}^kJ^{(i)}\\right)\$\n", 
    color=:gray, linewidth=4,
    size=(800, 600), 
    dpi=330
)

p3 |> display
savefig(p3, "Cum Inner Loop Itr Per Outer Loop N=$n.png")

# Total inner loop iterations vs log2(‖xk-yk‖)

Min_Residuals = Results.dy
Total_InnerLoop_IterJ = accumulate(+, Results.j)
p4 = plot(
    Total_InnerLoop_IterJ, 
    Min_Residuals, 
    xscale=:log2, yscale=:log2,
    minorgrid=true, minorticks=4, 
    xlabel="\$\\sum_{i = 1}^k J^{(i)}\$\n",
    ylabel="\n\$\\left\\Vert x_k - y_k\\right\\Vert\$", 
    title="Total Inner Loop Iterations VS Residual \$\\Vert x_k - y_k\\Vert\$", 
    size=(800, 600), dpi=330
)
p4|>display
savefig(p4, "Cum Inner Loop Itr vs Stationarity N=$n.png")

# Relative + Absolute Tolerance 
# VS Inner Loop Iteration per Outer Loop Iterations
# Expect Log Log Relations. 

Epsilons = Results.epsilon
Relative_Errors = @. (Results.dy^2)*rho*Results.ss/2
TotalErrors = @. Epsilons + Relative_Errors

p5 = scatter(
    Epsilons, TotalErrors, 
    xscale=:log2, yscale=:log2, 
    minorgrid=true, minorticks=2, 
    marker=:x, markerstrokewidth=2, markersize=5, 
    size=(800, 600), dpi=330, 
    title="\$\\epsilon_k\$ vs \$J^{(k)}\$", 
    xlabel="\$\\epsilon_k\$", 
    ylabel="\$J^{(k)}\$"
)

p5 |> display
savefig("Epsilonk vs Jk N=$n.png")
