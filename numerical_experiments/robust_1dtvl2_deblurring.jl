using Plots, SparseArrays, ProgressMeter, Statistics, StatsPlots,
    DataFrames, Latexify, LaTeXStrings, JLD2

include("../src/import_inner_loop.jl")
include("../src/outer_loop.jl")
include("function_maker.jl")
include("fast_finite_diff_matrix.jl")


n = 2048
WORKSPACE_DIR = "saved_workspace_n=$n"
WORKSPACE_FILE = "$WORKSPACE_DIR/workspace_N=$n.jld2"
OVERWRITE_WORKSPACE = false

# Setup the problems
if !OVERWRITE_WORKSPACE && isfile(WORKSPACE_FILE)
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

# prevent last one is -1.
InnerLoop_ItrJ_Cum = accumulate(+, Results.j[1:end - 1]) 
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
# We would need a reference plot, with the formulat `c(ε^(-1)\ln(1/ε))` for some 
# constant c. 


Residuals = Results.dy
Js = Results.j
J_Summed = accumulate(+, Js)

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
    xlabel=L"\sum_{i=1}^k J^{(i)}",
    ylabel=L"\Vert x_k - y_k\Vert",
    title="Residual vs Cumulative Inner Loop Iterations",
    label="Data",
    markershape=:+, markersize=5,
    size=(800, 600), dpi=330
)
plot!(
    p4, J_Summed, (@. max(2, log(J_Summed)^2.5)/(2e-2*J_Summed + 1)), # This model is as close as we can get. 
    label=L"c_1 \cdot \ln(J)/J",
    color=:red, linewidth=2
)
p4 |> display
savefig(p4, "Cum Inner Loop Itr vs Stationarity N=$n.png")



# ==============================================================================
# SAVE WORKSPACE
# ==============================================================================
mkpath(WORKSPACE_DIR)
jldsave(WORKSPACE_FILE;
    n, x, y, Blurred_Signal, NoisyBlurred_Signal, Results
)