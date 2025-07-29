function plot_histogram!( pp, _hedges::Vector{T}, _y::Vector{T}; plot_args...) where T
    _index_edges = vcat([[i,i] for i in 1:length(_hedges)]...);
    _index_y = vcat([[i,i] for i in 1:length(_y)]...);
    plot!(pp, _hedges[_index_edges[2:end-1]], _y[_index_y]; plot_args...)
    nothing
end

function plot_histogram( _hedges::Vector{T}, _y::Vector{T}; plot_args...) where T
    _index_edges = vcat([[i,i] for i in 1:length(_hedges)]...);
    _index_y = vcat([[i,i] for i in 1:length(_y)]...);
    pp = plot(_hedges[_index_edges[2:end-1]], _y[_index_y]; plot_args...)
    return pp
end

function f(du, u, p, t)
    bf_f = p[1]
    coeffs_f = p[2]
    du[1] = 0.0
    for n = 1:length(coeffs_f)
        du[1] += coeffs_f[n]*bf_f(n-1, u[1])
    end
end

function rate_memoryless(u,p,t) 
    bf_λ = p[5]
    Σ_λ = p[6]
    coeffs_λ = p[7]
    (u[1] > 6.2) && (return 100.0)
    return exp(rec_ortho_bases(u, Σ_λ, coeffs_λ, bf_λ)[1])
end

function rate_memory(u, p, t)
    Σ1 = p[5]
    Σ2 = p[6]
    coeffs = p[7]
    bf1 = p[8]
    bf2 = p[9]
    ln_rate = rec_ortho_bases( [u[1]], Σ1, [u[2]], Σ2, coeffs, bf1, bf2)
    return exp(ln_rate[1])
end

function affect!(integrator)
    integrator.u[2] = integrator.u[1]
    bf_g = integrator.p[3]
    coeffs_g = integrator.p[4]
    u_old = integrator.u[1]
    for n = 1:length(coeffs_g)
        integrator.u[1] -= coeffs_g[n] * bf_g(n-1, u_old)
    end
end

function plot_output(bf_f, coeffs_f::Vector{T}, bf_g, coeffs_g::Vector{T}, bf_λ1, Σ1::Matrix{T}, coeffs_lam::Vector{T},
    Tmax::T, Linput::Vector{T}, tinput::Vector{T}, isdiv_input::BitVector,
    dirname, nterms, memory,
    Σ2=nothing, bf_λ2= nothing) where T

    mkpath(dirname * "_n_" * string(nterms))
    dir = dirname * "_n_" * string(nterms) * "/"

    if memory == 1
        vrj = VariableRateJump(rate_memory, affect!)
        oprob = ODEProblem(f, [Linput[1];2 * Linput[1]], (0.0,Tmax), (bf_f, coeffs_f, bf_g, coeffs_g, Σ1, Σ2, coeffs_lam, bf_λ1, bf_λ2))
    else
        vrj = VariableRateJump(rate_memoryless, affect!)
        oprob = ODEProblem(f, [Linput[1];0], (0.0,Tmax), (bf_f, coeffs_f, bf_g, coeffs_g, bf_λ1,  Σ1, coeffs_lam))
    end

    jprob = JumpProblem(oprob, Direct(), vrj)
    sol = solve(jprob,Tsit5(),dtmax=0.01);
    Lresult = Array(sol')[:,1];
    tresult = sol.t;
    isdiv_result, _ = annotate_sim_traj(Lresult, tresult)

    # save data
    writedlm(dir * "coeffs_f.txt", coeffs_f, '\t')
    writedlm(dir * "coeffs_g.txt", coeffs_g, '\t')
    writedlm(dir * "coeffs_lam.txt", coeffs_lam, '\t')

    # trajectory
    p_traj = plot(; size=(1500,300), xlim=(0,10), ylim=(0,8), xtick=collect(0:3:10), ytick=collect(0:3:9), bottommargin=12Plots.mm,  leftmargin=9Plots.mm, xlabel="Time", ylabel="Cell length")
    plot!(p_traj, tinput, Linput)
    plot!(p_traj, tresult, Lresult, grid=false, legend=false, lw=2, lc=:dodgerblue4, size=(1500,300), tickfontsize=14, linestyle=:dash)
    savefig(dir * "resim_trajectory.pdf")

    # resim_histogram_div_len
    stephist(Linput[isdiv_input], normalize=:pdf,
        xlim=(2,10), xtick=[2,4,6,8, 10],ylim=(-0.02,1),ytick=collect(0:0.5:1), bins=collect(2.0:0.3:10),
        ylabel="PDF", xlabel="Cell length at division")
    stephist!(Lresult[isdiv_result], normalize=:pdf, lc=:dodgerblue4, linestyle=:dash)
    savefig(dir * "resim_histogram_div_len.pdf")

    open(dir * "KS_div_len.txt", "w") do f
        write(f, string(ApproximateTwoSampleKSTest(Linput[isdiv_input], Lresult[isdiv_result])))
    end

    # resim_histogram_gen_time
    stephist!(diff(tinput[isdiv_input]), normalize=:pdf,
        xlim=(0.2,1.4), xtick=collect(0:0.4:1.2),ylim=(-0.02,8),ytick=[0,2,4,6,8], bins=collect(0:0.05:1.2),
        ylabel="PDF", xlabel="Generation time" )
    stephist!(diff(tresult[isdiv_result]), normalize=:pdf, lc=:dodgerblue4, linestyle=:dash)
    savefig(dir * "resim_histogram_gen_time.pdf")

    open(dir * "KS_gen_time.txt", "w") do f
        write(f, string(ApproximateTwoSampleKSTest(diff(tresult[isdiv_result]),diff(tinput[isdiv_input]))))
    end

    # with memory, we plot the lambda
    if memory == 1
        uu = collect(2:0.1:9);
        uu_ = collect(2:0.1:9);
        ln_λ = rec_ortho_bases(uu, Σ1, uu_, Σ2, coeffs_lam);
        heatmap(uu, uu_, exp.(ln_λ), clim=(0,20), framestyle=:box)
        savefig(dir * "inferred_lambda.png")
    end

end