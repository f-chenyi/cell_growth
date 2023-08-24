function annotate_sim_traj(Lt::Vector{Float64}, tt::Vector{Float64})
    dts = zeros(length(tt));
    isdiv = falses(length(tt));
    dts[1] = 0.5*(tt[2] - tt[1]);
    dts[end] = 0.5*(tt[end] - tt[end-1]);
    @fastmath@inbounds for i = 2:(length(dts)-1)
        dts[i] = 0.5*(tt[i+1] - tt[i-1]);
        (tt[i+1] == tt[i]) && (isdiv[i] = true)
    end
    return isdiv, dts 
end


function filter_exp_traj(Lt::Vector{T}, tt::Vector{T}; select_time::Vector{Tuple{T,T}}=[(zero(T), tt[end])]) where T
    Lt_ = Vector{Vector{T}}(undef, length(select_time))
    tt_ = Vector{Vector{T}}(undef, length(select_time))
    for i = 1:length(select_time)
        Lt_[i] = Lt[ (tt .> select_time[i][1]) .& (tt .< select_time[i][2]) ]
        tt_[i] = tt[ (tt .> select_time[i][1]) .& (tt .< select_time[i][2]) ]
    end
    return Lt_, tt_
end


function annotate_exp_traj(Lt::Vector{Vector{Float64}}, tt::Vector{Vector{Float64}}; cutoff=0.75)
    res = annotate_exp_traj.(Lt, tt; cutoff=cutoff)
    isdiv_ = [res[i][1] for i in eachindex(res)]
    dts_ = [res[i][2] for i in eachindex(res)]
    return isdiv_, dts_
end



function annotate_exp_traj(Lt::Vector{Float64}, tt::Vector{Float64}; cutoff=0.75)
    # custom annotation 
    isdiv = vcat( (Lt[1:end-1] .- Lt[2:end]) .> cutoff, false);
    # dts
    dts = zeros(length(tt));
    dts[1] = 0.5*(tt[2] - tt[1]);
    dts[end] = 0.5*(tt[end] - tt[end-1]);
    @fastmath@inbounds for i = 2:(length(dts)-1)
        dts[i] = 0.5*(tt[i+1] - tt[i-1]);
    end
    return isdiv, dts 
end


function compute_ldiv_cov(lobs::Vector{Vector{Float64}}, isdiv::Vector{BitVector}, n::Int)
    x = vcat( [lobs[i][isdiv[i]][1:end-n] for i in eachindex(lobs) ]...)
    y = vcat( [lobs[i][isdiv[i]][1+n:end] for i in eachindex(lobs) ]...)
    return mean(x.*y) - mean(x) * mean(y)
end

function compute_ldiv_cov(lobs::Vector{Float64}, isdiv::BitVector, n::Int)
    x = lobs[isdiv][1:end-n]
    y = lobs[isdiv][1+n:end]
    return mean(x.*y) - mean(x) * mean(y)
end