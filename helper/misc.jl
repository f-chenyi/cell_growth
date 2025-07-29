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

function annotate_exp_traj2(Lt::Vector{Float64}, tt::Vector{Float64}; cutoff=0.75)
    # custom annotation 
    isdiv = vcat( (Lt[1:end-1] .- Lt[2:end]) .> cutoff, false);
    # dts
    dts = zeros(length(tt));
    dts[1] = 0.5*(tt[2] - tt[1]);
    dts[end] = 0.5*(tt[end] - tt[end-1]);
    @fastmath@inbounds for i = 2:(length(dts)-1)
        dts[i] = 0.5*(tt[i+1] - tt[i-1]);
    end
    llast = zeros(length(Lt))
    isdiv_idx = findall(isdiv)
    for k in 1:length(isdiv_idx)-1
        llast[isdiv_idx[k]+1:isdiv_idx[k+1]] .= Lt[isdiv_idx[k]]
    end
    llast[isdiv_idx[end]+1:end] .= Lt[isdiv_idx[end]]

    return isdiv, dts, llast
end

function annotate_exp_traj2(Lt::Vector{Vector{Float64}}, tt::Vector{Vector{Float64}}; cutoff=0.75)
    res = annotate_exp_traj2.(Lt, tt; cutoff=cutoff)
    isdiv_ = [res[i][1] for i in eachindex(res)]
    dts_ = [res[i][2] for i in eachindex(res)]
    llast_ = [res[i][3] for i in eachindex(res)]
    return isdiv_, dts_, llast_
end

function annotate_exp_traj3(Lt::Vector{Float64}, tt::Vector{Float64}; cutoff=0.75, fill_missing=:average)
    # custom annotation 
    isdiv = vcat( (Lt[1:end-1] .- Lt[2:end]) .> cutoff, false);
    # dts
    dts = zeros(length(tt));
    dts[1] = 0.5*(tt[2] - tt[1]);
    dts[end] = 0.5*(tt[end] - tt[end-1]);
    @fastmath@inbounds for i = 2:(length(dts)-1)
        dts[i] = 0.5*(tt[i+1] - tt[i-1]);
    end

    l_fill = 0.0
    (fill_missing == :average) && (l_fill = mean(Lt[isdiv]))
    (isa(fill_missing, Number)) && (l_fill = fill_missing)

    llast1 = l_fill * ones(length(Lt))
    llast2 = l_fill * ones(length(Lt))
    isdiv_idx = findall(isdiv)
    for k in 1:length(isdiv_idx)-1
        llast1[isdiv_idx[k]+1:isdiv_idx[k+1]] .= Lt[isdiv_idx[k]]
    end
    llast1[isdiv_idx[end]+1:end] .= Lt[isdiv_idx[end]]

    for k in 1:length(isdiv_idx)-2
        llast2[isdiv_idx[k+1]+1:isdiv_idx[k+2]] .= Lt[isdiv_idx[k]]
    end
    llast2[isdiv_idx[end]+1:end] .= Lt[isdiv_idx[end-1]]

    return isdiv, dts, llast1, llast2
end

function annotate_exp_traj3(Lt::Vector{Vector{Float64}}, tt::Vector{Vector{Float64}}; cutoff=0.75, fill_missing = :average)
    res = annotate_exp_traj3.(Lt, tt; cutoff=cutoff, fill_missing=fill_missing)
    isdiv_ = [res[i][1] for i in eachindex(res)]
    dts_ = [res[i][2] for i in eachindex(res)]
    llast_ = [res[i][3] for i in eachindex(res)]
    llast2_ = [res[i][4] for i in eachindex(res)]
    return isdiv_, dts_, llast_, llast2_
end


function annotate_exp_traj4(Lt::Vector{Float64}, tt::Vector{Float64}; cutoff=0.75, fill_missing=:average)
    # custom annotation 
    isdiv = vcat( (Lt[1:end-1] .- Lt[2:end]) .> cutoff, false);
    (sum(isdiv) == 0) && (return 0)
    (findfirst(isdiv) > 80) && (return 0)
    (sum(isdiv) == 1) && (return 0)   #if there's only 1 division event, then there will be no last2

    # dts
    dts = zeros(length(tt));
    dts[1] = 0.5*(tt[2] - tt[1]);
    dts[end] = 0.5*(tt[end] - tt[end-1]);
    @fastmath@inbounds for i = 2:(length(dts)-1)
        dts[i] = 0.5*(tt[i+1] - tt[i-1]);
    end

    l_fill = 0.0
    (fill_missing == :average) && (l_fill = mean(Lt[isdiv]))
    (isa(fill_missing, Number)) && (l_fill = fill_missing)

    llast1 = l_fill * ones(length(Lt))
    llast2 = l_fill * ones(length(Lt))
    isdiv_idx = findall(isdiv)
    for k in 1:length(isdiv_idx)-1
        llast1[isdiv_idx[k]+1:isdiv_idx[k+1]] .= Lt[isdiv_idx[k]]
    end
    llast1[isdiv_idx[end]+1:end] .= Lt[isdiv_idx[end]]

    for k in 1:length(isdiv_idx)-2
        llast2[isdiv_idx[k+1]+1:isdiv_idx[k+2]] .= Lt[isdiv_idx[k]]
    end
    llast2[isdiv_idx[end]+1:end] .= Lt[isdiv_idx[end-1]]

    isdiv_all = findall(isdiv)
    (isnothing(findlast(diff(isdiv_all).<80))) && (return 0)
    (isdiv_all[findlast(diff(isdiv_all).<80)] < 80) && (return 0)
    return isdiv[1:isdiv_all[findlast(diff(isdiv_all).<80)]], dts[1:isdiv_all[findlast(diff(isdiv_all).<80)]], llast1[1:isdiv_all[findlast(diff(isdiv_all).<80)]], llast2[1:isdiv_all[findlast(diff(isdiv_all).<80)]], Lt[1:isdiv_all[findlast(diff(isdiv_all).<80)]], tt[1:isdiv_all[findlast(diff(isdiv_all).<80)]]
end


function annotate_exp_traj4(Lt::Vector{Vector{Float64}}, tt::Vector{Vector{Float64}}; cutoff=0.75, fill_missing = :average)
    res = annotate_exp_traj4.(Lt, tt; cutoff=cutoff, fill_missing=fill_missing)
    Lt = Lt[.!(res .==0)]
    tt = tt[.!(res .==0)]
    res = res[.!(res .==0)]
    isdiv_ = [res[i][1] for i in eachindex(res)]
    dts_ = [res[i][2] for i in eachindex(res)]
    llast_ = [res[i][3] for i in eachindex(res)]
    llast2_ = [res[i][4] for i in eachindex(res)]
    Lt = [res[i][5] for i in eachindex(res)]
    tt = [res[i][6] for i in eachindex(res)]
    return Lt, tt, isdiv_, dts_, llast_, llast2_
end

function truncation(area, tobs, isdiv, upperbound, lowerbound)
    exclude = findall(area.>upperbound .|| area.<lowerbound);
    (isempty(exclude)) && (return area, tobs)
    if exclude[1] == 1
        return truncation(area[2:end], tobs[2:end], isdiv[2:end], upperbound, lowerbound)
    end
    first = 1;
    # last = exclude[1] - 1;
    last = findlast(isdiv[1:exclude[1] - 1]);
    (isnothing(last)) && (last = exclude[1])
    truncated = Vector{Float64}[];
    truncated_time = Vector{Float64}[];
    truncated = vcat(truncated, [area[first:last]]);
    truncated_time = vcat(truncated_time, [tobs[first:last]]);

    # isgen = exclude[1:end-1] .- exclude[2:end] .+ 1;
    # pos = findall(isgen.<0);
    # (isempty(pos)) && (return truncated, truncated_time)

    # if length(pos)>2
    #     ispos = pos[1:end-1] .- pos[2:end];
    #     pospos = findall(ispos.<-30);
    #     pos = [pos[1],pos[pospos.+1]...];
    # end

    # for i in 1:length(pos)
    #     this = pos[i];
    #     first = findfirst(isdiv_[exclude[this+1]:end]) + exclude[this];
    #     if this == pos[end]
    #         last = findlast(isdiv_[1:exclude[end] - 1]);
    #         truncated = vcat(truncated, [area[first:last]], [area[findfirst(isdiv_[exclude[end]:end]) + exclude[end]:end]]);
    #         truncated_time = vcat(truncated_time, [tobs[first:last]], [tobs[findfirst(isdiv_[exclude[end]:end]) + exclude[end]:end]])
    #         return truncated, truncated_time
    #     end
        
    #     next = pos[i+1];
    #     last = findlast(isdiv_[1:exclude[next]-1]);
    #     truncated = vcat(truncated, [area[first:last]])
    #     truncated_time = vcat(truncated_time, [tobs[first:last]])
    
    # end

    return truncated, truncated_time
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


using Dates, Printf, DelimitedFiles

function save_inference_results(traj::Matrix{<:Any}, div_annotation::AbstractVector{<:Any}, dts_annotation::Vector{<:Any},
        library::Matrix{<:Any}, sbl_history; data_type = "sim_memory0", learn_type = "learn_memory0", 
        output_dir = "./output/", use_uid=true, save_traj=true
)

    uid = ""
    (use_uid) && (uid = Dates.format(now(), "yy-mm-dd-THHMM"))
    
    save_dir = output_dir * data_type * "_" * learn_type * "/" * uid * "/";
    (!isdir(save_dir)) && ( mkpath(save_dir) )
    
    (save_traj) && ( writedlm(save_dir * "traj", convert(Matrix{Float32}, traj) ) )
    (save_traj) && ( writedlm(save_dir * "isdiv", convert(Vector{Int8}, div_annotation) ) )
    (save_traj) && ( writedlm(save_dir * "dts", convert(Vector{Float32}, dts_annotation) ) )
    writedlm(save_dir * "library", library )
    
    writedlm(save_dir * "history_summary", 
        vcat( ["nterms"  "cutoff"  "costfunc"  "error"  "logdetH"] ,
            [sbl_history[:nterms] sbl_history[:cutoff] sbl_history[:costfunc] sbl_history[:error] compute_logdet.(sbl_history[:H])]
        )
    )
    writedlm(save_dir * "history_coeffs", 
        hcat( sbl_history[:coeffs]... )
    )

    nothing

end


function readin_Spombe(datapath)
    data = readdlm(datapath, ' ');
    area_temp = [];
    area_obs = [];
    tobs = [];
    ttemp = [];
    open(datapath, "r") do file
        for line in eachline(file)
            parts = split(line, '\t')
            push!(area_temp, parts[3])
            push!(ttemp, parts[7])
            if parts[8] == "0"
                push!(area_obs, copy(area_temp))
                empty!(area_temp)
                push!(tobs, copy(ttemp))
                empty!(ttemp)
            end
        end
    end

    for i in eachindex(area_obs)
        area_obs[i] = parse.(Float64, area_obs[i][2:end]);
        tobs[i] = parse.(Float64, tobs[i][2:end]);
    end

    area_obs_ = convert(Vector{Vector{Float64}}, area_obs);
    tobs_ = convert(Vector{Vector{Float64}}, tobs);

    area_, t_, isdiv_, dts_, llast_, llast2_ = annotate_exp_traj4(area_obs_, tobs_, cutoff = 100.0);

    areaobs__=[]
    tobs__ = []
    for i in eachindex(area_)
        aa, tt = truncation(area_[i], t_[i], isdiv_[i], 500, 100)
        if isa(aa, Vector{Vector{Float64}})
            push!(areaobs__, aa[1]);
            push!(tobs__, tt[1]);
        else
            push!(areaobs__, aa);
            push!(tobs__, tt);
        end
    end
    areaobs__ = convert(Vector{Vector{Float64}}, areaobs__);
    tobs__ = convert(Vector{Vector{Float64}}, tobs__);

    tobs__ ./= 20.0;

    areaobs__, tobs__, isdiv_, dtobs_, last_, last2_ = annotate_exp_traj4(areaobs__, tobs__, cutoff = 100.0);

    areaobs__ = areaobs__.*0.0935;
    last_ = last_.*0.0935;
    last2_ = last2_.*0.0935;

    return areaobs__, tobs__, isdiv_, dtobs_, last_, last2_
end


function readin_bacillus(datapath)
    data = readdlm(datapath, ' ');
    time = []
    len = []
    mother = []
    id = []
    time_temp = []
    length_temp = []
    mother_temp = []
    id_temp = []
    open(datapath, "r") do file
        for line in eachline(file)
            parts = split(line, '\t')
            push!(time_temp, parts[1])
            push!(length_temp, parts[3])
            push!(mother_temp, parts[7])
            push!(id_temp, parts[6])
            if parts[2] == "1"
                push!(time, parse.(Float64, time_temp))
                push!(len, parse.(Float64, length_temp))
                push!(mother, mother_temp[1])
                push!(id, id_temp[1])
                
                empty!(time_temp)
                empty!(length_temp)
                empty!(mother_temp)
                empty!(id_temp)
            end
        end
    end

    mother = parse.(Float64, mother);
    id = parse.(Float64, id);

    traj = [];
    for dau = 1:size(mother, 1)
        mom_place = findall(id.==mother[dau]);
        if !isempty(mom_place)
            push!(traj, [mom_place[1], dau])
        end
    end

    # iterate a few times
    more_div = true;
    while more_div
        more_div = false;
        for dau = 1:size(traj, 1)
            dau_place = findall(mother.==id[traj[dau][end]]);
            if length(dau_place)>0
                push!(traj[dau], dau_place[1]);
                if length(dau_place)>1
                    push!(traj, copy(traj[dau]));
                    push!(traj[end], dau_place[2]);
                end
                more_div = true;
            end
        end
    end

    traj_len = [];
    traj_len_series = [];
    traj_time = [];
    traj_time_series = [];
    for i = 1:size(traj,1)
        init_time = 0;
        for j in traj[i]
            push!(traj_len_series, len[j])
            push!(traj_time_series, time[j].+init_time)
            init_time = time[j][end] + init_time;
        end
        push!(traj_len, vcat(traj_len_series...))
        push!(traj_time, vcat(traj_time_series...))
        empty!(traj_len_series)
        empty!(traj_time_series)
    end

    traj_time = map(v -> convert(Vector{Float64}, v), traj_time);
    traj_len = map(v -> convert(Vector{Float64}, v), traj_len);

    traj_time = traj_time./60;
    isdiv_, dtobs_, last_ = annotate_exp_traj2(traj_len, traj_time, cutoff = 0.5);

    return traj_len, traj_time, isdiv_, dtobs_, last_
end

function boundary_quadraticfit(traj_len, traj_time, isdiv_, dtobs_, last_, bf1, n_func1, l01)
    Θ1, Σ1 = mgs_gen_bases( [vcat(traj_len...), vcat(last_...)], n_func1, bf1, l01);
    Θdiv1 = Θ1[vcat(isdiv_...),:];
    Θdiv1 = sum(Θdiv1, dims=1)[:];
    history1 = sparse_optimization_sweep(sparse_optimization_fg, Θ1, Θdiv1, vcat(dtobs_...); init_cutoff=0.0, coeffs0=1e-4, EPSILON=1E-7, MAX_ITERS=20);

    _, imax1 = findmax(-history1[:error] .- history1[:nterms].*log.(size(Θ1, 1))/sum(vcat(dtobs_...)));

    F = svd(history1[:H][imax1])
    possigma = zeros(size(history1[:coeffs][imax1]))
    possigma[history1[:coeffs][imax1] .!= 0] = history1[:coeffs][imax1][history1[:coeffs][imax1] .!= 0] + F.U * sqrt.(1 ./ F.S) 
    negsigma = zeros(size(history1[:coeffs][imax1]))
    negsigma[history1[:coeffs][imax1] .!= 0] = history1[:coeffs][imax1][history1[:coeffs][imax1] .!= 0] - F.U * sqrt.(1 ./ F.S) 

    # sdiv_mean = mean(mean.(last_));
    # sdiv_var = var(vcat(last_...));
    # ll = collect(range(sdiv_mean - 2*sdiv_var, sdiv_mean + 2*sdiv_var, 200))
    # ll_ = collect(range(sdiv_mean - 2*sdiv_var, sdiv_mean + 2*sdiv_var, 200))
    sdiv_mean = mean(mean.(traj_len[i][isdiv_[i]] for i in eachindex(isdiv_)) );
    sdiv_std = std(vcat([traj_len[i][isdiv_[i]] for i in eachindex(isdiv_)]...));
    ll = collect(range(sdiv_mean - 3*sdiv_std, sdiv_mean + 3*sdiv_std, 200))
    ll_ = collect(range(sdiv_mean - 3*sdiv_std, sdiv_mean + 3*sdiv_std, 200))

    coeffs_origin = compute_quadratic(mgs_rec_bases([ll, ll_], Σ1, history1[:coeffs][imax1], bf1, l01), ll, ll_, sdiv_mean, sdiv_var, dtobs_, isdiv_)
    coeffs_possigma = compute_quadratic(mgs_rec_bases([ll, ll_], Σ1, vec(possigma), bf1, l01), ll, ll_, sdiv_mean, sdiv_var, dtobs_, isdiv_)
    coeffs_negsigma = compute_quadratic(mgs_rec_bases([ll, ll_], Σ1, vec(negsigma), bf1, l01), ll, ll_, sdiv_mean, sdiv_var, dtobs_, isdiv_)

return coeffs_origin, coeffs_possigma, coeffs_negsigma

end

function compute_quadratic(log_λ, ll, ll_, sdiv_mean, sdiv_var, dtobs_, isdiv_)
    λ_norm = exp.(log_λ').*sum(sum.(dtobs_))/sum(sum.(isdiv_));

    _,j_bin = findmin(abs.(ll .- sdiv_mean));
    _,i_bin = findmin(abs.(ll_ .- sdiv_mean));
    λ_norm_bin = λ_norm[i_bin, j_bin];

    ll_boundary = ll[ [findfirst(λ_norm[i,:].>λ_norm_bin) for i in axes(λ_norm,1)] ];

    around_mean = ll_.>(sdiv_mean-2*sqrt(sdiv_var)) .&& ll_.<(sdiv_mean+2*sqrt(sdiv_var));

    ll_norm_ = ( ll_[around_mean] .- sdiv_mean ) ./ sqrt(sdiv_var);
    ll_norm = ( ll_boundary[around_mean] .- sdiv_mean ) ./ sqrt(sdiv_var);
    plot(ll_norm_, ll_norm)

    return [ll_norm_ ll_norm_.^2] \ ll_norm
end