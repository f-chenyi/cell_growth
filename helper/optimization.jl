using Optim
using LinearAlgebra
using Random

# good for small vectors
function mydot(v1::Vector{T}, v2::Vector{T}) where T
    sdot = zero(T)
    for i in 1:length(v1)
        sdot += v1[i] * v2[i]
    end
    return sdot
end


# likelihood for the division events
function _logLikelihood_L2(Theta::Matrix{T}, ThetaDiv::Vector{T}, coeffs::Vector{T}, dts::Vector{T}, gamma::Vector{T}) where T
    
    @assert size(Theta,2) == length(coeffs)
    @assert length(coeffs) == length(ThetaDiv)
    @assert length(dts) == size(Theta,1)
    @assert length(coeffs) == length(gamma)
    
    _logL = -mydot(coeffs, ThetaDiv)
    
    for i in axes(Theta, 1)
        sdiv = zero(T)
        for k in 1:length(coeffs)
            sdiv += coeffs[k] * Theta[i,k]
        end
        _logL += exp(sdiv) * dts[i]
    end 
    
    for k in 1:length(gamma)
        _logL += (coeffs[k]^2 / 2 / gamma[k])
    end
    
    return _logL
    
end


function _fitting_error(Theta::Matrix{T}, ThetaDiv::Vector{T}, coeffs::Vector{T}, dts::Vector{T}) where T
    
    _logL = -mydot(coeffs, ThetaDiv)
    
    for i in axes(Theta, 1)
        sdiv = zero(T)
        for k in 1:length(coeffs)
            sdiv += coeffs[k] * Theta[i,k]
        end
        _logL += exp(sdiv) * dts[i]
    end

    _logL /= sum(dts)

    return _logL

end



function sparse_optimization(Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
                            coeffs0 = ones(size(Theta,2)), gamma0 = ones(size(Theta,2)), MAX_ITERS=100, EPSILON=1E-7, PRINT=true
    ) where T
    
    coeffs_o = copy(coeffs0)
    gamma_o  = copy(gamma0)
    
    count = 0
    cost_func = 0.0
    
    while count < MAX_ITERS
        res = optimize( x-> _logLikelihood_L2(Theta, ThetaDiv, x, dts, gamma_o),
                    coeffs_o, NelderMead() , Optim.Options(show_trace=false,iterations = 10_000); autodiff=:finite
                )
        #update coeffs
        copy!(coeffs_o, res.minimizer)
        # update gamma
        for k in 1:length(coeffs_o)
            gamma_o[k] = coeffs_o[k]^2
        end
        
        count += 1
        (PRINT) && println("itr=$count: $(res.minimum)")
        
        rel_error = abs(res.minimum - cost_func) / abs(cost_func);
        cost_func = res.minimum
        
        (rel_error < EPSILON) && (break)
        
    end
    
    return cost_func, coeffs_o
    
end



function sparse_bayesian_optimization(Phi::Matrix{Float64}, Y::Vector{Float64}; 
            MAX_ITERS=100, EPSILON=1e-6, lambda=1.0, gamma=0.5*ones(size(Phi,2)))

    # fitting Y = Phi * mu + GaussianNoise(lambda)
    
    N, M = size(Phi)
    @assert N == length(Y)
    
    mu = zeros(M)
    mu_old = zeros(M)
    
    PhiT_Phi = Phi'*Phi;
    G_inv = Diagonal(zeros(M));
    Sigma = zeros(M,M)
    Xi = zeros(M, N);
    deltaY = similar(Y)
    
    count = 0
    
    while true
        
        copyto!(mu_old, mu)
        
        G_inv.diag .= 1 ./ gamma
        
        Sigma .= PhiT_Phi ./ lambda .+ G_inv .+ 1e-8
        
        Q = lu!(Sigma);
        Sigma .= Q \ I;
        
#         LinearAlgebra.inv!(cholesky!(Sigma))
#         Sigma .= inv(Sigma)
        
        
        BLAS.gemm!('N','T',1/lambda, Sigma, Phi, false, Xi)
        mul!( mu, Xi, Y)
        
        copyto!(deltaY, Y)
        BLAS.gemm!('N','N',-1.0, Phi, mu, true, deltaY)
        
        for k in eachindex(gamma)
            gamma[k] = mu[k]*mu[k] + Sigma[k,k]
        end
        
        lambda = sum(abs2, deltaY)
        lambda /= N
                
        count += 1
        (count >= MAX_ITERS) && (break)
        (all(abs.(mu_old .- mu) .< EPSILON)) && (break)
        
    end
    
    return  1/2*log(lambda), mu

end



function sparse_optimization_sweep(Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
    annealing_rate::Int=1, init_cutoff=0.01, opt_args...
) where T

    history = Dict(
        :error => Vector{Float64}(undef, 0),
        :nterms => Vector{Int64}(undef,0),
        :cutoff => Vector{Float64}(undef,0),
        :coeffs => Vector{Vector{Float64}}(undef,0),
    )

    cutoff = init_cutoff
    bigind = trues(size(Theta,2))

    count = 0

    while true
        
        coeffs_n  = zeros(size(Theta,2)) 
        coeffs_o = zeros(size(Theta,2))
        count += 1

        println("===== start of epoch $count =====")

        while true 
            
            Theta_tmp = Theta[:, bigind]
            ThetaDiv_tmp = ThetaDiv[bigind]
            _, coeffs_tmp = sparse_optimization(Theta_tmp, ThetaDiv_tmp, dts; opt_args...)

            # cut coefficients
            copy!(@view(coeffs_n[bigind]), coeffs_tmp)
            coeffs_n[abs.(coeffs_n) .<= cutoff] .= 0
            bigind .= ( abs.(coeffs_n) .> cutoff )
            
            if maximum(abs.(coeffs_n .- coeffs_o)) < 1e-6
                break
            else
                copy!(coeffs_o, coeffs_n)
            end 

        end

        nn = sum(bigind);
        err = _fitting_error(Theta, ThetaDiv, coeffs_n, dts)

        # save
        history[:error] = vcat(history[:error], err)
        history[:nterms]= vcat(history[:nterms], nn)
        history[:cutoff]= vcat(history[:cutoff], cutoff)
        push!(history[:coeffs], coeffs_n)

        println("===== end of epoch $count =====")
        println("error = $err, nterms = $nn, cutoff = $cutoff")

        # update cutoff
        if nn == 0
            break;
        else
            (nn < annealing_rate) ? (cutoff = maximum(abs.(coeffs_n[bigind]))+1e-6) : (cutoff=partialsort!(abs.(coeffs_n[bigind]),annealing_rate)+1e-6);
        end

    end

    return history

end

## Lasso regression
# min cost + lam * norm(coeffs,1)
function _logLikelihood_L1(Theta::Matrix{T}, ThetaDiv::Vector{T}, coeffs::Vector{T}, dts::Vector{T}, lam::T) where T
    
    @assert size(Theta,2) == length(coeffs)
    @assert length(coeffs) == length(ThetaDiv)
    @assert length(dts) == size(Theta,1)

    
    _logL = -mydot(coeffs, ThetaDiv)
    
    for i in axes(Theta, 1)
        sdiv = zero(T)
        for k in 1:length(coeffs)
            sdiv += coeffs[k] * Theta[i,k]
        end
        _logL += exp(sdiv) * dts[i]
    end 
    
    _logL += lam * sum(abs.(coeffs))
    
    return _logL
    
end


function sparse_optimization_L1(Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
    coeffs0 = ones(size(Theta,2)), lam = 1.0) where T

    coeffs_o = copy(coeffs0)

    res = optimize( x-> _logLikelihood_L1(Theta, ThetaDiv, x, dts, lam),
    coeffs_o, NelderMead() , Optim.Options(show_trace=false,iterations = 10_000); autodiff=:finite
    )
    copy!(coeffs_o, res.minimizer)

    return res.minimum, coeffs_o


end


# Ridge regression
# min cost + lam * norm(coeffs,2)^2
function _logLikelihood_Ridge_fg!(F, G, Theta, ThetaDiv, coeffs, dts, lam)    

    # common: compute (lambda_t)_i = exp(sum_k Theta_ik w_k) dt_i
    Theta_coeff = Theta * coeffs
    lambda_t = exp.(Theta_coeff) .* dts

    if G != nothing
        BLAS.gemm!('T', 'N', true, Theta, lambda_t, false, G);
        G .-= ThetaDiv
        G .+= coeffs * lam
    end

    if F != nothing

        _logL = -mydot(coeffs, ThetaDiv)

        for i in eachindex(lambda_t)
            _logL += lambda_t[i]
        end

        _logL += lam * sum(coeffs.^2)
        
        return _logL

    end
    
end


function sparse_optimization_Ridge_fg(Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
    coeffs0 = 1.0, lam = 1.0, MAX_ITERS=100, EPSILON=1E-7, PRINT=true,
    opt_method=LBFGS(), opt_option=Optim.Options(show_trace=false,iterations = 10_000,f_reltol=1e-8)
) where T

coeffs_o = coeffs0 * ones(size(Theta,2))

count = 0
cost_func = 0.0

while count < MAX_ITERS

res = optimize( Optim.only_fg!((F,G,x) -> _logLikelihood_Ridge_fg!(F, G, Theta, ThetaDiv, x, dts, lam)),
coeffs_o, opt_method, opt_option
)
#update coeffs
copy!(coeffs_o, res.minimizer)


count += 1
(PRINT) && println("itr=$count: $(res.minimum)")

rel_error = abs(res.minimum - cost_func) / abs(cost_func);
cost_func = res.minimum

(rel_error < EPSILON) && (break)

end

return cost_func, coeffs_o

end

## elastic net regression
# min cost + lam_1 * norm(coeffs,1) + lam_2 * norm(coeffs,2)^2
function _logLikelihood_elasticnet(Theta::Matrix{T}, ThetaDiv::Vector{T}, coeffs::Vector{T}, dts::Vector{T}, lam_1::T, lam_2::T) where T
    
    @assert size(Theta,2) == length(coeffs)
    @assert length(coeffs) == length(ThetaDiv)
    @assert length(dts) == size(Theta,1)

    
    _logL = -mydot(coeffs, ThetaDiv)
    
    for i in axes(Theta, 1)
        sdiv = zero(T)
        for k in 1:length(coeffs)
            sdiv += coeffs[k] * Theta[i,k]
        end
        _logL += exp(sdiv) * dts[i]
    end 
    
    _logL += lam_1 * sum(abs.(coeffs)) + lam_2 * sum(coeffs.^2)
    
    return _logL
    
end


function sparse_optimization_elasticnet(Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
    coeffs0 = ones(size(Theta,2)), lam_1 = 1.0, lam_2 = 1.0) where T

    coeffs_o = copy(coeffs0)

    res = optimize( x-> _logLikelihood_elasticnet(Theta, ThetaDiv, x, dts, lam_1, lam_2),
    coeffs_o, NelderMead() , Optim.Options(show_trace=false,iterations = 10_000); autodiff=:finite
    )
    copy!(coeffs_o, res.minimizer)

    return res.minimum, coeffs_o


end


function _logLikelihood_L2_fg!(F, G, Theta, ThetaDiv, coeffs, dts, gamma)    

    # common: compute (lambda_t)_i = exp(sum_k Theta_ik w_k) dt_i
    Theta_coeff = Theta * coeffs
    lambda_t = exp.(Theta_coeff) .* dts

    if G != nothing
        BLAS.gemm!('T', 'N', true, Theta, lambda_t, false, G);
        G .-= ThetaDiv
        G .+= coeffs ./ gamma
    end

    if F != nothing

        _logL = -mydot(coeffs, ThetaDiv)

        for i in eachindex(lambda_t)
            _logL += lambda_t[i]
        end

        for k in 1:length(gamma)
            _logL += (coeffs[k]^2 / 2 / gamma[k])
        end
        
        return _logL

    end
    
end


function sparse_optimization_fg(Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
                            coeffs0 = 1.0, gamma0 = 1.0, MAX_ITERS=100, EPSILON=1E-7, PRINT=true,
                            opt_method=LBFGS(), opt_option=Optim.Options(show_trace=false,iterations = 10_000,f_reltol=1e-8)
    ) where T
    
    coeffs_o = coeffs0 * ones(size(Theta,2))
    gamma_o  = gamma0 * ones(size(Theta,2))
    
    count = 0
    cost_func = 0.0

    while count < MAX_ITERS

        res = optimize( Optim.only_fg!((F,G,x) -> _logLikelihood_L2_fg!(F, G, Theta, ThetaDiv, x, dts, gamma_o)),
                        coeffs_o, opt_method, opt_option
                )
        #update coeffs
        copy!(coeffs_o, res.minimizer)
        # update gamma

        gamma_o = coeffs_o.^2;
        
        count += 1
        (PRINT) && println("itr=$count: $(res.minimum)")
        
        rel_error = abs(res.minimum - cost_func) / abs(cost_func);
        cost_func = res.minimum
        
        (rel_error < EPSILON) && (break)
        
    end
    
    return cost_func, coeffs_o
    
end

### [23-09-27] This is deprecated and is replaced by the new sparse_optimization_sweep() with Hessian computation.

# function sparse_optimization_sweep(opt_func_handle, Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
#     annealing_rate::Int=1, init_cutoff=0.01, opt_args...
# ) where T

#     history = Dict(
#         :error => Vector{Float64}(undef, 0),
#         :nterms => Vector{Int64}(undef,0),
#         :cutoff => Vector{Float64}(undef,0),
#         :coeffs => Vector{Vector{Float64}}(undef,0),
#     )

#     cutoff = init_cutoff
#     bigind = trues(size(Theta,2))

#     count = 0

#     while true
        
#         coeffs_n  = zeros(size(Theta,2)) 
#         coeffs_o = zeros(size(Theta,2))
#         count += 1

#         println("===== start of epoch $count =====")

#         while true 
            
#             Theta_tmp = Theta[:, bigind]
#             ThetaDiv_tmp = ThetaDiv[bigind]
#             _, coeffs_tmp = opt_func_handle(Theta_tmp, ThetaDiv_tmp, dts; opt_args...)

#             # cut coefficients
#             copy!(@view(coeffs_n[bigind]), coeffs_tmp)
#             coeffs_n[abs.(coeffs_n) .<= cutoff] .= 0
#             bigind .= ( abs.(coeffs_n) .> cutoff )
            
#             if maximum(abs.(coeffs_n .- coeffs_o)) < 1e-6 || sum(bigind) == 0
#                 break
#             else
#                 copy!(coeffs_o, coeffs_n)
#             end 

#         end

#         nn = sum(bigind);
#         err = _fitting_error(Theta, ThetaDiv, coeffs_n, dts)

#         # save
#         history[:error] = vcat(history[:error], err)
#         history[:nterms]= vcat(history[:nterms], nn)
#         history[:cutoff]= vcat(history[:cutoff], cutoff)
#         push!(history[:coeffs], coeffs_n)

#         println("===== end of epoch $count =====")
#         println("error = $err, nterms = $nn, cutoff = $cutoff")

#         # update cutoff
#         if nn == 0
#             break;
#         else
#             (nn < annealing_rate) ? (cutoff = maximum(abs.(coeffs_n[bigind]))+1e-6) : (cutoff=partialsort!(abs.(coeffs_n[bigind]),annealing_rate)+1e-6);
#         end

#     end

#     return history

# end





function sparse_optimization_sweep(opt_func_handle, Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T},
    Theta_test::Matrix{T}, ThetaDiv_test::Vector{T}, dts_test::Vector{T};
    annealing_rate::Int=1, init_cutoff=0.01, opt_args...
) where T

    history = Dict(
        :train_error => Vector{Float64}(undef, 0),
        :test_error => Vector{Float64}(undef, 0),
        :nterms => Vector{Int64}(undef,0),
        :cutoff => Vector{Float64}(undef,0),
        :coeffs => Vector{Vector{Float64}}(undef,0),
    )

    cutoff = init_cutoff
    bigind = trues(size(Theta,2))

    count = 0

    while true
        
        coeffs_n  = zeros(size(Theta,2)) 
        coeffs_o = zeros(size(Theta,2))
        count += 1

        println("===== start of epoch $count =====")

        while true 
            
            Theta_tmp = Theta[:, bigind]
            ThetaDiv_tmp = ThetaDiv[bigind]
            _, coeffs_tmp = opt_func_handle(Theta_tmp, ThetaDiv_tmp, dts; opt_args...)

            # cut coefficients
            copy!(@view(coeffs_n[bigind]), coeffs_tmp)
            coeffs_n[abs.(coeffs_n) .<= cutoff] .= 0
            bigind .= ( abs.(coeffs_n) .> cutoff )
            
            if maximum(abs.(coeffs_n .- coeffs_o)) < 1e-6 || sum(bigind) == 0
                break
            else
                copy!(coeffs_o, coeffs_n)
            end 

        end

        nn = sum(bigind);
        train_err = _fitting_error(Theta, ThetaDiv, coeffs_n, dts)
        test_err =  _fitting_error(Theta_test, ThetaDiv_test, coeffs_n, dts_test)

        # save
        history[:train_error] = vcat(history[:train_error], train_err)
        history[:test_error] = vcat(history[:test_error], test_err)
        history[:nterms]= vcat(history[:nterms], nn)
        history[:cutoff]= vcat(history[:cutoff], cutoff)
        push!(history[:coeffs], coeffs_n)

        println("===== end of epoch $count =====")
        println("train_error = $(train_err), test_error=$(test_err), nterms = $nn, cutoff = $cutoff")

        # update cutoff
        if nn == 0
            break;
        else
            (nn < annealing_rate) ? (cutoff = maximum(abs.(coeffs_n[bigind]))+1e-6) : (cutoff=partialsort!(abs.(coeffs_n[bigind]),annealing_rate)+1e-6);
        end

    end

    return history

end




function sparse_optimization_sweep_bootstrap(opt_func_handle, Theta::Matrix{T}, isdiv::BitVector, dts::Vector{T},
    p_boot=0.5, Nrep = 5; 
    # Theta_test::Matrix{T}, ThetaDiv_test::Vector{T}, dts_test::Vector{T};
    annealing_rate::Int=1, init_cutoff=0.01, opt_args...
) where T

    Nt = size(Theta,1)
    Ns = ceil(Int, p_boot*size(Theta,1) )

    Theta_train = zeros(Ns, size(Theta,2))
    Theta_test = zeros(Nt-Ns, size(Theta,2))
    isdiv_train = falses(Ns)
    isdiv_test = falses(Nt-Ns)
    dts_train = zeros(Ns)
    dts_test = zeros(Nt-Ns)

    history_rep = Vector{Dict}(undef, Nrep)
        
    for rep = 1:Nrep
        
        time_partition = collect( Base.Iterators.partition(randperm(Nt), Ns) )
        time_train  = vcat( time_partition[1] )
        time_test = vcat( time_partition[2:end]... ) 
        copy!( Theta_train, @view(Theta[time_train,:]) )
        copy!( Theta_test, @view(Theta[time_test,:]) )
        copy!( isdiv_train, @view(isdiv[time_train]) )
        copy!( isdiv_test, @view(isdiv[time_test]) )
        ThetaDiv_train = sum(Theta_train[isdiv_train,:], dims=1)[:];
        ThetaDiv_test = sum(Theta_test[isdiv_test,:], dims=1)[:];
        copy!( dts_train, @view(dts[time_train]) )
        copy!( dts_test, @view(dts[time_test]) )
        
        history_rep[rep] = sparse_optimization_sweep(opt_func_handle, Theta_train, ThetaDiv_train, dts_train,
                                                     Theta_test, ThetaDiv_test, dts_test;
                                                     annealing_rate=annealing_rate, init_cutoff=init_cutoff, opt_args...
        )

    end

    return history_rep

end




function sparse_optimization_sweep_bootstrap(opt_func_handle, Theta::Matrix{T}, isdiv::BitVector, dts::Vector{T},
    p_boot=0.5; id::Int=1, save_data = true, save_dir = "./output_sweep_bootstrap/",
    annealing_rate::Int=1, init_cutoff=0.01, opt_args...
) where T

    out_dir = save_dir * "p=$(p_boot)_id=$(id)/"
    (!isdir(out_dir)) && (save_data) && ( mkpath(out_dir) )

    Nt = size(Theta,1)
    Ns = ceil(Int, p_boot*size(Theta,1) )

    Theta_train = zeros(Ns, size(Theta,2))
    Theta_test = zeros(Nt-Ns, size(Theta,2))
    isdiv_train = falses(Ns)
    isdiv_test = falses(Nt-Ns)
    dts_train = zeros(Ns)
    dts_test = zeros(Nt-Ns)

    time_partition = collect( Base.Iterators.partition(randperm(Nt), Ns) )
    time_train  = vcat( time_partition[1] )
    time_test = vcat( time_partition[2:end]... ) 
    copy!( Theta_train, @view(Theta[time_train,:]) )
    copy!( Theta_test, @view(Theta[time_test,:]) )
    copy!( isdiv_train, @view(isdiv[time_train]) )
    copy!( isdiv_test, @view(isdiv[time_test]) )
    ThetaDiv_train = sum(Theta_train[isdiv_train,:], dims=1)[:];
    ThetaDiv_test = sum(Theta_test[isdiv_test,:], dims=1)[:];
    copy!( dts_train, @view(dts[time_train]) )
    copy!( dts_test, @view(dts[time_test]) )
    
    history = sparse_optimization_sweep(opt_func_handle, Theta_train, ThetaDiv_train, dts_train,
                                                    Theta_test, ThetaDiv_test, dts_test;
                                                    annealing_rate=annealing_rate, init_cutoff=init_cutoff, opt_args...
    )

    # save data

    if save_data
        writedlm(out_dir * "fitting_error",[history[:nterms] history[:cutoff] history[:train_error] history[:test_error]])
        writedlm(out_dir * "coeffs", hcat(history[:coeffs]...))
        return nothing
    else
        return history
    end

end




# 23-09-27 addition

function compute_hessian!(H, Theta, ThetaDiv, coeffs, dts, gamma)
    
    Theta_coeff = Theta * coeffs
    lambda_t = exp.(Theta_coeff) .* dts
    Theta_lambda_t = Theta .* lambda_t
    BLAS.gemm!('T', 'N', true, Theta_lambda_t, Theta, false, H);

end


function compute_logdet(H)
    return sum(log.(eigen(H).values))
end


function sparse_optimization_sweep(opt_func_handle, Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
    annealing_rate::Int=1, init_cutoff=0.01, opt_args...
) where T

    history = Dict(
        :error => Vector{Float64}(undef, 0),
        :nterms => Vector{Int64}(undef,0),
        :cutoff => Vector{Float64}(undef,0),
        :coeffs => Vector{Vector{Float64}}(undef,0),
        :costfunc  => Vector{Float64}(undef, 0),
        :H => Vector{Matrix{Float64}}(undef, 0),
    )

    cutoff = init_cutoff
    bigind = trues(size(Theta,2))

    count = 0

    while true
        
        coeffs_n  = zeros(size(Theta,2)) 
        coeffs_o = zeros(size(Theta,2))
        count += 1
        cost_func = 0.0

        println("===== start of epoch $count =====")

        while true 
            
            Theta_tmp = Theta[:, bigind]
            ThetaDiv_tmp = ThetaDiv[bigind]
            cost_func_tmp, coeffs_tmp = opt_func_handle(Theta_tmp, ThetaDiv_tmp, dts; opt_args...)

            # cut coefficients
            cost_func = cost_func_tmp
            copy!(@view(coeffs_n[bigind]), coeffs_tmp)
            coeffs_n[abs.(coeffs_n) .<= cutoff] .= 0
            bigind .= ( abs.(coeffs_n) .> cutoff )
            
            if maximum(abs.(coeffs_n .- coeffs_o)) < 1e-6 || sum(bigind) == 0
                break
            else
                copy!(coeffs_o, coeffs_n)
            end 

        end

        nn = sum(bigind);
        err = _fitting_error(Theta, ThetaDiv, coeffs_n, dts)
        H = zeros(nn, nn)
        compute_hessian!(H, Theta[:,bigind], ThetaDiv[bigind], coeffs_n[bigind], dts, [])
        
        # save
        history[:error] = vcat(history[:error], err)
        history[:costfunc] = vcat(history[:costfunc], cost_func)
        history[:nterms]= vcat(history[:nterms], nn)
        history[:cutoff]= vcat(history[:cutoff], cutoff)
        push!(history[:coeffs], coeffs_n)
        push!(history[:H], H)

        println("===== end of epoch $count =====")
        println("error = $err, nterms = $nn, cutoff = $cutoff")

        # update cutoff
        if nn == 0
            break;
        else
            (nn < annealing_rate) ? (cutoff = maximum(abs.(coeffs_n[bigind]))+1e-6) : (cutoff=partialsort!(abs.(coeffs_n[bigind]),annealing_rate)+1e-6);
        end

    end

    return history

end
