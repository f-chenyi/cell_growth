using Optim
using LinearAlgebra

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