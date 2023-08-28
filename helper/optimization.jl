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


function sparse_optimization_multisparsity(Theta::Matrix{T}, ThetaDiv::Vector{T}, dts::Vector{T};
    coeffs0 = ones(size(Theta,2)), gamma0 = ones(size(Theta,2)), MAX_ITERS=100, EPSILON=1E-7, PRINT=true, COEFF_TRESHOLD = 0.01
) where T
coeff_return =[]
num_relevant_return = []
num_relevant = size(Theta,2)

_, coeffs_new = sparse_optimization(Theta, ThetaDiv, dts; EPSILON=1E-7, MAX_ITERS=40);

coeffs = copy(coeffs_new)
smallinds = findall(abs.(coeffs) .< COEFF_TRESHOLD)
biginds = [i for i in 1:length(coeffs) if !(i in smallinds)]
coeffs[abs.(coeffs) .< COEFF_TRESHOLD] .= 0.0
coeff_return = vcat(coeff_return, [coeffs])
mini_new = minimum(abs.(coeffs[biginds]))
num_relevant_return = vcat(num_relevant_return, [length(biginds)])

while num_relevant > length(biginds)
    num_relevant = length(biginds)
    _, coeffs_new[biginds] = sparse_optimization(Theta[:,biginds], ThetaDiv[biginds], dts; EPSILON=1E-7, MAX_ITERS=40);
    
    coeffs = copy(coeffs_new)
    smallinds = vcat(findall((abs.((abs.(coeffs) .- mini_new)) .<COEFF_TRESHOLD)), smallinds)
    biginds = [i for i in 1:length(coeffs) if !(i in smallinds)]
    coeffs[smallinds] .= 0.0
    coeff_return = vcat(coeff_return, [coeffs])
    mini_new = minimum(abs.(coeffs[biginds]))
    num_relevant_return = vcat(num_relevant_return, [length(biginds)])
    # make sure we do not exclude all coeffs
    (length(biginds) == 0) && (break)
end


return coeff_return, num_relevant_return

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