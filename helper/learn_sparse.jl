function derivative(data, time, isdiv)
# dxdt: continuous
# dydt: jump
    dxdt = (data[2:end] - data[1:end - 1])./(time[2:end] - time[1:end - 1])
    # dydt = dxdt[isdiv[1:end - 1]]    # for experimental data
    dydt = (data[2:end] - data[1:end - 1])   # for simulation data
    deleteat!(dxdt, findall(isdiv->isdiv,isdiv))   # drop where division happens for continuous
    return dxdt, dydt[isdiv[1:end - 1]]
end


function library_generate(data, time, isdiv, n)
    Theta = zeros(length(data), n + 1)
    for i = 1:(n + 1)
        Theta[:, i] = data.^(i - 1)
    end
    return Theta[.!isdiv,:][1:end - 1,:], Theta[isdiv,:]
end


function learn_sparse(Theta, dxdt, threshold; dim = 1)
# This function use least square approximation
# dxdt = Theta * Xi
# Input:
# Theta: each column represents a candidate function
# dxdt: derivative of X, obtained by differentiation
# threshold: threshold to exclude the big coefficients
# n: the dimension of the input data
# 
# Output:
# Xi: The coefficients of library terms

    Xi = Theta\dxdt
    for k = 1:10
        smallinds = abs.(Xi) .< threshold
        Xi[smallinds] .= 0
        for ind = 1:dim
            biginds = findall(!iszero, Xi[:, ind])
            Xi[biginds, ind] = Theta[:, biginds] \ dxdt[:, ind]
        end
    end
    return Xi
end

function least_square_L1_penalty(Theta, y, x, l)
    res = y - Theta * x
    return sum(res.^2) + l * abs.(x)
end

function learn_sparse_penalty(Theta, dxdt, threshold; 
    dim = 1, lambda = 0.1, coeff_0 = ones(size(Theta,2)),
    optim_options=Optim.Options(show_trace=false,iterations = 10_000) 
    )
    # This function has added a penalty term
    # dxdt = Theta * Xi
    # Input:
    # Theta: each column represents a candidate function
    # dxdt: derivative of X, obtained by differentiation
    # threshold: threshold to exclude the big coefficients
    # n: the dimension of the input data
    # lambda: penalized terms
    # 
    # Output:
    # Xi: The coefficients of library terms
    
        res = optimize( x-> least_square_L1_penalty(Theta, dxdt, x, lambda),
                        coeff_0, NelderMead() , optim_options; autodiff=:finite)
        Xi = res.minimizer
        for k = 1:10
            smallinds = abs.(Xi) .< threshold
            Xi[smallinds] .= 0
            for ind = 1:dim
                biginds = findall(!iszero, Xi[:, ind])
                res = optimize( x-> least_square_L1_penalty(Theta[:, biginds], dxdt[:, ind], x, lambda),
                                coeff_0[biginds], NelderMead() , optim_options; autodiff=:finite)
            Xi[biginds, ind] = res.minimizer
            end
        end
        return Xi
    end