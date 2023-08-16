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