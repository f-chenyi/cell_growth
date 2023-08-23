# import Base: convert

abstract type Basis end



# ================= trial functions in time =====================
# To add custom trial functions to the pipeline:
# (1) define struct NEW_FUNC
# (2) add a method in [eval_func] for NEW_FUNC
# (3) add a method in [eval_dfunc] for NEW_FUNC
# (4) add a method in [build_tfs] for NEW_FUNC (gen trial function libraries)

abstract type TrialFunctions{T,α} <: Basis end

struct PolynomialBumps{T,α} <: TrialFunctions{T,α} end
struct FourierBases{T,α} <: TrialFunctions{T,α} end
struct MexiHat{T,α} <: TrialFunctions{T,α} end
struct PolynomialHaar{T,α} <: TrialFunctions{T,α} end
(basis::TrialFunctions{T,0})(t, params) where {T} = eval_func(basis, t, params...)
(basis::TrialFunctions{T,1})(t, params) where {T} = eval_dfunc(basis, t, params...)

# (basis::PolynomialBumps{T,0})(t, params) where {T} = eval_func(basis, t, params...)
# (basis::PolynomialBumps{T,1})(t, params) where {T} = eval_dfunc(basis, t, params...)
# (basis::FourierBases{T,0})(t, params) where {T} = eval_func(basis, t, params...)
# (basis::FourierBases{T,1})(t, params) where {T} = eval_dfunc(basis, t, params...)
# (basis::MexiHat{T,0})(t, params) where {T} = eval_func(basis, t, params...)
# (basis::MexiHat{T,1})(t, params) where {T} = eval_dfunc(basis, t, params...)
# (basis::PolynomialHaar{T,0})(t, params) where {T} = eval_func(basis, t, params...)
# (basis::PolynomialHaar{T,1})(t, params) where {T} = eval_dfunc(basis, t, params...)


function eval_func(tf::PolynomialBumps{T,0}, t::T, p::Int, t1::T, tk::T) where {T}
    Cnorm = ((tk-t1)/2)^(2p)
    ( t1 <= t <= tk) ? (return (t-t1)^p*(tk-t)^p/Cnorm) : (return 0.0)
end

function eval_dfunc(tf::PolynomialBumps{T,1}, t::T, p::Int, t1::T, tk::T) where {T}
    Cnorm = ((tk-t1)/2)^(2p)
    ( t1 <= t <= tk) ? (return p*(t-t1)^(p-1)*(tk-t)^(p-1)*(-2t+t1+tk)/Cnorm) : (return 0.0)
end


function eval_func(tf::FourierBases{T,0}, t::T, n::Int, P::T, t0::T) where {T}
    return sin( π*(t-t0)/P*n )
end

function eval_dfunc(tf::FourierBases{T,1}, t::T, n::Int, P::T, t0::T) where {T}
    ω = π / P * n
    return cos( ω * (t-t0) ) * ω 
end


function eval_func(tf::MexiHat{T,0}, t::T, ::Int, t0::T, a::T) where {T}
    ts = (t-t0) / a
    Cnorm = 1/a^(1/2)
    ( -5.0 <= ts <= 5.0 ) ? (return Cnorm*(1-2*ts^2)*exp(-ts^2)) : (return 0.0)
end

function eval_dfunc(tf::MexiHat{T,1}, t::T, ::Int, t0::T, a::T) where {T}
    ts = (t-t0) / a
    Cnorm = 1/a^(3/2)
    ( -5.0 <= ts <= 5.0 ) ? (return Cnorm*2*ts*(2*ts^2-3)*exp(-ts^2)) : (return 0.0)
end


function eval_func(tf::PolynomialHaar{T,0}, t::T, n::Int, k::T, P::T) where {T}
    ts = 2^n * t / P - k
    return 2^(n/2)*haar(ts)
end

function eval_dfunc(tf::PolynomialHaar{T,1}, t::T, n::Int, k::T, P::T) where {T}
    ts = 2^n * t / P - k
    return 2^(n/2)*dhaar(ts)*2^n/P
end



function haar(t;p=4)
    Cnorm = (1/4)^(2p)
    if 0<=t<=0.5
        return (t)^p*(0.5-t)^p/Cnorm
    elseif 0.5<=t<=1
        return -(t-0.5)^p*(1.0-t)^p/Cnorm
    else
        return 0.0
    end
end

function dhaar(t;p=4)
    Cnorm = (1/4)^(2p)
    if 0<=t<=0.5
        return p*(t)^(p-1)*(0.5-t)^(p-1)*(-2t+0.5)/Cnorm
    elseif 0.5<=t<=1
        return -p*(t-0.5)^(p-1)*(1.0-t)^(p-1)*(-2t+1.5)/Cnorm
    else
        return 0.0
    end
end

# convert(::Type{TrialFunctions{T, α}},x::PolynomialBumps{T,α}) where {T,α}  = x

# ====================================================================





# =========================== Interaction Basis ==============================

abstract type InteractionBasis <: Basis end
import Base: getindex
getindex(f::F, ::Int64) where F <: InteractionBasis = f # to artificially promote F to Vector{F}

abstract type CentralForceInteraction <: InteractionBasis end
struct InversePolynomial <: CentralForceInteraction end
(basis::InversePolynomial)(n::Int, r) = 1/r^n
(basis::InversePolynomial)(n::Int, r, ::Any) = 1/r^n

struct Polynomial <: InteractionBasis end
(basis::Polynomial)(n::Int, x) = x^n
(basis::Polynomial)(n::Int, x, ::Any) = x^n

struct GeneralizedYukawa <: InteractionBasis end
(basis::GeneralizedYukawa)(n::Int, x::T) where T = (n==0) ? (one(T)) : (exp(-x)/x^n)
# (basis::GeneralizedYukawa)(n::Int, x::T, x0::T) where T = basis(n, x/x0)
(basis::GeneralizedYukawa)(n::Int, x::T, x0::T) where T = (n==0) ? (one(T)) : (exp(-x/x0)/x^n)

struct PolynomialCf <: CentralForceInteraction end
(basis::PolynomialCf)(n::Int, x) = x^n
(basis::PolynomialCf)(n::Int, x, ::Any) = x^n

function recursive_eval_array!(values::Vector{T}, op::InversePolynomial, x::T) where{T}
    
    nmax = length(values) - 1

    # Initialization
    values[1] = 1.0;
    
    # Recursive evaluation
    for i ∈ 1:1:nmax
        values[i+1] = values[i] / x
    end

end

function recursive_eval_array!(values::Vector{T}, op::PolynomialCf, x::T) where{T}
    
    nmax = length(values) - 1

    # Initialization
    values[1] = 1.0;
    
    # Recursive evaluation
    for i ∈ 1:1:nmax
        values[i+1] = values[i] * x
    end

end


recursive_eval_array!(values::Vector{T}, op::InversePolynomial, x::T, ::Any) where T = recursive_eval_array!(values, op, x)
recursive_eval_array!(values::Vector{T}, op::PolynomialCf, x::T, ::Any) where T = recursive_eval_array!(values, op, x)



abstract type OrthogonalFunctionBasis <: Basis end
abstract type OrthogonalPolynomialBasis{T} <: OrthogonalFunctionBasis end
initialvalue0(::OrthogonalPolynomialBasis{T}, x) where {T} = one(T) # This is often the case! 
(basis::OrthogonalPolynomialBasis)(n::Int, x::Float64) = recursive_eval(basis, n, x)
(basis::OrthogonalPolynomialBasis)(n::Int, x::Float64, r0::Float64) = recursive_eval(basis, n, x/r0)
(basis::OrthogonalPolynomialBasis)(indices::Vector{Int}, x::Float64) = recursive_eval_array(basis, indices, x)
(basis::OrthogonalPolynomialBasis)(indices::Vector{Int}, x::Float64, r0::Float64) = recursive_eval_array(basis, indices, x/r0)

recursive_eval_array!(values::Vector{T}, op::OrthogonalPolynomialBasis{T}, x::T, r0::T) where T = recursive_eval_array!(values, op, x/r0)
recursive_eval_array!(values::Vector{T}, op::OrthogonalPolynomialBasis{T}, x::T, params::Array{T}) where T = recursive_eval_array!(values, op, x/params[1])

# adding ! version of recursive_eval_array - more efficient to use in loops
function recursive_eval_array!(values::Vector{T}, op::OrthogonalPolynomialBasis, x::T) where{T}
    
    nmax = length(values) - 1

    # Initialization
    values[1] = initialvalue0(op, x)
    nmax > 0 && (values[2] = initialvalue1(op, x))
    
    # Recursive evaluation
    for i ∈ 1:1:nmax-1
        a1, a2, a3, a4 = recurrence_coefficients(op, i)
        values[i+2] = muladd(x, a3, a2)*values[i+1] - a4*values[i]
        values[i+2] /= a1
    end

end


function recursive_eval_array(op::OrthogonalPolynomialBasis, indices::Vector{Int}, x::T) where{T}
    nmax = maximum(indices)
    values = ones(T, nmax+1) # Values of the polynomials at x from 0 to nmax

    # Initialization
    values[1] = initialvalue0(op, x)
    nmax > 0 && (values[2] = initialvalue1(op, x))
    
    # Recursive evaluation
    for i ∈ 1:1:nmax-1
        a1, a2, a3, a4 = recurrence_coefficients(op, i)
        values[i+2] = muladd(x, a3, a2)*values[i+1] - a4*values[i]
        values[i+2] /= a1
    end

    return values[indices .+ 1]
end


# function recursive_eval(op::OrthogonalPolynomialBasis, n::Int, x::T) where {T}
#     recursive_eval_array(op, [n], x)[1]
# end

# use the legacy recursive_eval
function recursive_eval(op::OrthogonalPolynomialBasis{T}, n::Int, x::T) where {T} 
    # Intialization 
    n <  0 && return one(T)
    n == 0 && return initialvalue0(op, x)
    n == 1 && return initialvalue1(op, x)

    x0, x1 = initialvalue0(op, x), initialvalue1(op, x)

    # Recursive evaluation
    for i = 1:n - 1
        a1, a2, a3, a4 = recurrence_coefficients(op, i) # typo? n=>i
        x1, x0 = muladd(x, a3, a2)*x1 - a4*x0, x1 
        x1 /= a1
    end

    return x1 
end


struct GeneralizedLaguerreFunction{T, α} <: OrthogonalPolynomialBasis{T} end # Scaled by e^(-r/2) not technically a polynomial but statistfies the recurrence
@inline recurrence_coefficients(::GeneralizedLaguerreFunction{T, α}, n) where {T, α} = n + 1, 2n + α + 1, -1, n + α
initialvalue1(::GeneralizedLaguerreFunction{T, α}, x) where {T, α} = (1 - x + α)*exp(-x / 2)
initialvalue0(::GeneralizedLaguerreFunction{T, α}, x) where {T, α} = exp(- x / 2)
domain(::GeneralizedLaguerreFunction) = (0, Inf)

const LaguerreFunction{T} = GeneralizedLaguerreFunction{T, 0}
(basis::LaguerreFunction)(n::Int, x::Float64, params::Array{Float64}) = basis(n, x, params[1])
(basis::LaguerreFunction)(indices::Vector{Int}, x::Float64, params::Array{Float64}) = basis(indices, x, params[1])

struct GeneralizedLaguerreCf{T, α} <: CentralForceInteraction
    polynomials::GeneralizedLaguerreFunction{T, α}
    GeneralizedLaguerreCf{T, α}() where {T, α} = new(GeneralizedLaguerreFunction{T, α}())
end
(basis::GeneralizedLaguerreCf)(n::Int, x::Float64) = basis.polynomials(n, x)
(basis::GeneralizedLaguerreCf)(n::Int, x::Float64, r0::Float64) = basis.polynomials(n, x, r0)
(basis::GeneralizedLaguerreCf)(indices::Vector{Int}, x::Float64) = basis.polynomials(indices, x)
(basis::GeneralizedLaguerreCf)(indices::Vector{Int}, x::Float64, r0::Float64) = basis.polynomials(indices, x, r0)

const LaguerreCf{T} = GeneralizedLaguerreCf{T, 0}
(basis::LaguerreCf)(n::Int, x::Float64, params::Array{Float64}) = basis(n, x, params[1])
(basis::LaguerreCf)(indices::Vector{Int}, x::Float64, params::Array{Float64}) = basis(indices, x, params[1])

recursive_eval_array!(values::Vector{T}, op::GeneralizedLaguerreCf{T}, x::T, r0::T) where T = recursive_eval_array!(values, op.polynomials, x/r0)
recursive_eval_array!(values::Vector{T}, op::GeneralizedLaguerreCf{T}, x::T, params::Array{T}) where T = recursive_eval_array!(values, op.polynomials, x/params[1])


struct Jacobi{T, α, β}  <: OrthogonalPolynomialBasis{T} end
@inline function recurrence_coefficients(::Jacobi{T, α, β}, n) where {T, α, β}
    n2αβ = 2n + α + β
    return 2(n + 1)*(n + α + β + 1)*n2αβ, (n2αβ + 1)*(α - β)*(α + β), (n2αβ + 1)*(n2αβ + 2)*n2αβ, 2(n + α)*(n + β)*(n2αβ + 2) #typo? add * between ()()
end
initialvalue1(::Jacobi{T, α, β}, x) where {T, α, β} = T(0.5*(α - β) + 0.5*(α + β + 2)*x)
domain(::Jacobi) = (-1, 1)


# const LegendreFunction{T} = Jacobi{T, 0, 0}
# const ChebyshevFunction{T} = Jacobi{T, 0.5, 0.5}

struct LegendreP{T} <: OrthogonalPolynomialBasis{T} end
@inline recurrence_coefficients(::LegendreP{T}, n) where T = n + 1, zero(T), 2n + 1, n
initialvalue1(::LegendreP{T}, x) where T = x
domain(::LegendreP) = (-1, 1)
(basis::LegendreP)(n::Int, x::Float64, ::Any) = basis(n,x)

struct ChebyshevT{T} <: OrthogonalPolynomialBasis{T} end # This is not quite UltraSpherical{1} because of initalvalue1
@inline recurrence_coefficients(::ChebyshevT{T}, n) where T = one(T), zero(T), 2*one(T), one(T)
initialvalue1(::ChebyshevT{T}, x) where {T} = x
domain(::ChebyshevT) = (-1, 1)
(basis::ChebyshevT)(n::Int, x::Float64, ::Any) = basis(n,x)

struct ChebyshevU{T} <: OrthogonalPolynomialBasis{T} end
@inline recurrence_coefficients(::ChebyshevU{T}, n) where T = one(T), zero(T), 2*one(T), one(T)
initialvalue1(::ChebyshevU{T}, x) where {T} = 2x
domain(::ChebyshevU) = (-1, 1)
(basis::ChebyshevU)(n::Int, x::Float64, ::Any) = basis(n,x)

struct ChebyshevC{T} <: OrthogonalPolynomialBasis{T} end
(basis::ChebyshevC{T})(m::Int, θ::Float64) where T = eval_chebyshev(m, T(θ))
(basis::ChebyshevC{T})(m::Int, θ::Float64,::Any) where T = basis(m, θ)

function eval_chebyshev(m::Int, θ::T) where T
    x = cos(θ)
    y = sin(θ)
    (m >= 0) && ( return ChebyshevT{T}()(m, x))
    (m < 0) && ( return ChebyshevU{T}()(-m-1, x)*y)
end

struct AssociatedLegendreP{T} <: OrthogonalPolynomialBasis{T} end
domain(::AssociatedLegendreP) = (-1, 1)
(basis::AssociatedLegendreP)(index::Tuple{Int, Int}, x::Float64) = recursive_eval(basis, index, x) # indices go (l, m)
(basis::AssociatedLegendreP)(index::Tuple{Int, Int}, x::Float64, r0::Float64) = recursive_eval(basis, index, x/r0)
(basis::AssociatedLegendreP)(indices::Vector{Tuple{Int, Int}}, x::Float64) = recursive_eval_array(basis, indices, x)
(basis::AssociatedLegendreP)(indices::Vector{Tuple{Int, Int}}, x::Float64, r0::Float64) = recursive_eval_array(basis, indices, x/r0)
(basis::AssociatedLegendreP)(index::Tuple{Int, Int}, x::Float64, ::Any) = basis(index, x)

function recursive_eval_array(op::AssociatedLegendreP, indices::Vector{Tuple{Int, Int}}, x::T) where{T}
    lmax = maximum([i[1] for i ∈ indices])
    mmax = maximum([i[2] for i ∈ indices]) # Can slightly be optimized by taking a different value of mmax per value of l
    values = ones(T, lmax+1, mmax+1) # Values of the polynomials at (l, m) from (0, 0) to (lmax, mmax)

    # Initialization
    values[1, 1] = one(T)
    lmax > 0 && (values[2, 1] = x)
    
    # Recursive evaluation
    for l ∈ 1:1:lmax # First, recurrence on l
        if l > 1
            values[l+1, 1] = (2*l-1)*x*values[l, 1]-(l-1)*values[l-1, 1]
            values[l+1, 1] /= l
        end
        
        for m ∈ 1:1:min(mmax, l)
            if abs(x) == 1
                values[l+1, m+1] = 0
            else
                if m > 1 # Recurrence on m
                    values[l+1, m+1] = -2*(m-1)*x*values[l+1, m]/sqrt(1-x^2)-(l+m-1)*(l-m+2)*values[l+1, m-1]
                else # Initialization of recurrence on m
                    values[l+1, m+1] = l*x*values[l+1, 1]-l*values[l, 1]
                    values[l+1, m+1] /= sqrt(1-x^2)
                end
            end
        end

    end

    return [values[index.+(1,1)...] for index ∈ indices]
end

function recursive_eval(op::AssociatedLegendreP, index::Tuple{Int, Int}, x::T) where {T}
    recursive_eval_array(op, [index], x)[1]
end






# for each polynomials F_n above (Chebyshev, Legendre, Laguerre, etc.), 
# one often needs to calculate F_n(x/y), but y->0 is not really great numerically.
# To get around, one can define modified polynomials f_n(x,y) from 
#    F_n(x/y) === f_n(x,y) y^(-n)
# we can readily show that if F_n(z) follows the recurrence relation
#    a1 * F_{n+1}(z) = (a2 + a3 * z) * F_{n}(z) - a4 * F_{n-1}(z)
# then f_n(x,y) follows
#    a1 * f_{n+1}(x,y) = (a2*y + a3*x) * f_{n}(x,y) - a4*y^2 * f_{n-1}(x,y)

abstract type modOrthogonalPolynomialBasis{T} <: InteractionBasis end
initialvalue0(::modOrthogonalPolynomialBasis{T}, x, y) where {T} = one(T) 
(basis::modOrthogonalPolynomialBasis)(n::Int, x::Float64, y::Float64) = recursive_eval(basis, n, x, y)
function recursive_eval(op::modOrthogonalPolynomialBasis, n::Int, x::T, y::T) where {T} 
    # Intialization 
    n == 0 && return initialvalue0(op, x, y)
    n == 1 && return initialvalue1(op, x, y)

    x0, x1 = initialvalue0(op, x, y), initialvalue1(op, x, y)

    # Recursive evaluation
    for i = 1:n - 1
        a1, a2, a3, a4 = recurrence_coefficients(op, i) # typo? n=>i
        x1, x0 = (a2*y+a3*x)*x1 - a4*y^2*x0, x1 
        x1 /= a1
    end

    return x1 
end


struct modChebyshevT{T} <: modOrthogonalPolynomialBasis{T} end 
@inline recurrence_coefficients(::modChebyshevT{T}, n) where T = one(T), zero(T), 2*one(T), one(T)
initialvalue1(::modChebyshevT{T}, x, y) where {T} = x


struct modChebyshevU{T} <: modOrthogonalPolynomialBasis{T} end 
@inline recurrence_coefficients(::modChebyshevU{T}, n) where T = one(T), zero(T), 2*one(T), one(T)
initialvalue1(::modChebyshevU{T}, x, y) where {T} = 2*x


# new recursive eval to populate an array of base functions
function recursive_eval_array!(values::Vector, op::modOrthogonalPolynomialBasis, numr, denom)
    
    N = length(values) - 1;

    values[1] = initialvalue0(op, numr, denom)
    N > 0 && (values[2] = initialvalue1(op, numr, denom))

    for n = 2:N
        a1, a2, a3, a4 = recurrence_coefficients(op, n-1)
        values[n+1] = ((a2*denom + a3*numr)*values[n] - a4*denom*denom*values[n-1]) / a1
    end

end

function recursive_eval_array!(values::Vector, op::modOrthogonalPolynomialBasis, rij_vec, ni_vec, nj_vec;
                               crossproduct=false)
    
    ni_rij = dot(ni_vec, rij_vec)
    nj_rij = dot(nj_vec, rij_vec)
    ni_nj  = dot(ni_vec, nj_vec)
    
    numr = ni_nj - ni_rij*nj_rij
    denom = sqrt(1.0-ni_rij*ni_rij) * sqrt(1.0 - nj_rij*nj_rij)

    recursive_eval_array!(values, op, numr, denom)

    if crossproduct
        nixnj_rij = crossdot(ni_vec, nj_vec, rij_vec)
        values .*= nixnj_rij
    end 

end



# for asscoiated Legendre P_l^m (m>=0)
# one can define "modified" associated Legendre p_l^m from
#    P_l^m(x) === p_l^m(x) * (1-x^2)^(m/2)
# Using the recurrence relation of the associated Legendre,
# one can derive the recurrence relation for p_l^m (see notes)
struct modAssociatedLegendreP{T} <: modOrthogonalPolynomialBasis{T} end

function recursive_eval_array!(values::Vector, ::modAssociatedLegendreP, x)
    
    # P[ell, m] is stored in values[ell*(ell+1)/2+m+1]
    
    N = floor(Int,sqrt(length(values)*2)) - 1;

    fill!(values, 0.0)
    
    # diagonal
    for ell = 0:N
        ind  = div((ell+2)*(ell+1), 2)
        ind_ = div((ell+1)*ell, 2)
        (ell == 0) ? (values[ind] = 1.0; continue) : ()
        values[ind] = -(2*ell-1)*values[ind_]
    end
    
    # off diagonal -1
    for ell = 1:N
        ind  = div((ell+3)*ell, 2)
        ind_ = div((ell+1)*ell, 2)
        values[ind] = x*(2*ell-1)*values[ind_]
    end
    
    # others
    for dlm = 2:N
        for m = 0:(N-dlm)
            ell = m + dlm
            ind = div(ell*(ell+1),2)+m+1
            ind_ = div((ell-1)*ell,2)+m+1
            ind__ = div((ell-1)*ell,2)+m+2
            values[ind] = ( (ell+m)*x*values[ind_] + (1-x*x)*values[ind__] ) / dlm
        end
    end
    
    # normalization constants
    for ell = 0:N

        norm_coeff = ell + 0.5
        
        for m = 0:ell
            
            ind = div(ell*(ell+1),2)+m+1
            
            values[ind] *= sqrt(norm_coeff)
            
            (m==ell) ? (break;) : (norm_coeff /= (ell + m + 1); norm_coeff /= (ell - m);)    
            
        end
        
    end
    
    nothing
end

recursive_eval_array!(values::Vector, op::modAssociatedLegendreP, rij_vec, n_vec) =  recursive_eval_array!(values, op, dot(rij_vec, n_vec) )
# =============================================================================





abstract type Geometry end
struct FlatND <: Geometry end
struct PbcND <: Geometry
    lbox::Vector{Float64}
    ispbc::BitVector
end
struct SphericalSurface <: Geometry
    R::Float64
end





# =============================================================================
abstract type VecForceTerm{r_basis, n_basis} end


struct ϕ{r_basis, n_basis} <: VecForceTerm{r_basis, n_basis}
    k::Int
    nx::Int
    ny::Int
    nz::Int
    sym::Symbol
    uvec::Symbol
    function ϕ{r_basis, n_basis}(k::Int, nx::Int, ny::Int, nz::Int, s::Symbol, u::Symbol) where {r_basis, n_basis}
        okterm = term_check(Val(s), k, nx, ny, nz, u);
        !okterm && error("ϕ symmetry not satisfied!")
        new{r_basis, n_basis}(k, nx, ny, nz, s, u)
    end
end
ϕ(r_basis, n_basis, k::Int, nx::Int, ny::Int, nz::Int, s::Symbol, u::Symbol) = ϕ{r_basis, n_basis}(k, nx, ny, nz, s, u)
# uvec
term_vec(trm::ϕ, ni_vec, nj_vec, rij_vec) = term_vec(Val(trm.uvec), ni_vec, nj_vec, rij_vec)
term_vec(::Val{:rij}, ni_vec, nj_vec, rij_vec) = rij_vec
term_vec(::Val{:nj}, ni_vec, nj_vec, rij_vec) = nj_vec
term_vec(::Val{:ni}, ni_vec, nj_vec, rij_vec) = ni_vec
term_vec(::Val{:rij}, rij_vec::Vector{Float64}) = rij_vec
term_vec(::Val{:zxrij}, rij_vec::Vector{Float64}) = [-rij_vec[2],-rij_vec[1]]

# symmetry check
term_check(::Val{:ϕ₁}, k::Int, nx::Int, ny::Int, nz::Int,u::Symbol) = iseven(nx + nz) && iseven(ny + nz) && (k >= 0) & (u == :rij)
term_check(::Val{:ϕ₂}, k::Int, nx::Int, ny::Int, nz::Int,u::Symbol) = iseven(nx + nz) && isodd(ny + nz) && (k >= 0) & (u == :nj)
term_check(::Val{:ϕ₃}, k::Int, nx::Int, ny::Int, nz::Int,u::Symbol) = isodd(nx + nz) && iseven(ny + nz) && (k >= 0) & (u == :ni)
term_check(::Val{:ψ₁}, k::Int, nx::Int, ny::Int, nz::Int,u::Symbol) = isodd(nx + nz) && iseven(ny + nz) && (k >= 0) & (u == :rij)
term_check(::Val{:ψ₂}, k::Int, nx::Int, ny::Int, nz::Int,u::Symbol) = isodd(nx + nz) && isodd(ny + nz) && (k >= 0) & (u == :nj)
term_check(::Val{:u}, k::Int, nx::Int, ny::Int, nz::Int,u::Symbol) = true
term_check(::Val{:v}, k::Int, nx::Int, ny::Int, nz::Int,u::Symbol) = true
term_check(::Val{:w}, k::Int, nx::Int, ny::Int, nz::Int,u::Symbol) = true


# magnitude
function term_mag(trm::ϕ{r_basis, n_basis}, rr, ni_rij::Float64, nj_rij::Float64, ni_nj::Float64, r0) where {r_basis, n_basis}
    fmag = 1.0
    fmag *= r_basis(trm.k, rr, r0)
    fmag *= n_basis(trm.nx, ni_rij)
    fmag *= n_basis(trm.ny, nj_rij)
    fmag *= n_basis(trm.nz, ni_nj)
    return fmag
end



function crossdot(v1, v2, v3)
    s = 0.0
    s += (v1[1]*v2[2] - v1[2]*v2[1])*v3[3]
    s += (v1[2]*v2[3] - v1[3]*v2[2])*v3[1]
    s += (v1[3]*v2[1] - v1[1]*v2[3])*v3[2]
    s
end





function gen_ortho_bases(xdata::Vector{T}, nmax::Int, bf=Polynomial()) where T
    
    
    ndata = length(xdata)
    library_matrix = zeros(T, ndata, nmax+1)
    u_this = zeros(T, ndata)
    u_before = zeros(T, ndata)
    c_this = zeros(T, nmax+1)
    
    coeffs_matrix = T.(Matrix(I, nmax+1, nmax+1))
    library_matrix[:,1] .= one(T)
    (nmax==0) && (return library_matrix, coeffs_matrix;)
    
    for n in 1:nmax
        
        u_this .= bf.(n,xdata);
        c_this .= 0.; c_this[n+1] = 1.0
        
        for m in 0:(n-1)
            copy!(u_before, @view(library_matrix[:,m+1]))
            dot_mn = dot(u_before, u_this) / ndata;
            c_this[m+1] = dot_mn
        end
        for m in 0:(n-1)
            copy!(u_before, @view(library_matrix[:,m+1]))
            u_this .-= c_this[m+1] .* u_before
        end
        norm_n = sqrt(dot(u_this,u_this)/ndata)
        c_this ./= norm_n
        u_this ./= norm_n
        copyto!(@view(library_matrix[:, n+1]), u_this)
        copyto!(@view(coeffs_matrix[n+1, :]), c_this)
    end
    
    return library_matrix, coeffs_matrix

end


function gen_ortho_bases(xdata::Vector{T}, Sigma::Matrix{T}, bf=Polynomial()) where T
    
    
    ndata = length(xdata)
    nmax  = size(Sigma, 1) - 1;
    library_matrix = zeros(T, ndata, size(Sigma, 1))
    u_this = zeros(T, ndata)
    u_before = zeros(T, ndata)
    c_this = zeros(T, nmax+1)
    
    library_matrix[:,1] .= one(T)
    (nmax==0) && (return library_matrix[:,1])
    
    for n in 1:nmax
        
        u_this .= Sigma[n+1, n+1] .* bf.(n, xdata);
        
        for m = 0:(n-1)
            copy!(u_before, @view(library_matrix[:,m+1]))
            u_this .-= Sigma[n+1, m+1] .* u_before
        end
        
        copy!(@view(library_matrix[:,n+1]), u_this)
    end
    
    return library_matrix
end



function rec_ortho_bases(xdata::Vector{T}, Sigma::Matrix{T}, coeffs::Vector{T}, bf=Polynomial()) where T
    
    
    ndata = length(xdata)
    nmax  = size(Sigma, 1) - 1;
    library_matrix = zeros(T, ndata, size(Sigma, 1))
    u_this = zeros(T, ndata)
    u_before = zeros(T, ndata)
    c_this = zeros(T, nmax+1)
    
    library_matrix[:,1] .= one(T)
    (nmax==0) && (return coeffs[1]*library_matrix[:,1])
    
    for n in 1:nmax
        
        u_this .= Sigma[n+1, n+1] .* bf.(n, xdata);
        
        for m = 0:(n-1)
            copy!(u_before, @view(library_matrix[:,m+1]))
            u_this .-= Sigma[n+1, m+1] .* u_before
        end
        
        copy!(@view(library_matrix[:,n+1]), u_this)
    end
    
    ff = zeros(T, ndata)
    for i in eachindex(coeffs)
        ff .+= coeffs[i] .* @view(library_matrix[:,i])
    end
    
    return ff
end



function gen_bases(xdata::Vector{T}, nmax::Int, bf=Polynomial()) where T
    
    ndata = length(xdata)
    library_matrix = zeros(T, ndata, nmax+1)
    
    @inbounds@fastmath for i in axes(library_matrix,1), n in 0:nmax
        library_matrix[i, n+1] = bf(n, xdata[i])
    end
    
    return library_matrix
    
end