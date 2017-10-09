# bilinear transformation of transfer function from s-domain to z-domain
# via s = 2/T (z-1)/(z+1)
# let ζ = z^(-1) we have s = -2/T (ζ-1)/(ζ+1)
# 
#          b_m s^m + b_(m-1) s^(m-1) + ... + b_1 s + b_0
# H(s) = -------------------------------------------------
#          a_n s^n + a_(n-1) s^(n-1) + ... + a_1 s + a_0
#
# So 
#
#          b_m (-2/T)^m (ζ-1)^m / (ζ+1)^m  + ... + b_1 (-2/T) (ζ-1)/(ζ+1) + b_0 
# H(ζ) = -------------------------------------------------------------------------
#          a_n (-2/T)^n (ζ-1)^n / (ζ+1)^n  + ... + a_1 (-2/T) (ζ-1)/(ζ+1) + a_0
#
# Since we assume H(s) is rational, so n ≥ m, multiply num/den with (ζ+1)^n ans we have
#
#          b_m (-2/T)^m (ζ-1)^m (ζ+1)^(n-m)  + b_(m-1) (-2/T)^(m-1) (ζ-1)^(m-1) (ζ+1)^(n-m+1) + ... + b_1 (-2/T) (ζ-1)(ζ+1)^(n-1) + b_0 (ζ+1)^n
# H(ζ) = ---------------------------------------------------------------------------------------------------------------------------------------
#          a_n (-2/T)^n (ζ-1)^n  + a_(n-1) (-2/T)^(n-1) (ζ-1)^(n-1) (ζ+1) ... + a_1 (-2/T) (ζ-1)(ζ+1)^(n-1) + a_0 (ζ+1)^n
#
#
#         B[0] + B[1]ζ + B[2]ζ^2 + ... B[m]ζ^m
# H(ζ) = ---------------------------------------
#         A[0] + A[1]ζ + A[2]ζ^2 + ... A[n]ζ^n
using Polynomials

function bilinear(b, a, fs)
    m = size(b,1)-1
    n = size(a,1)-1
    p = Polynomials.Poly{Float64}(0.0)
    q = Polynomials.Poly{Float64}(0.0)
    g = (-2*fs)^n

    br = flipdim(b,1)
    ar = flipdim(a,1)

    for i = m:-1:0
        p = p + (br[i+1] * ((-2*fs)^i) * poly(ones(i)) * poly(-ones(n-i)))
    end
    for i = n:-1:0
        q = q + (ar[i+1] * ((-2*fs)^i) * poly(ones(i)) * poly(-ones(n-i)))        
    end
    p = coeffs(p)
    q = coeffs(q)
    g = q[1]
    return (p/g,q/g)
end

function convolve(a::Array{T,1}, b::Array{T,1}) where T <: AbstractFloat
    m = size(a,1)
    n = size(b,1)

    ar = flipdim(a,1)

    bx = [zeros(T,m-1); b; zeros(T,m-1)]
    ax = [zeros(T,(n-1)+(m-1)); ar; zeros(T,n-1)]
    y = zeros(T,m+n-1)

    for i = 1:m+n-1
        y[i] = dot(bx[i:i+(m+n-2)], ax[(n+1)-i:(m+n-2)+(n+1)-i])
    end
    y
end

# example: create a-weighting filter
function weighting_a()
    
    f1 = BigFloat(20.598997)
    f2 = BigFloat(107.65265)
    f3 = BigFloat(737.86223)
    f4 = BigFloat(12194.217)
    A1000 = BigFloat(1.9997)

    p = [ ((2π*f4)^2) * (10^(A1000/20)), 0, 0, 0, 0 ]
    q = conv([1, 4π*f4, (2π*f4)^2], [1, 4π*f1, (2π*f1)^2])
    q = conv(conv(q,[1, 2π*f3]),[1, 2π*f2])
    
    println(p)
    println(q)
end