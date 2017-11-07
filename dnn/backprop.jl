# backpropagation of fully connected dnn
# lixun 2017-11-07
#
#  X(4) => 8 => 8 => 4 fully connected nn
# a: activations, for example sigmoid(z) = sigmoid(aw+b) 
# w: weights
# b: biases
# L: nn depth
# z: wa+b
function backprop(x, y)

    # neron-net specification
    # input layer is not included in the net
    a = Dict{Int64, Array{Float64,1}}(1=>zeros(3), 2=>zeros(2), 3=>zeros(3))
    z = deepcopy(a)
    δ = deepcopy(a)

    w = Dict{Int64, Array{Float64,2}}()
    b = Dict{Int64, Array{Float64,1}}()
    L = length(keys(a))

    w[1] = randn(length(a[1]), size(x,1))
    for i = 2:L
        w[i] = randn(length(a[i]), length(a[i-1]))
    end
    for i = 1:L
        b[i] = randn(length(a[i]))
    end

    ∂C∂b = deepcopy(b)
    ∂C∂w = deepcopy(w)

    # 1. feedforward
    a[0] = view(x, :, 1)
    for i = 1:L    
        z[i] .= (w[i]*a[i-1]) .+ b[i]
        a[i] .= sigmoid.(z[i])
    end
    #a,z

    # 2. output error ∇aC ⦿ σ'(z[L])
    δ[L] .= ∂C∂a.(a[L], y) .* sigmoid_prime.(z[L])
    ∂C∂b[L] .= δ[L]
    #for k = 1:size(w[L],2)
    #    for j = 1:size(w[L],1)
    #        ∂C∂w[L][j,k] = a[L-1][k] * δ[L][j]
    #    end
    #end
    ∂C∂w!(∂C∂w[L], a[L-1], δ[L])

    # 3. for each l = L-1, L-2 ... 1 compute δ[l] = (transpose(w[l+1])δ[l+1]) ⦿ σ'(z[l])
    #    ∂C/∂w[l][j,k] = a[l-1][k] * δ[l][j]
    #    ∂C/∂b[l][j] = δ[l][j]
    for l = L-1:-1:1
        δ[l] .= (transpose(w[l+1]) * δ[l+1]) .* sigmoid_prime.(z[l])
        ∂C∂b[l] .= δ[l]
        #for k = 1:size(w[l],2)
        #    for j = 1:size(w[l],1)
        #        ∂C∂w[l][j,k] = a[l-1][k] * δ[l][j]
        #    end
        #end
        ∂C∂w!(∂C∂w[l], a[l-1], δ[l])    
    end
    ∂C∂w, ∂C∂b
end




sigmoid(x) = 1.0 / (1.0 + exp(-x))
sigmoid_prime(x) = sigmoid(x) * (1.0 - sigmoid(x))

# ∂C/∂a for output activations
∂C∂a(a, y) = a - y

function ∂C∂w!(∂C∂w, a, δ)
    for k = 1:size(∂C∂w,2)
        for j = 1:size(∂C∂w,1)
            ∂C∂w[j,k] = a[k] * δ[j]
        end
    end    
end