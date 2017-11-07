# backpropagation of fully connected dnn
# lixun 2017-11-07
#
#  X(4) => 8 => 8 => 4 fully connected nn
function backprop(x, y)

    # neron-net specification
    # input layer is not included in the net
    a = Dict{Int64, Array{Float64,1}}(1=>randn(8), 2=>randn(8), 3=>randn(4))
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

    z = Dict{Int64, Array{Float64,1}}()
    # feedforward
    a[0] = view(x, :, 1)
    for i = 1:L    
        z[i] = w[i]*a[i-1] + b[i]
        a[i] = sigmoid.(z[i])
    end
end


sigmoid(x) = 1.0 / (1.0 + exp(-x))
sigmoid_prime(x) = sigmoid(x) * (1.0 - sigmoid(x))

# ∂C/∂a for output activations
∂C∂a(a, y) = a - y

