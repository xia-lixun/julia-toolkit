# backpropagation of fully connected dnn
# lixun 2017-11-07
#
#  X(4) => 8 => 8 => 4 fully connected nn
# a: activations, for example sigmoid(z) = sigmoid(aw+b) 
# w: weights
# b: biases
# L: nn depth
# z: wa+b

mutable struct net{T <: AbstractFloat}

    a::Array{Array{T,1},1}  # activations, 1st layer as input
    z::Array{Array{T,1},1}  # activation input, z[1] is not in use
    δ::Array{Array{T,1},1}  # errors, [1] is not in use

    w::Array{Array{T,2},1}  # weights, w[1] is not used
    b::Array{Array{T,1},1}  # biases, b[1] is not used

    ∂C∂w::Array{Array{T,2},1}  # ∂C/∂w of all weights, [1] is not in use
    ∂C∂b::Array{Array{T,1},1}  # ∂C/∂b of all biases, [1] is not in use

    L::Int64  # layers

    # for SGD
    ∇w::Array{Array{T,2},1}
    ∇b::Array{Array{T,1},1}
    
    function net{T}(layers) where T <: AbstractFloat
        L = length(layers)

        a = [zeros(T,i) for i in layers]
        z = deepcopy(a)
        δ = deepcopy(a)
    
        w = [[zeros(T,length(a[1]),1)]; [randn(T,length(a[i]),length(a[i-1])) for i = 2:L]]
        b = [[zeros(T,length(a[1]))]; [randn(T,length(a[i])) for i = 2:L]]
    
        ∂C∂w = deepcopy(w)
        ∂C∂b = deepcopy(b)
        ∇w = deepcopy(w)
        ∇b = deepcopy(b)

        new(a,z,δ,w,b,∂C∂w,∂C∂b,L,∇w,∇b)
    end
end



# x: Array{Array{T,1},1} for best performance
function backprop!(nn::net{T}, x, y) where T <: AbstractFloat

    L = nn.L

    # feedforward
    nn.a[1] = x
    for i = 2 : L
        nn.z[i] .= (nn.w[i] * nn.a[i-1]) .+ nn.b[i]
        nn.a[i] .= sigmoid.(nn.z[i])
    end
    # nn.a, nn.z

    # output error ∇aC ⦿ σ'(z[L])
    nn.δ[L] .= ∂C∂a.(nn.a[L], y) .* sigmoid_prime.(nn.z[L])
    nn.∂C∂b[L] .= nn.δ[L]
    ∂C∂w!(nn.∂C∂w[L], nn.a[L-1], nn.δ[L])
   
    # for each l = L-1, L-2 ... 2 compute δ[l] = (transpose(w[l+1])δ[l+1]) ⦿ σ'(z[l])
    #    ∂C/∂w[l][j,k] = a[l-1][k] * δ[l][j]
    #    ∂C/∂b[l][j] = δ[l][j]
    for l = L-1:-1:2
        nn.δ[l] .= (transpose(nn.w[l+1]) * nn.δ[l+1]) .* sigmoid_prime.(nn.z[l])
        nn.∂C∂b[l] .= nn.δ[l]
        ∂C∂w!(nn.∂C∂w[l], nn.a[l-1], nn.δ[l])
    end
    nothing
end    



function update_minibatch(nn::net{T}, minibatch::Array{Tuple{Array{T,1},Array{T,1}},1}, η) where T <: AbstractFloat
    
    for i in nn.∇w
        fill!(i, zero(T))
    end
    for i in nn.∇b
        fill!(i, zero(T))
    end
    for (x,y) in minibatch
        backprop!(nn, x, y)
        for (i,j) in enumerate(nn.∇w)
            j .+= nn.∂C∂w[i]
        end
        for (i,j) in enumerate(nn.∇b)
            j .+= nn.∂C∂b[i]
        end
    end
    m = length(minibatch)
    nn.w .-= ((η/m) .* nn.∇w)
    nn.b .-= ((η/m) .* nn.∇b)
end


function evaluate(nn::net{T}, x) where T <: AbstractFloat
    # feedforward
    nn.a[1] = x
    for i = 2 : nn.L
        nn.z[i] .= (nn.w[i] * nn.a[i-1]) .+ nn.b[i]
        nn.a[i] .= sigmoid.(nn.z[i])
    end
    nn.a[nn.L]
end


f(x,y) = (sin(x+y)+1)/2
function simple()
    nn = net{Float64}([2,16,16,1])
    epoch = 500

    for i = 1:epoch
        x = rand(1000)
        y = rand(1000)
        z = f.(x,y)
        minibatch = [([x[j], y[j]], [z[j]]) for j = 1:1000]
        update_minibatch(nn, minibatch, 0.001)
        info(nn.w[2][4])
    end
    evaluate(nn, [0.3, 0.6])
end




# example net([2,3,2,3])
function backprop(x, y)

    # neron-net specification
    # input layer is not included in the net
    #a = Dict{Int64, Array{Float64,1}}(1=>zeros(3), 2=>zeros(2), 3=>zeros(3))
    a = [zeros(x) for x in layers]
    z = deepcopy(a)
    δ = deepcopy(a)

    w = Dict{Int64, Array{Float64,2}}()
    b = Dict{Int64, Array{Float64,1}}()
    L = length(layers)

    w[1] = randn(length(a[2]), length(a[1]))
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