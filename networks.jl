## Builds the networks from scratch using code from the social FFN project
## arranges them into populations, and assesses their evolution


## Capture 
## W_m = weight matrices for node connection weights
## W_b = weight vector for node biases
## prev_out = running total of the iterations on the node
## j = current layer of iteration
## activation_scale = population level scaling parameter of inputs to influence behavior response
## activation_function = the thing we're testing :)


function calcOj(activation_function::Function, activation_scale::Float64, j::Int, prev_out, Wm, Wb)
    ##############################
    ## Iterates a single layer of the Feed Forward network
    ##############################
    ## dot product of Wm and prev_out, + node weights. Equivalent to x = dot(Wm[1:j,j], prev_out[1:j]) + Wb[j]
    ## doing it this way allows scalar indexing of the static arrays, which is significantly faster and avoids unnecessary array invocation
    x = 0
    for i in 1:j-1
        x += (Wm[i, j] * prev_out[i]) 
    end
    x += Wb[j]
    return(activation_function(activation_scale * x)) 
end


function iterateNetwork(activation_function::Function, activation_scale, input, Wm, Wb, prev_out)
    ##############################
    ## Calculates the total output of the network,
    ## iterating over calcOj() for each layer
    ##############################
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = calcOj(activation_function, activation_scale, j, prev_out, Wm, Wb)
    end
    return prev_out
end

## test network with arbitrary values
NetSize = 5
NetWeight = 0.0
NetBias = 0.5
W_m = fill(NetWeight, NetSize, NetSize)
W_b = fill(NetBias, NetSize)

## Temporarily holds layer outputs as the network iterates itself
LayerOutputs = zeros(NetSize)

## test params
Φ(x) = x ## Identity function to test activation
α = 1.0 ## activation scale


## To do:

## Define g(i), the network response to some input
## is the gradient iterator, g(i) = -1 + (2 * i/100)
g = 0.0:1.0:-1.0
iterateNetwork(Φ, α, i, )

## Define R(g(i)), the target the network is training towards

## Fitness Evaluation of network

## Definition of 