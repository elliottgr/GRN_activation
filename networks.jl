## Builds the networks from scratch using code from the social FFN project
## arranges them into populations, and assesses their evolution

using LegendrePolynomials, Statistics

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


function iterateNetwork(activation_function::Function, activation_scale, input, network, prev_out)
    ##############################
    ## Calculates the total output of the network,
    ## iterating over calcOj() for each layer
    ##############################
    Wm, Wb = network ## for clarity in the future
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
network = [W_m, W_b]
## Temporarily holds layer outputs as the network iterates itself
LayerOutputs = zeros(NetSize)

## test params
Φ(x) = x ## Identity function to test activation
α = 1.0 ## activation scale
K = 5.0
calcOj(Φ, α, 1, LayerOutputs, W_m, W_b)
polynomialDegree = 3
## To do:

## Define g(i), the network response to some input
## is the gradient iterator, g(i) = -1 + (2 * i/100)
## Here I'm just using it as an input range 
## N(g(i)) is just the network evaluation at some g(i) 
## Still calling the function iterateNetwork for clarity

iterateNetwork(Φ, α, 0, network, LayerOutputs)

## Define R(g(i)), the target the network is training towards
## R is just some Legendre Polynomial, which I'm defining
## as R(i, n)  with i as value and with n as degree. 
## Using the "LegendrePolynomial" package to generate these
##  

## Fitness Evaluation of network
## Need to generate N(g(i)) - R(g(i))

## this function measures the fit of the network versus the chosen legendre polynomial 
function measureNetwork(activation_function, activation_scale, polynomialDegree, network)
    x = 0
    for i in -1:0.02:1
        LayerOutputs = zeros(Float64, size(network[2])) ## size of the bias vector
        x += (last(iterateNetwork(activation_function, activation_scale, i, network, LayerOutputs)) - Pl(i, polynomialDegree))
    end
    return x
end

measureNetwork(Φ, α, polynomialDegreee, network)

## Defining the fitness function

function fitness(activation_function, activation_scale, K, polynomialDegree, network)
    Wm, Wb = network
    Var_F = var([Pl(i, polynomialDegree) for i in -1:0.02:1])
    return exp(-K * measureNetwork(activation_function, activation_scale, polynomialDegree, network))

end
fitness(Φ, α, K, polynomialDegree, [W_m, W_b])


## Define population of N networks

N = 100
population = fill(network, N)