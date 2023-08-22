## Shared dependency containing functions used for simulating gene regulatory networks
## Only includes functions used in all individual files

## Need to use the Legendre Polynomial package to calculate arbitrary degree LPs
## Statistics is used to quickly calculate variance for the fitness function
## StatsBase is used for sampling when calculating which weight to mutate
using LegendrePolynomials, Statistics, StatsBase

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

## Define g(i), the network response to some input
## is the gradient iterator, g(i) = -1 + (2 * i/100)
## Here I'm just using it as an input range 
## N(g(i)) is just the network evaluation at some g(i) 
## Still calling the function iterateNetwork for clarity

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


## Define R(g(i)), the target the network is training towards
## R is just some Legendre Polynomial, which I'm defining
## as R(i, n)  with i as value and with n as degree. 
## Using the "LegendrePolynomial" package to generate these

## Fitness Evaluation of network
## Need to generate N(g(i)) - R(g(i))

## this function measures the fit of the network versus the chosen legendre polynomial 
function measureNetwork(activation_function, activation_scale, polynomialDegree, network)
    x = 0
    for i in -1:0.02:1
        LayerOutputs = zeros(Float64, size(network[2])) ## size of the bias vector
        # print((last(iterateNetwork(activation_function, activation_scale, i, network, LayerOutputs)) - Pl(i, polynomialDegree) ))
        x += (last(iterateNetwork(activation_function, activation_scale, i, network, LayerOutputs)) - Pl(i, polynomialDegree))
    end
    return x
end


function fitness(activation_function, activation_scale, K, polynomialDegree, network)
    Wm, Wb = network
    Var_F = var(collect([Pl(i, polynomialDegree) for i in -1:0.02:1]))
    return exp((-K * (measureNetwork(activation_function, activation_scale, polynomialDegree, network))^2) / (100*Var_F))

end

function mutateNetwork(μ_size, network)

    ## samples a random weight and shifts it
    ## Le Nagard method

    NetSize = size(network[2])[1]
    weightID = sample(1:(NetSize^2 + NetSize))

    ## Need to allocate a new array and fill it 
    ## copy() of the original network doesn't allocate new elements
    ## so the output network overwrites the old one
    newNetwork = [fill(0.0, (NetSize, NetSize)), fill(0.0, NetSize)] 
    for i in eachindex(network)
        newNetwork[i] = copy(network[i])
    end
    mutationSize = randn()*μ_size
    if weightID <= NetSize^2
        newNetwork[1][weightID] += mutationSize
    else
        newNetwork[2][weightID-(NetSize^2)] += mutationSize
    end
## Alternative code used by JVC 
    #newNetwork[1] += rand(Binomial(1, μ_trait), (NetSize, NetSize)) .* rand(Normal(0, μ_size), (NetSize, NetSize))
    #newNetwork[2] += rand(Binomial(1, μ_trait), NetSize) .* rand(Normal(0, μ_size),NetSize)
    return newNetwork
end

# ## Testing network code
# ## This generates a 2 node network with no weights and calculates 
# ## the resulting fitness to double check implementation of all
# ## network code + the fitness function

# W_m = fill(0.0, (2,2))
# W_b = [0.0, 0.0]
# Φ(x) = (1 - exp(-x^2))
# K = 5.0
# Var_F = var(collect([Pl(i, 1) for i in -1:0.02:1]))
# testFitness = exp( (-K * (-101.0)^2) / (100* Var_F) ) ## Explicitly calculating the fitness without measuring the networks :)

# iterateNetwork(Φ, 1.0, 1, [W_m, W_b], LayerOutputs) ## returns expected value (0) for a zero network
# measureNetwork(Φ, 1.0, 1, [W_m, W_b]) ## returns the expected value (-101.0) for a zero network measured against the null polynomial
# fitness(Φ, 1.0, K, 1, [W_m, W_b])

