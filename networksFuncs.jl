## Shared dependency containing functions used for simulating gene regulatory networks
## Only includes functions used in all individual files

## Need to use the Legendre Polynomial package to calculate arbitrary degree LPs
## Statistics is used to quickly calculate variance for the fitness function
## StatsBase is used for sampling when calculating which weight to mutate
using LegendrePolynomials, Statistics, StatsBase

## Generates a randomized network containing 
## netDepth = layers of network
## netWidth = number of nodes per layer
## this is instantiated as a series of node connection weights (W_m) and node biases (W_b)
## W_m is a matrix (i = netDepth, j = netWidth) of submatrices
## Each submatrix (k, l) contains a float that characterizes the weight from nodes (k, l) -> (i, j)
## W_b is a matrix for the activation bias of node (i, j)

function generateNetwork(netDepth, netWidth)
    Wm = fill(fill(rand(Float64), (netDepth, netWidth)), (netDepth, netWidth))
    # for i_j in eachindex(Wm)
    #     Wm[i_j] = rand(Float64, (netDepth, netWidth))
    # end
    Wb = rand(Float64, (netDepth, netWidth))
    return [Wm, Wb]
end

## calcOj calculates the value of a single node j in layer i

function calcOj(activation_function::Function, activation_scale::Float64, j::Int, i::Int, activationMatrix, Wm, Wb)
    netWidth = size(Wm[1])[2]
    x = 0 ## activation
    for k in 1:(i-1)
        for l in 1:netWidth
            ## Wm[i, j][k, l] is the network weight
            x += (Wm[i, j][k, l] * activationMatrix[k, l])
        end
    end
    x += Wb[i, j]
    return(activation_function(activation_scale * x)) 
end

## Testing the functions as I go
Φ = (f(x) = (1 - exp(-x^2))) 
α = 1.0
testNetwork = generateNetwork(5,6)
testActivationNetwork = rand(Float64, (5,6)) ## dummy network, will be generated as part of network iteration later on
calcOj(Φ, α, 1, 1, testActivationNetwork, testNetwork...)

## Define g(i), the network response to some input
## is the gradient iterator, g(i) = -1 + (2 * i/100)
## Here I'm just using it as an input range 
## N(g(i)) is just the network evaluation at some g(i) 
## Still calling the function iterateNetwork for clarity, originally taken from socialGRN project

function iterateNetwork(activation_function::Function, activation_scale, input, network, prev_out)

end

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

## Define R(g(i)), the target the network is training towards
## R is just some Legendre Polynomial, which I'm defining
## as R(i, n)  with i as value and with n as degree. 
## Using the "LegendrePolynomial" package to generate these

## We need a function to normalize the Legendre polynomials to a specific range
## since the polynomials are already fixed to [-1, 1], we can 
## easily rescale it to any input/output. Le Nagard et al used [.1, .9] as their interval
function PlNormalized(x, l, min, max)
    a = (max - min) / 2
    b = 1 - (a)
    return (a * Pl(x, l)) + b
end

## Fitness Evaluation of network
## Need to generate N(g(i)) - R(g(i))
## this function measures the fit of the network versus the chosen legendre polynomial 

function measureNetwork(activation_function, activation_scale, polynomialDegree, network)
    x = 0
    for i in -1:0.02:1
        LayerOutputs = zeros(Float64, size(network[2])) ## size of the bias vector
        N_i = last(iterateNetwork(activation_function, activation_scale, i, network, LayerOutputs))
        R_i = PlNormalized(i, polynomialDegree, 0, 1)
        # print(" N_i = $N_i   |   R_i = $R_i   |  N - R = $(N_i - R_i) \n")
        x += abs(N_i - R_i) ## This is different from Le Nagard et al, as they merely summed the difference rather than the absolute value
        # x += N_i - R_i ## Le Nagard's method
    end
    return x
end

function fitness(activation_function, activation_scale, K, polynomialDegree, network)
    Wm, Wb = network
    Var_F = var(collect([PlNormalized(i, polynomialDegree, 0, 1) for i in -1:0.02:1]))
    # return exp((-K * (measureNetwork(activation_function, activation_scale, polynomialDegree, network))^2) / (100*Var_F)) ## With Squared measure
    return exp((-K * (measureNetwork(activation_function, activation_scale, polynomialDegree, network))) / (100*Var_F))
end

function mutateNetwork(μ_size, network)

    ## samples a random weight and shifts it
    ## Le Nagard method

    netSize = size(network[2])[1]


    ## Need to allocate a new array and fill it 
    ## copy() of the original network doesn't allocate new elements
    ## so the output network overwrites the old one
    newNetwork = [fill(0.0, (netSize, netSize)), fill(0.0, netSize)] 
    for i in eachindex(network)
        newNetwork[i] = copy(network[i])
    end
    mutationSize = randn()*μ_size

    ## There are N*(N+1)/2 elements in an upper triangular matrix, including the diagonal
    ## If we consider those plus the N elements in the bias vector, we have N + (N*(N+1)/2) possible weights to mutate

    weightID = Int64(sample(1:(netSize+(netSize*(netSize+1)/2))))
    if weightID <= netSize
        newNetwork[2][weightID] += mutationSize
    else
        i = weightID - netSize
        j = 1
        ## should iterate down to provide the i, j pair of the upper triangular matrix :)
        while i > netSize
            i -= netSize - j
            j += 1
        end
        newNetwork[1][j, Int(i)] += mutationSize
    end
    ## Alternate code that can mutate any weight, even those unused by the simulation
    # weightID = sample(1:(netSize^2 + netSize))
    # if weightID <= netSize^2
    #     newNetwork[1][weightID] += mutationSize
    # else
    #     newNetwork[2][weightID-(netSize^2)] += mutationSize
    # end

    ## Alternative code used by JVC 
    #newNetwork[1] += rand(Binomial(1, μ_trait), (netSize, netSize)) .* rand(Normal(0, μ_size), (netSize, netSize))
    #newNetwork[2] += rand(Binomial(1, μ_trait), netSize) .* rand(Normal(0, μ_size),netSize)
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

