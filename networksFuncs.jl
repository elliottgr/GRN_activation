## Shared dependency containing functions used for simulating gene regulatory networks
## Only includes functions used in all individual files

## Need to use the Legendre Polynomial package to calculate arbitrary degree LPs
## Statistics is used to quickly calculate variance for the fitness function
## StatsBase is used for sampling when calculating which weight to mutate
using LegendrePolynomials, Statistics, StatsBase
import Base: copy, copy!, (==), size

## Generates a randomized network containing 
## netDepth = layers of network
## netWidth = number of nodes per layer
## this is instantiated as a series of node connection weights (W_m) and node biases (W_b)
## W_m is a matrix (i = netDepth, j = netWidth) of submatrices
## Each submatrix (k, l) contains a float that characterizes the weight from nodes (k, l) -> (i, j)
## W_b is a matrix for the activation bias of node (i, j)

##JVC: Probably save some hassle by using a struct for the network and defining exactly how copies are made, etc.
struct Network
    Wm::Matrix{Matrix{Float64}}
    Wb::Matrix{Float64}

    function Network(Wm::Matrix{Matrix{Float64}}, Wb::Matrix{Float64})
        netDepth, netWidth = size(Wm)
        if size(Wb) != (netDepth, netWidth)
            throw(ArgumentError("Wm and Wb must have the same width and depth"))
        end
        new(Wm, Wb)
    end
end

## Network methods
(==)(x::Network, y::Network) = (x.Wm == y.Wm && x.Wb == y.Wb)
size(x::Network) = size(x.Wb)

function copy(src::Network)
    Wm = hcat([[copy(src.Wm[i,j]) for i in axes(src.Wm)[1]] for j in axes(src.Wm)[2]]...)
    return Network(Wm, copy(src.Wb))
end

function copy!(dest::Network, src::Network)
    if (size(dest) != size(src))
        throw(ArgumentError("Networks must be the same size"))
    end
    
    for i in axes(src.Wm)[1]
        for j in axes(src.Wm)[2]
            dest.Wm[i,j] .= src.Wm[i,j]
        end
    end
    dest.Wb .= src.Wb
end

function generateNetwork(netDepth, netWidth)
    Wm = hcat([[rand(netDepth, netWidth) for i in 1:netDepth] for j in 1:netWidth]...)
    Wb = rand(netDepth, netWidth)

    return Network(Wm, Wb)
end

function generateFilledNetwork(netDepth, netWidth,val::Float64)
    Wm = hcat([[fill(val, (netDepth, netWidth)) for i in 1:netDepth] for j in 1:netWidth]...)
    Wb = fill(val, (netDepth, netWidth))
    
    return Network(Wm, Wb)
end

## calcOj calculates the value of a single node j in layer i
function calcOj(activationFunction::Function, activationScale::Float64, j::Int, i::Int, activationMatrix, Wm, Wb)
    netWidth = size(Wb)[2]
    x = 0 ## activation
    for k in 1:(i-1)
        for l in 1:netWidth
            ## Wm[i, j][k, l] is the network weight
            x += (Wm[i, j][k, l] * activationMatrix[k, l])
        end
    end
    x += Wb[i, j]
    return(activationFunction(activationScale * x)) 
end


## Define g(i), the network response to some input
## is the gradient iterator, g(i) = -1 + (2 * i/100)
## Here I'm just using it as an input range 
## N(g(i)) is just the network evaluation at some g(i) 
## Still calling the function iterateNetwork! for clarity, originally taken from socialGRN project

function iterateNetwork!(activationFunction::Function, activationScale, input, network::Network, activationMatrix)

    ## Calculates the total output of the network,
    ## iterating over calcOj() for each node

    Wm, Wb = network.Wm, network.Wb
    netDepth, netWidth = size(network)
    activationMatrix[1, :] = input ## whatever vector gets passed as the initial response
    for i in 2:netDepth ## Iterating over each layer
        for j in 1:netWidth ## iterating over each node
            activationMatrix[i,j] = calcOj(activationFunction, activationScale, j, i, activationMatrix, Wm, Wb)
        end
    end
    return activationMatrix
end


## Define R(g(i)), the target the network is training towards
## R is just some Legendre Polynomial, which I'm defining
## as R(i, n)  with i as value and with n as degree. 
## Using the "LegendrePolynomial" package to generate these

## We need a function to normalize the Legendre polynomials to a specific range
## since the polynomials are already fixed to [-1, 1], we can 
## easily rescale it to any input/output. Le Nagard et al used [.1, .9] as their interval
function PlNormalized(x, l, min, max)
    r = (max - min) / 2
    return r * (Pl(x, l) + 1) + min
end

## Fitness Evaluation of network
## Need to generate N(g(i)) - R(g(i))
## this function measures the fit of the network versus the chosen legendre polynomial 

function measureNetwork(activationFunction, activationScale, polynomialDegree, network::Network, envRange)
    netDepth, netWidth = size(network)
    x = 0
    for i in envRange
        activationMatrix = zeros(netDepth, netWidth) ## size of the bias vector

        ## Network input is simply a vector containing a single element in the first position
        ## this allows for expanding to more complex problem dimensions later

        input = fill(0.0, netWidth)
        input[1] = i

        ## I've set N_i to simply take the last value from the network activation
        ## in practice, this means most of the outputs in the last layer don't influence anything
        ## but this makes the code functionally similar to what has been used previously
        ## Importantly, networks of width 1 should behave as they have in previous versions of the model

        iterateNetwork!(activationFunction, activationScale, input, network, activationMatrix)
        N_i = activationMatrix[netDepth, netWidth]
        R_i = PlNormalized(i, polynomialDegree, 0, 1)
        # print(" N_i = $N_i   |   R_i = $R_i   |  N - R = $(N_i - R_i) \n")
        x += (N_i - R_i) ^2
    end
    return x
end

function fitness(activationFunction, activationScale, K, polynomialDegree, network)
    envRange = -1:0.02:1
    Var_F = var([PlNormalized(i, polynomialDegree, 0, 1) for i in envRange])
    # return exp((-K * (measureNetwork(activationFunction, activationScale, polynomialDegree, network))^2) / (100*Var_F)) ## With Squared measure
    return exp((-K * (measureNetwork(activationFunction, activationScale, polynomialDegree, network, envRange))) / (length(envRange)*Var_F))
end

## Testing the functions as I go


function mutateNetwork(μ_size, network::Network)

    ## samples a random weight and shifts it
    ## Le Nagard method

    netDepth, netWidth = size(network)

    ## Need to allocate a new array and fill it 
    ## copy() of the original network doesn't allocate new elements
    ## so the output network overwrites the old one
    newNetwork = generateFilledNetwork(netDepth, netWidth, 0.0)
    copy!(newNetwork, network)

    mutationSize = randn()*μ_size

    ## Randomly selecting a weight (i,j,k,l) to mutate
    ## A large portion of them are "silent mutations"
    ## because only a fraction of the mutants will change network outputs
    ## k = 0 will simply be the bias vectors
    while newNetwork == network
        i = sample(1:netDepth)
        j = sample(1:netWidth)
        k = sample(0:i-1)
        l = sample(1:netWidth)

        if k > 0
            newNetwork.Wm[i, j][k, l] += mutationSize
        else
            newNetwork.Wb[i, j] += mutationSize
        end
    end
    return newNetwork
end

function mutateNetwork!(μ_size, network::Network)

    ## samples a random weight and shifts it
    ## Le Nagard method

    netDepth, netWidth = size(network)

    mutationSize = randn()*μ_size

    i = sample(1:netDepth)
    j = sample(1:netWidth)
    k = sample(0:i-1)
    l = sample(1:netWidth)

    if k > 0
        network.Wm[i, j][k, l] += mutationSize
    else
        network.Wb[i, j] += mutationSize
    end
end

## plots the fitness of each timestep in a simulation run
## has a comment capable of plotting the invasion probability as well
function plotReplicatesFitness(simulationResults)
    netDepth, netWidth = size(simulationResults[3][1])
    numReps = length(simulationResults[3])
    titleStr = string("Fitness of $numReps replicates for a $(netDepth) layer network with $netWidth nodes")

    fitnessPlot = plot(1:length(simulationResults[1][1]), simulationResults[1], title = titleStr)
    
    ## This can also produce a stacked plot, uncomment below :)
    # invasionProbPlot = plot(1:length(simulationResults[1][1]), simulationResults[2], legend = :none, title = "Invasion probability")
    # plot(fitnessPlot, invasionProbPlot, layout = (2,1), sharex=true) ## Returns a stacked plot of both figures
    return fitnessPlot
end


## Samples the final network at the end of each replicate simulation
## plots it relative to the predicted value
function plotResponseCurves(activationFunction, activationScale, polyDegree, simulationResults)

    netDepth, netWidth = size(simulationResults[3][1])
    titleStr = string("Network response for a $(netDepth) layer network with $netWidth nodes")
    
    ## Plotting the polynomial curve
    plt = plot(-1:0.02:1, collect([PlNormalized(i, polyDegree, 0, 1) for i in -1:0.02:1]), label = "Target", title = titleStr)
    
    ## Updating it with the fitness of each replicate
    for network in simulationResults[3]
        valueRange = collect(-1:0.02:1) ## the range of input values the networks measure against
        responseCurveValues = []
        for i in valueRange
            LayerOutputs = zeros(Float64, size(network)) ## size of the bias vector
            input = fill(0.0, netWidth)
            input[1] = i
            push!(responseCurveValues, last(iterateNetwork!(activationFunction, activationScale, input, network, LayerOutputs)[netDepth]))
        end
        plt = plot!(-1:0.02:1, responseCurveValues, alpha = 0.5)
    end
    return plt
end

##############################################
## Unit Testing and Debugging functions
##############################################

## Testing that mutation can generate new mutants without overwriting the old individual
function testMutationFunction(netDepth=50, netWidth=10)
    count = 0
    for _ in 1:100
        testNetwork = generateNetwork(netDepth,netWidth)
        testNetworkMutant = mutateNetwork(0.1, testNetwork)
        if testNetwork != testNetworkMutant ## Checking mutation process
            count+= 1
        end
    end
    return count
end

# Φ = (f(x) = (1 - exp(-x^2))) 
# α = 1.0
# testNetwork = generateNetwork(5,6)
# testNetworkMutant = mutateNetwork(0.1, testNetwork)
# testActivationNetwork = zeros(Float64, (5,6)) ## dummy network, will be generated as part of network iteration later on
# calcOj(Φ, α, 1, 1, testActivationNetwork, testNetwork...)
# testInput = rand(Float64, 6)
# polyDegree = 2
# K = 5.0
# N = 100
# iterateNetwork!(Φ, α, testInput, testNetwork, testActivationNetwork)
# measureNetwork(Φ, α, polyDegree, testNetwork)
# fitness(Φ, α, K, polyDegree, testNetwork)
# invasionProbability(Φ, α, K, polyDegree, N, testNetwork, testNetworkMutant)

# ## Blank network testing
# ## Should return zero output
# netDepth = 3
# netWidth = 3
# blankNetwork = generateFilledNetwork(netDepth,netWidth, 0.0)
# testInput = fill(0.0, netWidth)
# testActivationMatrix = zeros(Float64, (netDepth, netWidth))
# print(iterateNetwork!(Φ, α, testInput, blankNetwork, testActivationMatrix))


## Testing / debugging functions
## Generating random phenotypes and showing that they'll have some invasion chance
function debugInvasionProbability(trials, netDepth, netWidth)
    resNet = generateNetwork(netDepth, netWidth)
    testHistory = fill(0.0, trials)
    Φ(x) = (1 - exp(-x^2))
    α = 1.0
    K = 1.0
    polyDegree = 1
    N = 100
    for t in 1:trials
        mutNet = generateNetwork(netDepth, netWidth)
        testHistory[t] = invasionProbability(Φ, α, K, 1, N, resNet, mutNet)
    end
    return testHistory
end