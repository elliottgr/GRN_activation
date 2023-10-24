## Shared dependency containing functions used for simulating gene regulatory networks
## Only includes functions used in all individual files

## Need to use the Legendre Polynomial package to calculate arbitrary degree LPs
## Statistics is used to quickly calculate variance for the fitness function
## StatsBase is used for sampling when calculating which weight to mutate
using LegendrePolynomials, Statistics, StatsBase, Random
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
            activationMatrix[i, j] = calcOj(activationFunction, activationScale, j, i, activationMatrix, Wm, Wb)
        end
    end
    return activationMatrix
end

## function to normalize the Legendre polynomials to a specific range
function PlNormalized(x, l, min, max)
    r = (max - min) / 2
    return r * (Pl(x, l) + 1) + min
end

## this function measures the fit of the network versus the chosen legendre polynomial 
function measureNetwork(activationFunction, activationScale, polynomialDegree, network::Network, envRange)
    netDepth, netWidth = size(network)
    x = 0
    input = fill(0.0, netWidth)
    for i in envRange
        activationMatrix = zeros(netDepth, netWidth) ## size of the bias vector
        input[1] = i
        iterateNetwork!(activationFunction, activationScale, input, network, activationMatrix)
        N_i = activationMatrix[netDepth, netWidth]
        R_i = PlNormalized(i, polynomialDegree, 0, 1)
        x += (N_i - R_i) ^2
    end
    return x
end

function fitness(activationFunction, activationScale, K, polynomialDegree, network)
    envRange = -1:0.02:1
    Var_F = var([PlNormalized(i, polynomialDegree, 0, 1) for i in envRange])
    return exp((-K * (measureNetwork(activationFunction, activationScale, polynomialDegree, network, envRange))) / (length(envRange)*Var_F))
end

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

## Necessary to call the distributed workers in higher level scripts, kinda pointless for single threaded versions (will still work tho)
struct simParams
    N::Int
    T::Int
    reps::Int
    activationFunction::Function
    activationScale::Float64
    K::Float64
    polyDegree::Int
    netDepth::Int
    netWidth::Int
    μ_size::Float64
end

## Equivalent to Eq. 5, P(f_0 -> f_i), in le Nagard (2011)
function invasionProbability(activationFunction, activationScale, K, polynomialDegree, N, resNet::Network, mutNet::Network)

    resFitness = fitness(activationFunction, activationScale, K, polynomialDegree, resNet)
    mutFitness = fitness(activationFunction, activationScale, K, polynomialDegree, mutNet)
    fitnessRatio =  resFitness / mutFitness
    if mutFitness == 0.0
        fixp = 0.0
    else
        num = 1 - (fitnessRatio)^2
        den = 1 - (fitnessRatio)^(2*N)
        if fitnessRatio == 1.0 
            fixp = 1 / (2*N)
        else
            fixp = num/den
        end
    end
    return fixp, resFitness, mutFitness
end

function simulate(parameters::simParams)

    ## Unpacking the parameters object
    N = parameters.N
    T = parameters.T
    reps = parameters.reps
    activationFunction = parameters.activationFunction
    activationScale = parameters.activationScale
    K = parameters.K
    polyDegree = parameters.polyDegree 
    netDepth = parameters.netDepth
    netWidth = parameters.netWidth
    μ_size = parameters.μ_size
    netSaveStep = 1000

    ## Generates a random network, then mutates it
    fitnessHistories = fill(0.0, Int(T*reps/netSaveStep))
    initialNetworks = [generateFilledNetwork(netDepth, netWidth, 0.0) for _ in 1:reps]
    finalNetworks = [generateFilledNetwork(netDepth, netWidth, 0.0) for _ in 1:reps]
    replicateIDs = fill("", Int(T*reps/netSaveStep))
    timesteps = fill(0, Int(T*reps/netSaveStep))
    for r in 1:reps
        
        resNet = generateNetwork(netDepth, netWidth) ## Initial resident network
        replicateID = randstring(25) ## generates a long ID to uniquely identify replicates
        copy!(initialNetworks[r], resNet) ## saving this for later :)
        mutNet = copy(resNet)

        ## Main timestep loop
        for t in 1:T

            copy!(mutNet, resNet)
            mutateNetwork!(μ_size, mutNet)

            invasionProb, resFitness, mutFitness = invasionProbability(activationFunction, activationScale, K, polyDegree, N, resNet, mutNet)
            if rand() <= invasionProb
                copy!(resNet, mutNet)
                if mod(t, netSaveStep) == 0
                    fitnessHistories[Int(r*t/netSaveStep)] = mutFitness
                end
            else
                if mod(t, netSaveStep) == 0
                    fitnessHistories[Int(r*t/netSaveStep)] = resFitness
                end
            end
            if mod(t, netSaveStep) == 0
                replicateIDs[Int(r*t/netSaveStep)] = replicateID
                timesteps[Int(r*t/netSaveStep)] = t
            end

        end

        copy!(finalNetworks[r], resNet)
    end
    OutputDict = Dict([ ("T", timesteps),
                        ("N", fill(N, Int(T*reps/netSaveStep))),
                        ("activationFunction", fill(String(Symbol(activationFunction)), Int(T*reps/netSaveStep))),
                        ("K", fill(K, Int(T*reps/netSaveStep))),
                        ("envChallenge", fill(polyDegree, Int(T*reps/netSaveStep))),
                        ("netDepth", fill(netDepth, Int(T*reps/netSaveStep))),
                        ("netWidth", fill(netWidth, Int(T*reps/netSaveStep))),
                        ("μ_size", fill(μ_size, Int(T*reps/netSaveStep))),
                        ("fitness", fitnessHistories),
                        ("initialNetworks", initialNetworks),
                        ("finalNetworks", finalNetworks),
                        ("replicateID", replicateIDs)])
    return OutputDict
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