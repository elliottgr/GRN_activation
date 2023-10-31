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

## Necessary to call the distributed workers in higher level scripts, kinda pointless for single threaded versions (will still work tho)
struct simParams
    N::Int
    T::Int
    SaveStep::Int
    reps::Int
    activationFunction::Function
    α::Float64
    β::Float64
    γ::Float64
    activationScale::Float64
    K::Float64
    polyDegree::Int
    netDepth::Int
    netWidth::Int
    μ_size::Float64
end

## calcOj calculates the value of a single node j in layer i
function calcOj(parameters::simParams, j::Int, i::Int, activationMatrix, Wm, Wb)
    netWidth = size(Wb)[2]
    x = 0 ## activation
    for k in 1:(i-1)
        for l in 1:netWidth
            ## Wm[i, j][k, l] is the network weight
            x += (Wm[i, j][k, l] * activationMatrix[k, l])
        end
    end
    x += Wb[i, j]
    return(parameters.activationFunction(parameters.activationScale * x, parameters.α, parameters.β, parameters.γ)) 
end


## Define g(i), the network response to some input
## is the gradient iterator, g(i) = -1 + (2 * i/100)
## Here I'm just using it as an input range 
## N(g(i)) is just the network evaluation at some g(i) 
## Still calling the function iterateNetwork! for clarity, originally taken from socialGRN project

function iterateNetwork!(parameters::simParams, input, network::Network, activationMatrix)

    ## Calculates the total output of the network,
    ## iterating over calcOj() for each node
    
    Wm, Wb = network.Wm, network.Wb
    activationMatrix[1, :] = input ## whatever vector gets passed as the initial response
    for i in 2:parameters.netDepth ## Iterating over each layer
        for j in 1:parameters.netWidth ## iterating over each node
            activationMatrix[i, j] = calcOj(parameters, j, i, activationMatrix, Wm, Wb)
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
function measureNetwork(parameters, network::Network, envRange)
    x = 0
    input = fill(0.0, parameters.netWidth)
    for i in envRange
        activationMatrix = zeros(parameters.netDepth, parameters.netWidth) ## size of the bias vector
        input[1] = i
        iterateNetwork!(parameters, input, network, activationMatrix)
        N_i = activationMatrix[parameters.netDepth, parameters.netWidth]
        R_i = PlNormalized(i, parameters.polyDegree, 0, 1)
        x += (N_i - R_i) ^2
    end
    return x
end

function fitness(parameters, network)
    envRange = -1:0.02:1
    Var_F = var([PlNormalized(i, parameters.polyDegree, 0, 1) for i in envRange])
    return exp((-parameters.K * (measureNetwork(parameters, network, envRange))) / (length(envRange)*Var_F))
end

function mutateNetwork(parameters, network::Network)

    ## samples a random weight and shifts it
    ## Le Nagard method

    ## Need to allocate a new array and fill it 
    ## copy() of the original network doesn't allocate new elements
    ## so the output network overwrites the old one
    newNetwork = generateFilledNetwork(parameters.netDepth, parameters.netWidth, 0.0)
    copy!(newNetwork, network)

    mutationSize = randn()*parameters.μ_size

    ## Randomly selecting a weight (i,j,k,l) to mutate
    ## A large portion of them are "silent mutations"
    ## because only a fraction of the mutants will change network outputs
    ## k = 0 will simply be the bias vectors
    while newNetwork == network
        i = sample(1:parameters.netDepth)
        j = sample(1:parameters.netWidth)
        k = sample(0:i-1)
        l = sample(1:parameters.netWidth)

        if k > 0
            newNetwork.Wm[i, j][k, l] += mutationSize
        else
            newNetwork.Wb[i, j] += mutationSize
        end
    end
    return newNetwork
end

function mutateNetwork!(parameters, network::Network)

    ## samples a random weight and shifts it
    ## Le Nagard method

    mutationSize = randn()*parameters.μ_size

    i = sample(1:parameters.netDepth)
    j = sample(1:parameters.netWidth)
    k = sample(0:i-1)
    l = sample(1:parameters.netWidth)

    if k > 0
        network.Wm[i, j][k, l] += mutationSize
    else
        network.Wb[i, j] += mutationSize
    end
end

## Equivalent to Eq. 5, P(f_0 -> f_i), in le Nagard (2011)
function invasionProbability(parameters::simParams, resNet::Network, mutNet::Network)

    resFitness = fitness(parameters, resNet)
    mutFitness = fitness(parameters, mutNet)
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
    totalTimesteps = Int(parameters.T*parameters.reps/parameters.SaveStep)

    ## Generates a random network, then mutates it
    fitnessHistories = fill(0.0, totalTimesteps)
    initialNetworks = [generateFilledNetwork(parameters.netDepth, parameters.netWidth, 0.0) for _ in 1:parameters.reps]
    finalNetworks = [generateFilledNetwork(parameters.netDepth, parameters.netWidth, 0.0) for _ in 1:parameters.reps]
    replicateIDs = fill("", totalTimesteps)
    timesteps = fill(0, totalTimesteps)
    for r in 1:parameters.reps
        
        resNet = generateNetwork(parameters.netDepth, parameters.netWidth) ## Initial resident network
        replicateID = randstring(25) ## generates a long ID to uniquely identify replicates
        copy!(initialNetworks[r], resNet) ## saving this for later :)
        mutNet = copy(resNet)

        ## Main timestep loop
        for t in 1:T

            copy!(mutNet, resNet)
            mutateNetwork!(parameters, mutNet)
            invasionProb, resFitness, mutFitness = invasionProbability(parameters, resNet, mutNet)
            if rand() <= invasionProb
                copy!(resNet, mutNet)
                if mod(t, parameters.SaveStep) == 0
                    fitnessHistories[Int(r*t/parameters.SaveStep)] = mutFitness
                end
            else
                if mod(t, parameters.SaveStep) == 0
                    fitnessHistories[Int(r*t/parameters.SaveStep)] = resFitness
                end
            end
            if mod(t, parameters.SaveStep) == 0
                replicateIDs[Int(r*t/parameters.SaveStep)] = replicateID
                timesteps[Int(r*t/parameters.SaveStep)] = t
            end

        end

        copy!(finalNetworks[r], resNet)
    end
    OutputDict = Dict([ ("T", timesteps),
                        ("N", fill(N, totalTimesteps)),
                        ("activationFunction", fill(String(Symbol(parameters.activationFunction)), totalTimesteps)),
                        ("activationScale", fill(parameters.activationScale, totalTimesteps)),
                        ("α", fill(parameters.α, totalTimesteps)),
                        ("β", fill(parameters.β, totalTimesteps)),
                        ("γ", fill(parameters.γ, totalTimesteps)),
                        ("K", fill(parameters.K, totalTimesteps)),
                        ("envChallenge", fill(parameters.polyDegree, totalTimesteps)),
                        ("netDepth", fill(parameters.netDepth, totalTimesteps)),
                        ("netWidth", fill(parameters.netWidth, totalTimesteps)),
                        ("μ_size", fill(parameters.μ_size, totalTimesteps)),
                        ("fitness", fitnessHistories),
                        ("initialNetworks", initialNetworks),
                        ("finalNetworks", finalNetworks),
                        ("replicateID", replicateIDs)])
    return OutputDict
end