## this file calls simulations of various network sizes to compare how this influences adaptation
## Network parameters explored are both the total number of nodes, as well as the distribution of these nodes (width vs depth)

include("networksInvasionProbability.jl") ## Could probably use the Wright-Fisher version if you wanted, but that would be much slower

## testing how the size of the network influences the final evolved fitness
## only varying network depth (number of layers) and keeping the number of nodes per layer the same
## This will be a long function since there will need to be specific code for handling the sim outputs and comparing them
function compareNetworkSize(maxNetSize = 10, N = 10, T = 10, reps = 1, activationFunction = (f(x) = (1-exp(-x^2))), activationScale = 1.0, K = 5.0, polyDegree = 1,  μ_size = .1)
    netWidth = 1
    fitnessHistories = Array{Vector}(undef, maxNetSize)## Only saving the fitness history to save memory, should be able to retrieve networks at a later date if needed
    ## The structure of the outputs will be a three element vector of vectors
    ## We only care about the first vector, which is r (# of replicates) different timeseries 
    ## showing the evolutionary history of that parameter set 
    for i in 1:maxNetSize
        simResults = simulate(N, T, reps, activationFunction, activationScale, K, polyDegree, netDepth, netWidth, μ_size)
        fitnessHistories[i] = simResults[1]
    end

    ## Generating time series of mean fitness at each timestep
    ## Each index in this meanFitnessHistories represents a different iterated net size
    meanFitnessHistories = Array{Vector}(undef, maxNetSize)
    for i in 1:maxNetSize
        meanFitnessHistories[i] = Array{Float64}(undef, T)
        for t in 1:T
            x = 0
            for r in 1:reps
                x += fitnessHistories[i][r][t]
            end
            meanFitnessHistories[i][t] = x/reps
        end
        
    end
    plot(meanFitnessHistories)
end

maxNetSize = 5
N = 10000
T = 10000
reps = 5
compareNetworkSize(maxNetSize, N, T, reps)