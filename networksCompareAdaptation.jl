## this file calls simulations of various network sizes to compare how this influences adaptation
## Network parameters explored are both the total number of nodes, as well as the distribution of these nodes (width vs depth)
using StatsPlots ## For violin plots
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
        simResults = simulate(N, T, reps, activationFunction, activationScale, K, polyDegree, i, netWidth, μ_size)
        fitnessHistories[i] = simResults[1]
    end
    return fitnessHistories
end

function calculateMeanFitnessHistories(fitnessHistories)
    ## Generating time series of mean fitness at each timestep
    ## Each index in this meanFitnessHistories represents a different iterated net size
    meanFitnessHistories = Array{Vector}(undef, length(fitnessHistories)) 
    reps = length(fitnessHistories[1])
    T = length(fitnessHistories[1][1])
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
    return meanFitnessHistories
end

function fitnessHistoryTimeSeries(meanFitnessHistories)
    ## comparison of time series
    labels = permutedims([string("NetSize ", x) for x in 1:length(fitnessHistories)][:,:])
    plt = plot()
    plot!(plt, meanFitnessHistories, label = labels) 
    return plt
end

function fitnessHistoryViolinPlot(fitnessHistories)
    ## selecting the final timestep of each replicates
    finalFitnesses = Array{Vector}(undef, length(fitnessHistories)) 
    for i in 1:length(fitnessHistories)
        replicateFitnesses = Array{Float64}(undef, length(fitnessHistories[1]))
        for r in 1:length(fitnessHistories[1])
            replicateFitnesses[r] = last(fitnessHistories[i][r])
        end
        finalFitnesses[i] = replicateFitnesses
    end
    plt = plot()
    labels = permutedims([string("NetSize ", x) for x in 1:length(fitnessHistories)][:,:])
    plt = violin(2:length(finalFitnesses), finalFitnesses[2:length(finalFitnesses)], label = labels)
end

maxNetSize = 5
N = 1000
T = 100
reps = 10
fitnessHistories = compareNetworkSize(maxNetSize, N, T, reps)
meanFitnessHistories = calculateMeanFitnessHistories(fitnessHistories)
fitnessHistoryTimeSeries(meanFitnessHistories)
fitnessHistoryViolinPlot(fitnessHistories)