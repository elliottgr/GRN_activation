## this file calls simulations of various network sizes to compare how this influences adaptation
## Network parameters explored are both the total number of nodes, as well as the distribution of these nodes (width vs depth)
using StatsPlots, JLD2, Dates ## For violin plots
include("networksInvasionProbability.jl") ## Could probably use the Wright-Fisher version if you wanted, but that would be much slower

## testing how the size of the network influences the final evolved fitness
## only varying network depth (number of layers) and keeping the number of nodes per layer the same
## This will be a long function since there will need to be specific code for handling the sim outputs and comparing them
function compareNetworkDepth(maxNetSize = 10, N = 10, T = 10, reps = 1, activationFunction = (f(x) = (1-exp(-x^2))), activationScale = 1.0, K = 5.0, polyDegree = 1,  μ_size = .1)
    netWidth = 1
    netSizeStep = 5
    simulationResults = Array{Vector}(undef, maxNetSize)## Only saving the fitness history to save memory, should be able to retrieve networks at a later date if needed
    ## The structure of the outputs will be a three element vector of vectors
    ## We only care about the first vector, which is r (# of replicates) different timeseries 
    ## showing the evolutionary history of that parameter set 
    for i in 1:netSizeStep:maxNetSize
        simulationResults[i] = simulate(N, T, reps, activationFunction, activationScale, K, polyDegree, i, netWidth, μ_size)
    end
    return simulationResults
end

## generates fitness histories for all networks of a given size
## only tests networks that have the same number of total nodes, but with different depths / widths
function compareNetworkWidth(maxNetSize = 20, maxNetWidth = 10, N = 10, T = 10, reps = 1, activationFunction = (f(x) = (1-exp(-x^2))), activationScale = 1.0, K = 5.0, polyDegree = 1,  μ_size = .1)
    simulationResults = []
    for width in 1:maxNetWidth
        if mod(maxNetSize, width) == 0 ## only iterating with valid network sizes
            push!(simulationResults, simulate(N, T, reps, activationFunction, activationScale, K, polyDegree, Int(maxNetSize/width), width, μ_size))
        end
    end
    return simulationResults
end

## takes a vector of simulation results and extracts the individual fitness timeseries
generateFitnessHistories(simulationResults) = [simulationResults[i][1] for i in 1:length(simulationResults)]

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

function fitnessHistoryTimeSeries(simulationResults)
    fitnessHistories = generateFitnessHistories(simulationResults)
    ## comparison of time series
    labels = permutedims([string("NetSize ", x) for x in 1:length(fitnessHistories)][:,:])
    plt = plot()
    plot!(plt, calculateMeanFitnessHistories(fitnessHistories), label = labels) 
    return plt
end

function fitnessHistoryViolinPlot(simulationResults)
    ## selecting the final timestep of each replicates
    fitnessHistories = generateFitnessHistories(simulationResults)
    finalFitnesses = Array{Vector}(undef, length(fitnessHistories)) 
    for i in 1:length(fitnessHistories)
        replicateFitnesses = Array{Float64}(undef, length(fitnessHistories[1]))
        for r in 1:length(fitnessHistories[1])
            replicateFitnesses[r] = last(fitnessHistories[i][r])
        end
        finalFitnesses[i] = replicateFitnesses
    end
    plt = plot()
    labels = permutedims([string("NetSize ", size(simulationResults[x][3][1])) for x in 1:length(fitnessHistories)][:,:])
    plt = violin(1:length(finalFitnesses), finalFitnesses[1:length(finalFitnesses)], label = labels)
end

function generateSimulations(maxNetSize = 30, N = 1000, T = 1000, reps = 10)
    dateString = string("GRN_Adaptation_Comparisons_",Dates.now(), ".jld2")

    ##Global Parameters for all simulations

    activationFunction = (f(x) = (1-exp(-x^2)))
    activationScale = 1.0
    K = 5.0
    envChallenges = [3, 9, 27] ## Vector of each polynomial degree to check
    μ_size = .1
    simulationOutputs = Dict() ## Dictionary where the keys are parameters (environmental challenge)

    for polyDegree in envChallenges
        networkDepthComparisons = compareNetworkDepth(maxNetSize, N, T, reps, activationFunction, activationScale, K, polyDegree, μ_size)
        networkWidthComparisons = compareNetworkWidth(maxNetSize, maxNetSize, N, T, reps, activationFunction, activationScale, K, polyDegree, μ_size)
        simulationOutputs[polyDegree] = [networkDepthComparisons, networkWidthComparisons]
    end
    jldsave(dateString; simulationOutputs)
    return simulationOutputs
end

maxNetSize = 
maxNetWidth = 30
N = 1000
T = 500000
reps = 100
simulationOutputs = generateSimulations(maxNetSize, N, T, reps)
# simulationResults = compareNetworkSize(maxNetSize, N, T, reps)
simulationResults = generateSimulations(maxNetSize, N, T, reps)
meanFitnessHistories = calculateMeanFitnessHistories(fitnessHistories)
fitnessHistoryTimeSeries(simulationResults)
fitnessHistoryViolinPlot(simulationOutputs[3][2])
fitnessHistoryViolinPlot(simulationOutputs[9][2])
fitnessHistoryViolinPlot(simulationOutputs[27][2])