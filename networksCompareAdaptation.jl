## this file calls simulations of various network sizes to compare how this influences adaptation
## Network parameters explored are both the total number of nodes, as well as the distribution of these nodes (width vs depth)
using Distributed

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

function generateSimulations(maxNetSize = 30, maxNetWidth = 30, netSizeStep = 5, N = 1000, T = 1000, reps = 10)
    # dateString = string("GRN_Adaptation_Comparisons_",Dates.now(), ".jld2")

    ##Global Parameters for all simulations

    activationFunction = (f(x) = (1-exp(-x^2)))
    activationScale = 1.0
    K = 5.0
    # envChallenges = [3, 9, 27] ## Vector of each polynomial degree to check
    envChallenges = [3]
    μ_size = .1
    simulationOutputs = Dict() ## Dictionary where the keys are parameters (environmental challenge)
    print("Beginning simulations with \n 
            maxNetSize : $maxNetSize \n 
            N : $N (Population size) \n 
            T : $T (Number of timesteps) \n 
            reps : $reps (number of replicates) \n")
            ## nproc : $(nprocs()) (number of processes) \n"

    for polyDegree in envChallenges
        print("Now testing Legendre Polynomials of degree $polyDegree \n")

        ## Rewritten to use multi-processing
        
        ## testing how the size of the network influences the final evolved fitness
        ## only varying network depth (number of layers) and keeping the number of nodes per layer the same
        networkDepthComparisons = []

        ## generates fitness histories for all networks of a given size
        ## only tests networks that have the same number of total nodes, but with different depths / widths
        networkWidthComparisons = []

        for i in 1:netSizeStep:maxNetSize
            push!(networkDepthComparisons, simParams(N, T, reps, activationFunction, activationScale, K, polyDegree, i, 1, μ_size))
        end

        for width in 1:maxNetWidth
            if mod(maxNetSize, width) == 0 ## only iterating with valid network sizes
                push!(networkWidthComparisons, simParams(N, T, reps, activationFunction, activationScale, K, polyDegree, Int(maxNetSize/width), width, μ_size))
            end
        end
        
        # out = 

        ## Parallel processing
        outputDepthComparisons = pmap(simulate, networkDepthComparisons)
        outputWidthComparisons = pmap(simulate, networkWidthComparisons)
        simulationOutputs[polyDegree] = [outputDepthComparisons, outputWidthComparisons]
    end
    # jldsave(dateString; simulationOutputs)
    return simulationOutputs
    
end

maxNetSize = 10
maxNetWidth = 10
N = 1000
T = 10000
reps = 2

## Comparing different parameters for multi-processing

## 1 Process
nprocs()
using StatsPlots, JLD2, Dates ## For violin plots
include("networksInvasionProbability.jl") ## Could probably use the Wright-Fisher version if you wanted, but that would be much slower

@time simulationOutputs = generateSimulations(maxNetSize, maxNetWidth, 1, N, T, reps)
## around 90 seconds

addprocs(15)
nprocs()
@everywhere using StatsPlots, JLD2, Dates ## For violin plots
@everywhere include("networksInvasionProbability.jl") 

@time simulationOutputs = generateSimulations(maxNetSize, maxNetWidth, 1, N, T, reps)