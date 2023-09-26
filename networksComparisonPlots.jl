## This file loads the JLD2 file generated from the networksCompareAdaptation.jl file
## and then runs them through some plots :)

using JLD2, StatsPlots, DataFrames
include("networksInvasionProbability.jl")
## generates fitness histories for all networks of a given size
## only tests networks that have the same number of total nodes, but with different depths / widths

## takes a vector of simulation results and extracts the individual fitness timeseries
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
    fitnessHistories =  [simulationResults[i][1] for i in 1:length(simulationResults)]
    ## comparison of time series
    labels = permutedims([string("NetSize ", x) for x in 1:length(fitnessHistories)][:,:])
    plt = plot()
    plot!(plt, calculateMeanFitnessHistories(fitnessHistories), label = labels) 
    return plt
end

function fitnessHistoryViolinPlot(simulationResults)
    ## selecting the final timestep of each replicates
    fitnessHistories = [simulationResults[i][1] for i in 1:length(simulationResults)]
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


function groupedBoxPlots(simulationResults, depthORwidth = 1)
    envChallenges = keys(simulationResults)
    fitnessDistributionsPerEnvChallenge = Dict()
    for envChal in envChallenges
        simResult = simulationResults[envChal]
        fitnessHistories = [simResult[depthORwidth][i][1] for i in 1:length(simResult)]
        finalFitnesses = Array{Vector}(undef, length(fitnessHistories)) 
        for i in 1:length(fitnessHistories)
            replicateFitnesses = Array{Float64}(undef, length(fitnessHistories[1]))
            for r in 1:length(fitnessHistories[1])
                replicateFitnesses[r] = last(fitnessHistories[i][r])
            end
            finalFitnesses[i] = replicateFitnesses
        end
        fitnessDistributionsPerEnvChallenge[envChal] = finalFitnesses
    end
    return fitnessDistributionsPerEnvChallenge
    # plt = plot()
    # labels = permutedims([string("NetSize ", size(simulationResults[x][3][1])) for x in 1:length(fitnessHistories)][:,:])
    # plt = violin(1:length(finalFitnesses), finalFitnesses[1:length(finalFitnesses)], label = labels)
end

## Can access the files in the following way:
## simulationData[fileNumber]["simulationOutputs"][envChallenge][width or depth comparison][networkSize][timeseries/invProb/networks]

function loadSimulationResults(path = pwd())
    simulationData = []
    for file in readdir(path)
        if splitext(file)[2] == ".jld2"
            # JLD2.load(file)
            push!(simulationData, JLD2.load(file))
        end
    end
    return simulationData
end

simulationResults = last(loadSimulationResults())["simulationOutputs"] ## load latest simulation
envChallenge = 3
depthORwidth = 1 #1 = depth, 2 = width
fitnessHistoryViolinPlot(simulationResults[envChallenge][depthORwidth])

groupedBoxPlots(simulationResults)

## generating a tabular dataset of the fitness time series

function generateTabular(simulationResults)
    T = []
    envChallenge = []
    netDepth = []
    netWidth = []
    fitness = []
    ## lot of nested loops :(
    for envChal in keys(simulationResults)
        for widthORdepth in [1, 2]
            for replicate in 1:length(simulationResults[envChal][widthORdepth])
                for t in simulationResults[envChal][widthORdepth][replicate][1]
                    
                end
            end
        end
    end
end