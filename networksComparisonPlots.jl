## This file loads the JLD2 file generated from the networksCompareAdaptation.jl file
## and then runs them through some plots :)

using JLD2, StatsPlots, DataFrames
include("networksInvasionProbability.jl")


## Can access the files in the following way:
## simulationData[fileNumber]["simulationOutputs"][envChallenge][experimentType][networkSize][timeseries/invProb/networks]

function loadSimulationResults(path = pwd())
    simulationData = []
    for file in readdir(path)
        if splitext(file)[2] == ".jld2"
            push!(simulationData, JLD2.load(file))
        end
    end
    return simulationData
end


## generating a tabular dataset of the files
function generateTabular(simulationResults)
    T = []
    envChallenge = []
    netDepths = []
    netWidths = []
    fitnesses = []
    experimentType = [] ## will update this to some sort of "ExperimentID in the future. For now, 1 = variation in depth, 2 = variation in width
    replicateID = [] ## assigns a unique ID to each replicate
    repID = 1
    ## lot of nested loops :(
    for envChal in keys(simulationResults)
        for expType in [1, 2]
            for networkSizeIndex in 1:length(simulationResults[envChal][expType])
                for replicate in 1:length(simulationResults[envChal][expType][networkSizeIndex])
                    netDepth, netWidth = size(simulationResults[envChal][expType][networkSizeIndex][3][replicate])
                    for t in 1:length(simulationResults[envChal][expType][networkSizeIndex][1][replicate])
                        w = simulationResults[envChal][expType][networkSizeIndex][1][replicate][t]
                        push!(T, t)
                        push!(envChallenge, envChal)
                        push!(netDepths, netDepth)
                        push!(netWidths, netWidth)
                        push!(fitnesses, w)
                        push!(experimentType, expType)
                        push!(replicateID, repID)
                        
                    end
                    repID += 1
                end
            end
        end
    end
    return DataFrame(T = T, 
    envChallenge = envChallenge, 
    netDepth = netDepths, 
    netWidth = netWidths,
    fitness = fitnesses,
    experimentType = experimentType,
    replicateID = replicateID)
end

simulationResults = loadSimulationResults()[1]["simulationOutputs"]
df = generateTabular(simulationResults)

## plotting by depth
@df df[(df.netWidth .== 1) .& (df.T .== maximum(df.T)) .& (df.netDepth .<= 5), :] boxplot(string.(tuple.(:envChallenge, :netDepth)), :fitness, group=(:envChallenge, :netDepth))

## plotting width and depth
@df df[(df.netWidth .> 1) .& (df.T .== maximum(df.T)) .& (df.netDepth .<= 10) .& ((df.netDepth .>= 3)), :] boxplot(string.(tuple.(:envChallenge, :netDepth)), :fitness, group=(:envChallenge, :netDepth))