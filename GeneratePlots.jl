## This file loads the JLD2 file generated from the networksCompareAdaptation.jl file
## and then runs them through some plots :)

using JLD2, StatsPlots, DataFrames
include("Networks.jl")


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
    experimentType = [] ## will update this to some sort of "ExperimentID in the future. For now, 1 = variation in depth, 2 = variation in width, >2 is activation function comparisons
    replicateID = [] ## assigns a unique ID to each replicate
    repID = 1
    ## lot of nested loops :(
    for envChal in keys(simulationResults)
        for expType in 1:length(simulationResults[envChal])
            for networkSizeIndex in 1:length(simulationResults[envChal][expType])
                for replicate in 1:length(simulationResults[envChal][expType][networkSizeIndex][1])
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

simulationResults = loadSimulationResults()[3]["simulationOutputs"]
df = generateTabular(simulationResults)

## Plots of all data
################

minDepth, maxDepth = (1, 15)
minWidth, maxWidth = (1, 15)
minFitness, maxFitness = (0.15, 1.0)


ncolors(df, dfSelection) = permutedims(repeat(collect(1:size(unique(df[dfSelection, [:netDepth, :netWidth]]))[1]),length(unique(df.envChallenge)))[:,:])
colorScale(df, dfSelection) = [RGB(x,x,x) for x in 0.0:(1/size(unique(df[dfSelection, [:netDepth, :netWidth]]))[1]):1.0]
labels(df, dfSelection) = permutedims(cat([string("Depth: ", label[1], ", ", "Width: ", label[2])  for label in eachrow(unique(df[dfSelection, [:netDepth, :netWidth]]))], 
                                        fill("", (length(ncolors(df, dfSelection)) - size(unique(df[dfSelection, [:netDepth, :netWidth]]))[1])), 
                                    dims =(1))[:,:])

## plotting by depth
depthSelection = (df.netWidth .== 1) .& (df.T .== maximum(df.T)) .& (df.netDepth .<= maxDepth) .& (df.netDepth .>= minDepth)
@df df[depthSelection, :] boxplot(string.(tuple.(:envChallenge, :netDepth)), :fitness, group=(:envChallenge, :netDepth), color = ncolors(df, depthSelection), palette = colorScale(df, depthSelection), labels = labels(df, depthSelection), title = "Fitness for networks of size ≤ $maxDepth \n and environmental challenges ≤ $(maximum(:envChallenge))")
## plotting width and depth
widthSelection = (df.netWidth .> minWidth) .& (df.T .== maximum(df.T)) .& (df.netDepth .<= maxDepth) .& ((df.netDepth .>= minDepth))
@df df[widthSelection, :] boxplot(string.(tuple.(:envChallenge, :netDepth)), :fitness, color = ncolors(df, widthSelection), palette = colorScale(df, widthSelection), group=(:envChallenge, :netDepth), labels = labels(df, widthSelection), title = "Fitness for networks of depth ≤ $maxDepth & width ≤ $maxWidth) \n and environmental challenges ≤ $(maximum(:envChallenge))")

## Plots of Le Nagard filter (>0.15 fitness)
filteredDepthSelection = (df.fitness .>= minFitness) .& (df.netWidth .== 1) .& (df.T .== maximum(df.T)) .& (df.netDepth .<= maxDepth)
@df df[filteredDepthSelection, :] boxplot(string.(tuple.(:envChallenge, :netDepth)), :fitness, color = ncolors(df, filteredDepthSelection), group=(:envChallenge, :netDepth), legend = :none)
@df df[(df.fitness .>= minFitness) .& (df.netWidth .> minWidth) .& (df.T .== maximum(df.T)) .& (df.netDepth .<= maxDepth) .& ((df.netDepth .>= minDepth)), :] boxplot(string.(tuple.(:envChallenge, :netDepth)), :fitness, group=(:envChallenge, :netDepth), legend = :none)

## Plots comparing by Activation function
ActivationFunctionSelection = (df.T .== maximum(df.T) .& df.experimentType .>= 3)
@df df[ActivationFunctionSelection, :] boxplot(string.(tuple.(:envChallenge, :experimentType)), :fitness, color = ncolors(df, ActivationFunctionSelection), palette = colorScale(df, ActivationFunctionSelection), group=(:envChallenge, :experimentType), labels = labels(df, ActivationFunctionSelection), title = "")