## This file loads the JLD2 file generated from the networksCompareAdaptation.jl file
## and then runs them through some plots :)

using JLD2, StatsPlots, DataFrames
include("Networks.jl")

function loadSimulationResults(path = pwd())
    simulationData = []
    for file in readdir(path)
        if splitext(file)[2] == ".jld2"
            push!(simulationData, JLD2.load(file))
        end
    end
    return simulationData
end

simulationResults = last(loadSimulationResults())["simulationOutputs"]
df = DataFrame(T = simulationResults["T"],
               N = simulationResults["N"],
               activationFunction = simulationResults["activationFunction"] ,
               K = simulationResults["K"],
               envChallenge = simulationResults["envChallenge"],
               netDepth = simulationResults["netDepth"],
               netWidth = simulationResults["netWidth"],
               μ_size = simulationResults["μ_size"],
               fitness = simulationResults["fitness"],
               replicateID = simulationResults["replicateID"]
)


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