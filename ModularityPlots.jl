
using JLD2, DataFrames, Plots, StatsPlots
include("Networks.jl")
include("NetworkGraphFunctions.jl")
## Directly loading the file
filepath = "/better_scratch/elliott/GRN_activation/ModularityTests2024-02-08T18:19:44.252.jld2"
df = JLD2.load(filepath)["SimulationData"] ## Files from "CompareAdaptations" have a different access key, should prevent accidental data mixing :)
df.Graph = generateGraph.(Tuple.(eachrow(df[!, [:Networks, :netDepth, :netWidth, :regulationDepth]])))
df.GlobalClusteringCoefficient = global_clustering_coefficient.(df.Graph)
## Comparison of regulation depth and modularity
selection = (df.timesteps .== maximum(df.timesteps)) .& (df.netWidth .>= 1) 
# .& (df.Modularity .<= 1)


scatter(df[selection, :].regulationDepth, df[selection, :].GlobalClusteringCoefficient)
scatter(df[selection, :].regulationDepth, df[selection, :].Modularity)
## selecting correct datapoints
@df df[selection, :] boxplot(:regulationDepth, :Modularity)