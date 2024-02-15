## This file loads the JLD2 file generated from the networksCompareAdaptation.jl file
## and then runs them through some plots :)

using JLD2, StatsPlots, DataFrames, LaTeXStrings
include("Networks.jl")

function loadSimulationResults(path = pwd())
    simulationData = []
    for file in readdir(path)
        if splitext(file)[2] == ".jld2"
            try
                simulationFile = JLD2.load(string(path, file))["simulationOutputs"]
                simulationFile.filename = fill(file, length(simulationFile.fitness))
                push!(simulationData, simulationFile)
                print("Sucessfully loaded: $file with length $(length(simulationFile.fitness)) entries of $(length(unique(simulationFile[simulationFile.T .== maximum(simulationFile.T), :].replicateID))) replicates \n")
            ## Detecting files that dont have the same structure
            catch KeyError
               print("Incompatible file found. $file has not been loaded into the dataset.") 
            end
        end
    end
    return vcat(simulationData..., cols = :union)
end

## Depth Boxplots Functions
ncolors(df, dfSelection) = permutedims(repeat(collect(1:size(unique(df[dfSelection, [:netDepth, :netWidth]]))[1]),length(unique(df.envChallenge)))[:,:])
colorScale(df, dfSelection) = [RGB(x,x,x) for x in 0.0:(1/size(unique(df[dfSelection, [:netDepth, :netWidth]]))[1]):1.0]
labels(df, dfSelection) = permutedims(cat([string("Depth: ", label[1], ", ", "Width: ", label[2])  for label in eachrow(unique(df[dfSelection, [:netDepth, :netWidth]]))], 
                                        fill("", (length(ncolors(df, dfSelection)) - size(unique(df[dfSelection, [:netDepth, :netWidth]]))[1])), 
                                    dims =(1))[:,:])



## Plots of all data
################

df = loadSimulationResults("/better_scratch/elliott/GRN_activation/")
# df = loadSimulationResults(string(pwd(), "/"))

minDepth, maxDepth = (1, 15)
minWidth, maxWidth = (1, 15)
minFitness, maxFitness = (0.15, 1.0)


function plotPleiotropy(df, T, maxDepth, activationFunction = "Logistic", filename = "")
    minDepth = 3
    filter = (df.μ_size .== 0.1) .& (df.netDepth .<= maxDepth) .& (df.regulationDepth .<= maxDepth) .& (df.netDepth .>= minDepth) .& (df.T .== T) .& (df.activationFunction .== activationFunction) .& .!ismissing.(df.pleiotropy)
    pleiodf = df[filter, :]
    plt = plot()
    for i in 3:6
        selection = pleiodf[(pleiodf.netDepth .== i) .& (pleiodf.netWidth .== 1) .& (pleiodf.envChallenge .== 1), :]
        plt = scatter!(selection.pleiotropy, selection.fitness, label = "Net Depth = $i")
    end
    return plt
end

function plotRegulationDepth(df, T, maxDepth, activationFunction = "Logistic", filename = "")
    minDepth = 3
    envChal = 3

    for i in 1:length(df.regulationDepth) ## bad code but w/e
        df.regulationDepth[i] = minimum([df.regulationDepth[i], df.netDepth[i]])
    end

    filter = (df.μ_size .== 0.1) .& (df.envChallenge .== envChal)  .& (df.netDepth .<= maxDepth) .& (df.regulationDepth .<= maxDepth) .& (df.netDepth .>= minDepth) .& (df.netWidth .== 1) .& (df.T .== T) .& (df.activationFunction .== activationFunction) 
    regDF = (groupby(df[filter, :], :regulationDepth), :fitness=>mean)
    outCols = []
    for i in eachindex(regDF[1])
        tempDF = combine(groupby(regDF[1][i], :netDepth), :fitness=>mean)
        tempDF[!, :regulationDepth] = fill(regDF[1][i].regulationDepth[1], length(tempDF.fitness_mean))
        push!(outCols, tempDF)
    end
    meanFitnessDF = vcat(outCols...)
    plt = plot()
    colors = [RGB(1-(((x-minDepth)/maxDepth)*.8), 1-(((x-minDepth)/maxDepth)*.8),1-(((x-minDepth)/maxDepth)*.8)) for x in minDepth:maxDepth]
    print(colors)
    i = 0
    for netDepth in [3, 4, 5, 6, 7, 8]
        Rstring = L"\(R\)"
        i+=1
        xs = meanFitnessDF[(meanFitnessDF.netDepth .== netDepth), :regulationDepth]
        ys = meanFitnessDF[(meanFitnessDF.netDepth .== netDepth), :fitness_mean]
        plot!(xs, ys, label = netDepth,
                title = "Comparison of regulation depth $Rstring and fitness", 
                xlabel = Rstring, 
                ylabel = "Mean Fitness",
                legendtitle = "Genes",
                legend = :bottomright,
                c = reverse(colors)[i])
    end
    return plt
end
plotRegulationDepth(df[(df.filename .== "RegulationDepth2023-11-16T13:34:38.043.jld2"), :], maximum(df.T), 9, "Logistic", "") 
plotRegulationDepth(df[(df.filename .== "RegulationDepth2023-11-16T13:34:38.043.jld2"), :], maximum(df.T), 9, "LeNagardExp")

## plotting final fitness by increasing activation function steepness
function plotActivationFunctionSteepness(df, envChallenge = 3, activationFunction = "Logistic")
    depthMin = 3
    depthMax = 12
    plt = plot(title = "Comparison of $activationFunction activation function \n steepness and network fitness", xlabel = "α", ylabel = "Fitness")
    colors = [RGB((x-depthMin)/depthMax, (x-depthMin)/depthMax, (x-depthMin)/depthMax) for x in depthMin:depthMax]
    for depth in depthMin:2:depthMax
        steepnessSelection = (df.μ_size .== 0.1) .& (df.activationFunction .== activationFunction) .& (df.netDepth .== depth) .& (df.envChallenge .== envChallenge) .& (df.T .== maximum(df.T)) .& (df.regulationDepth .>= depth) 
        α_df = sort!(combine(groupby(df[steepnessSelection, :], :α), :fitness=> mean), :α)
        print(α_df)
        plot!(plt, α_df.α, α_df.fitness_mean, label = depth, color = colors[depth-depthMin+1])
    end
    return plt
end

plotActivationFunctionSteepness(df, 3, "Logistic")

function plotMutationSize(df, activationFunction = "Logistic")
    maxDepth = 9
    filter =  (df.netDepth .<= maxDepth) .& (df.regulationDepth .<= maxDepth) .& (df.netDepth .>= minDepth) .& (df.netWidth .== 1) .& (df.T .== maximum(df.T)) .& (df.activationFunction .== activationFunction) 
    groupDF = (groupby(df[filter, :], :μ_size), :fitness=>mean)
    for netDepth in unique(groupDF.netDepth)
        i+=1
        xs = meanFitnessDF[(meanFitnessDF.netDepth .== netDepth), :regulationDepth]
        ys = meanFitnessDF[(meanFitnessDF.netDepth .== netDepth), :fitness_mean]
        plot!(xs, ys, label = netDepth,
                title = "Comparison of mutant effect size and fitness \n for activation function \"$activationFunction\"", 
                xlabel = "μ Std. Dev.", 
                ylabel = "Mean Fitness",
                legend = :bottomright)
    end
end

plotMutationSize(df[(df.filename .== "MutationSizeTests2023-11-16T10:56:55.668.jld2"), :])

## Box Plots

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

## plotting shortrange Boxplots
shortrangeSelection = (df.regulationDepth .>= 1) .& (df.netWidth .== 1) .& (df.T .== maximum(df.T))
@df df[shortrangeSelection, :] boxplot(string.(tuple.(:envChallenge, :netDepth)), :fitness, group=(:envChallenge, :netDepth), color = ncolors(df, shortrangeSelection), palette = colorScale(df, shortrangeSelection), label = labels(df, shortrangeSelection), title = "Fitness for networks of size ≤ $maxDepth \n and environmental challenges ≤ $(maximum(:envChallenge))")
