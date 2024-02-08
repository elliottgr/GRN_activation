## Showing that modularity is related to regulation depth
using Graphs, SimpleWeightedGraphs, Plots, DataFrames
include("Networks.jl")


## simpler simulate loop to generate tabular data for modularity
function simulateForModularity(parameters)
    totalTimesteps = Int(parameters.T*parameters.reps/parameters.SaveStep)

    ## Generates a random network, then mutates it
    fitnessHistories = fill(0.0, totalTimesteps)
    Networks = [generateFilledNetwork(parameters.netDepth, parameters.netWidth, 0.0) for _ in 1:totalTimesteps]
    # finalTimesteps = [parameters.T for _ in 1:parameters.reps]'
    timesteps = fill(0, totalTimesteps)
    regulationDepth = fill(parameters.regulationDepth, totalTimesteps)
    netDepth = fill(parameters.netDepth, totalTimesteps)
    netWidth = fill(parameters.netWidth, totalTimesteps)
    for r in 1:parameters.reps
        
        resNet = generateNetwork(parameters.netDepth, parameters.netWidth) ## Initial resident network
        ## saving this for later :)
        mutNet = copy(resNet)
        edgeSet = fill((1,1,1,1), parameters.pleiotropy)
        ## Main timestep loop
        for t in 1:parameters.T

            copy!(mutNet, resNet)
            mutateNetwork!(parameters, mutNet, edgeSet)
            invasionProb, resFitness, mutFitness = invasionProbability(parameters, resNet, mutNet)

            ## Testing Invasion
            if rand() <= invasionProb
                copy!(resNet, mutNet)
                resFitness = copy(mutFitness)
            end

            if mod(t, parameters.SaveStep) == 0
                
                i = Int((((r-1)*parameters.T) + t)/parameters.SaveStep)
                copy!(Networks[i], resNet) 
                fitnessHistories[i] = copy(resFitness)

                timesteps[i] = copy(t)
            end

        end

    end
    OutputDict = Dict([ ("fitnessHistories", fitnessHistories),
                        ("timesteps", timesteps),
                        ("Networks", Networks),
                        ("regulationDepth", regulationDepth),
                        ("netWidth", netWidth),
                        ("netDepth", netDepth)
                        ])
    return OutputDict
end

function generateGraph(Data::Tuple)
    network, netDepth, netWidth, regulationDepth = Data
    return generateGraph(network, netDepth, netWidth, regulationDepth)
end

function generateGraph(network, netDepth, netWidth, regulationDepth)
    netDepth, netWidth = netDepth, netWidth
    g = SimpleWeightedDiGraph()
    NodeDict = Dict()
    nodeID = 1
    ## generating all vertices
    for i in 1:netDepth
        for j in 1:netWidth
            NodeDict[(i,j)] = nodeID
            nodeID += 1
            add_vertex!(g)
        end
    end

    ## generating all edges
    for i in 2:netDepth
        for j in 1:netWidth
            for k in maximum([1, i-regulationDepth]):i-1
                for l in 1:netWidth
                    add_edge!(g, NodeDict[k,l], NodeDict[i,j], network.Wm[i, j][k, l])
                end
            end
        end
    end

    return g
end


function main()
    N = 1000
    T = 10
    netDepth, netWidth = (5, 3)
    regulationDepth = 5
    saveStep = 10
    reps = 2
    Logistic(x, α = 1.0, β = 0.0, γ = 1.0) = γ/(1+exp(-α * (x - β)))
    α, β, γ = (1.0, 0.0, 1.0)
    activationScale = 1.0
    K = 5.0
    polyDegree = 3
    μ_size = 0.1
    pleiotropy = 1

    parameterSets = [simParams( N,T,saveStep,reps,Logistic,α,β,γ,activationScale,K,
            polyDegree,netDepth,netWidth,regulationDepth,μ_size,pleiotropy    ) for regulationDepth in 1:3]
    rawSimulationDataSets = map(simulateForModularity, parameterSets)
    SimulationData = vcat(DataFrame.(rawSimulationDataSets)...)
    selection = SimulationData[!, [:Networks, :netDepth, :netWidth, :regulationDepth]]
    # return selection
    NetworkGraphs = map(generateGraph,  Tuple.(eachrow(SimulationData[!, [:Networks, :netDepth, :netWidth, :regulationDepth]])))
    NetworkLabels = getfield.(map(label_propagation, NetworkGraphs), 1)
    NetworkModularities = [modularity(network, labels) for (network, labels) in zip(NetworkGraphs, NetworkLabels)]
    SimulationData[!, :Modularity] = NetworkModularities

    
    return SimulationData

end


