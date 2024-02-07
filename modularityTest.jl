## Showing that modularity is related to regulation depth
using Graphs, SimpleWeightedGraphs, Plots
include("Networks.jl")


## simpler simulate loop to generate tabular data for modularity
function simulateForModularity(parameters)
    ## Generates a random network, then mutates it
    finalFitnesses = fill(0.0, parameters.reps)
    initialNetworks = [generateFilledNetwork(parameters.netDepth, parameters.netWidth, 0.0) for _ in 1:parameters.reps]
    finalNetworks = [generateFilledNetwork(parameters.netDepth, parameters.netWidth, 0.0) for _ in 1:parameters.reps]
    finalTimesteps = [parameters.T for _ in 1:parameters.reps]

    for r in 1:parameters.reps
        
        resNet = generateNetwork(parameters.netDepth, parameters.netWidth) ## Initial resident network
        copy!(initialNetworks[r], resNet) ## saving this for later :)
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

            if t == parameters.T
                finalFitnesses[r] = copy(resFitness)
            end
        end

        copy!(finalNetworks[r], resNet)
    end
    OutputDict = Dict([ ("finalFitnesses", finalFitnesses),
                        ("finalTimesteps", finalTimesteps),
                        ("initialNetworks", initialNetworks),
                        ("finalNetworks", finalNetworks)
                        ])
    return OutputDict
end

function generateParams(activationFunction, netDepth, netWidth, saveStep, polyDegree, regulationDepth)
    envRange = -1:0.01:1
    μ_size = 0.1
    return simParams(100, 10000, saveStep, 100, activationFunction, sqrt(1/2), 0.0, 1.0, 1.0, 5.0, polyDegree, netDepth, netWidth, regulationDepth, μ_size, 1)
end


function generateOutNets(resNet, parameters, generations = 10000)
    outNets = []
    edgeSet = fill((1,1,1,1), parameters.pleiotropy)
    for i in 1:generations
        newNet = copy(resNet)
        mutateNetwork!(parameters, newNet, edgeSet)
        if fitness(parameters,newNet) >= fitness(parameters, resNet)
            resNet = newNet
        end
        if mod(i, parameters.SaveStep) == 0
            push!(outNets, copy(resNet))
        end
    end
    return outNets
end

function generateGraph(network, parameters)
    netDepth, netWidth = parameters.netDepth, parameters.netWidth
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
            for k in maximum([1, i-parameters.regulationDepth]):i-1
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
    T = 10000
    netDepth, netWidth = (5, 3)
    regulationDepth = 5
    saveStep = 1000
    reps = 10
    Logistic(x, α = 1.0, β = 0.0, γ = 1.0) = γ/(1+exp(-α * (x - β)))
    α, β, γ = (1.0, 0.0, 1.0)
    activationScale = 1.0
    K = 5.0
    polyDegree = 3
    μ_size = 0.1
    pleiotropy = 1
    parameters = simParams(
            N,
            T,
            saveStep,
            reps,
            Logistic,
            α,
            β,
            γ,
            activationScale,
            K,
            polyDegree,
            netDepth,
            netWidth,
            regulationDepth,
            μ_size,
            pleiotropy
        )
    SimulationData = simulateForModularity(parameters)
    InitialNetworkGraphs = [generateGraph(x, parameters) for x in SimulationData["initialNetworks"]]
    FinalNetworkGraphs = [generateGraph(x, parameters) for x in SimulationData["finalNetworks"]]
    InitialNetworkModularities = [modularity(g, label_propagation(g)[1]) for g in InitialNetworkGraphs]
    FinalNetworkModularities = [modularity(g, label_propagation(g)[1]) for g in FinalNetworkGraphs]

    OutputData = Dict([("InitialNetworkModularities", InitialNetworkModularities),
                        ("FinalNetworkModularities", FinalNetworkModularities),
                        ("finalFitnesses", SimulationData["finalFitnesses"]),
                        ("finalTimesteps", SimulationData["finalTimesteps"])])
    
    return OutputData

end

function generatePlot()
    OutputData = main()
    plt = plot()
    for rep in 1:length(OutputData["finalTimesteps"])
        plot!([1, OutputData["finalTimesteps"][rep]], [OutputData["InitialNetworkModularities"][rep], OutputData["FinalNetworkModularities"][rep]])
    end
    return plt
end