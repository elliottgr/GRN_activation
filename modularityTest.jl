## Showing that modularity is related to regulation depth
using Graphs, SimpleWeightedGraphs
include("Networks.jl")

function generateParams(netDepth, netWidth, saveStep, polyDegree)
    envRange = -1:0.01:1
    LeNagardExp(x, α=sqrt(1/2), β=0.0, γ = 1.0) = 1 - (γ * exp(-((x-β)^2/(2*(α^2)))))

    regulationDepth = 3
    μ_size = 0.1
    return simParams(100, 10000, saveStep, 100, LeNagardExp, sqrt(1/2), 0.0, 1.0, 1.0, 5.0, polyDegree, netDepth, netWidth, regulationDepth, μ_size, 1)
end

netDepth, netWidth = (5, 3)
parameters = generateParams(netDepth, netWidth, 100, 3)
network = generateFilledNetwork(netDepth, netWidth, 1.0)


## 
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
    print(vertices(g))
    ## generating partition vector (expected number of edges per node)
    partitions = fill(1, netWidth) ## creating

    ## generating all edges
    for i in 2:netDepth
        for j in 1:netWidth
            edges = 0
            for k in maximum([1, i-parameters.regulationDepth]):i-1
                for l in 1:netWidth
                    add_edge!(g, NodeDict[k,l], NodeDict[i,j], network.Wm[i, j][k, l])
                    edges += 1
                end
            end
            push!(partitions, edges)
        end
    end

    return g, partitions
end

g, partitions = generateGraph(network, parameters)
modularity(g, partitions)