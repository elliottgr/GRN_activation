include("Networks.jl")

function generateParams(netDepth, netWidth, saveStep, polyDegree)
    envRange = -1:0.01:1
    LeNagardExp(x, α=sqrt(1/2), β=0.0, γ = 1.0) = 1 - (γ * exp(-((x-β)^2/(2*(α^2)))))

    regulationDepth = 100
    μ_size = 0.1
    return simParams(100, 10000, saveStep, 100, LeNagardExp, sqrt(1/2), 0.0, 1.0, 1.0, 5.0, polyDegree, netDepth, netWidth, regulationDepth, μ_size)
end

netDepth, netWidth = (3, 3)
parameters = generateParams(netDepth, netWidth, 100, 3)
network = generateFilledNetwork(netDepth, netWidth, 0.0)
testNet = copy(network)
for i in 1:100
    mutateNetwork!(parameters, testNet)
end 

edge_set = []
for i in 1:parameters.netDepth
    for j in 1:parameters.netWidth
        for k in 1:i-1
            for l in 1:parameters.netWidth
                push!(edge_set, [[i,j],[k,l]])
            end
        end
        push!(edge_set, [[i,j], [0,0]])
    end
end
function generateEdge(parameters) 
    i = sample(1:parameters.netDepth)
    j = sample(1:parameters.netWidth)
    k = sample(0:i-1)
    l = sample(1:parameters.netWidth)
    return (i, j, k, l)
end
function mutateNetwork!(parameters, network::Network, edgeSet)

    ## samples a random weight and shifts it
    ## Le Nagard method
    mutationSize = randn()*parameters.μ_size
    for edge in 1:length(edgeSet)
        testEdge = generateEdge(parameters)
        ## testing if first edge generated is a unique edge
        ## this is a balance for performance, since manually
        ## generating only unique edge samples ended up having sifnificant performance costs
        if testEdge ∈ edgeSet
            testEdge = generateEdge(parameters)
        else 
            edgeSet[edge] = testEdge
        end
    end
    for edge in edgeSet
        i, j, k, l = edge
        if k > 0
            network.Wm[i, j][k, l] += mutationSize
        else
            network.Wb[i, j] += mutationSize
        end
    end
    return network
end

function newMutateNetwork(parameters, network, edge_set)

    ## can use StatsBase to sample without replacement

    ## creating a list of all possible mutations
    pleiotropy = 1
    edgeSamples = sample(1:length(edge_set), pleiotropy, replace = false)

    for edge in edge_set[edgeSamples]
        if edge[2][1] != 0
            network.Wm[edge[1][1], edge[1][2]][edge[2][1], edge[2][2]] += randn()*parameters.μ_size
        else
            network.Wb[edge[1][1], edge[1][2]] += randn()*parameters.μ_size
        end
    end

    return network
end

testMutations = 10^6
pleiotropy = 1
edgeSet = fill((1,1,1,1), pleiotropy)
@time ([newMutateNetwork(parameters,network, edge_set) for i in 1:testMutations])
@time ([oldMutateNetwork!(parameters,network, edgeSet) for i in 1:testMutations][1])
@time ([mutateNetwork!(parameters,network) for i in 1:testMutations])
# @time 5+5