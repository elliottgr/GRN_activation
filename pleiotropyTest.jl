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

function newMutateNetwork(parameters, network, edge_set)

    ## can use StatsBase to sample without replacement

    ## creating a list of all possible mutations


    pleiotropy = 2
    # edgeSamples = sample(1:length(edge_set), pleiotropy, replace = false)

    for edge in edge_set[edgeSamples]
        if edge[2][1] != 0
            network.Wm[edge[1][1], edge[1][2]][edge[2][1], edge[2][2]] += randn()*parameters.μ_size
        else
            network.Wb[edge[1][1], edge[1][2]] += randn()*parameters.μ_size
        end
    end
    # for edge_i in edgeSamples
    #     if edge_set[edge_i][2][1] == 0
    #         network.Wb[edge_set[edge_i][1]...] += randn()*parameters.μ_size
    #     else
    #         network.Wm[edge_set[edge_i][1]...][edge_set[edge_i][2]...] += randn()*parameters.μ_size
    #     end
    # end
end

testMutations = 10^6
@time ([newMutateNetwork(parameters,network, edge_set) for i in 1:testMutations])
@time ([mutateNetwork!(parameters,network) for i in 1:testMutations])
# @time 5+5