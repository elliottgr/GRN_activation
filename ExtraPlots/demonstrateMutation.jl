## This file demonstrates some basic properties of network mutation and then makes a few plots :)

using Plots
include("../Networks.jl")


function generateParams(netDepth, netWidth)
    envRange = -1:0.01:1
    LeNagardExp(x, α=sqrt(1/2), β=0.0, γ = 1.0) = 1 - (γ * exp(-((x-β)^2/(2*(α^2)))))

    regulationDepth = 100
    μ_size = 0.1
    return simParams(100, 100, 100, 100, LeNagardExp, sqrt(1/2), 0.0, 1.0, 1.0, 5.0, 5, netDepth, netWidth, regulationDepth, μ_size)
end

function N_iHistogramData(parameters, Nmutants = 100, Mutations = 1, initNet = 0.0, inputVal = 0.5)
    ## Generating Histograms 
    parentNetworks = []
    if initNet == 0.0
        for _ in 1:Nmutants
            initNet = rand()
            push!(parentNetworks, generateFilledNetwork(parameters.netDepth, parameters.netWidth, initNet))
        end
    else
        for _ in 1:Nmutants
            push!(parentNetworks, generateFilledNetwork(parameters.netDepth, parameters.netWidth, initNet))
        end
    end
    mutants = []
    for net in parentNetworks
        newNet = copy(net)
        for _ in 1:Mutations
            newNet = mutateNetwork(parameters, newNet)
        end
        push!(mutants, newNet) 
    end

    outputs = []
    input = fill(inputVal, parameters.netWidth)

    initVals = []
    for network in parentNetworks

        input = fill(inputVal, parameters.netWidth)
        activationMatrix = zeros(parameters.netDepth, parameters.netWidth) ## size of the bias vector
        input[1] = inputVal
        iterateNetwork!(parameters, input, network, activationMatrix)
        N_i = activationMatrix[parameters.netDepth, parameters.netWidth]
        push!(initVals, N_i)
    end

    for network in mutants
        activationMatrix = zeros(parameters.netDepth, parameters.netWidth) ## size of the bias vector
        input = fill(inputVal, parameters.netWidth)
        iterateNetwork!(parameters, input, network, activationMatrix)
        N_i = activationMatrix[parameters.netDepth, parameters.netWidth]
        push!(outputs, N_i)
    end

    return outputs, initVals 
end

function generateHistograms(parameterSets, nMutants)
    plt = histogram()
    Ncolors = length(parameterSets)
    succesfulMutants = Dict()
    sdΔN = Dict()
    meanΔN = Dict()
    color_i = 0
    for parameters in parameterSets
        succesfulMutantCount = 0
        silentMutantCount = 0
        μ1, initVals = N_iHistogramData(parameters, nMutants, 1, 0.5, 0.5) 
        out = []
        for (μ, initVal) in zip(μ1, initVals)
            silentMutantCount += 1
            if μ != initVal
                succesfulMutantCount += 1
                push!(out, μ - initVal)
            end
        end
        color_i += 1
        succesfulMutants[parameters.netDepth] = succesfulMutantCount/silentMutantCount
        sdΔN[parameters.netDepth] = std(out)
        meanΔN[parameters.netDepth] = mean(out)
        plt = histogram!(out, normalize = :probability, ylim = (0.0, 0.05), title = "Distribution of mutant effects on \n gene activation of $nMutants mutations", ylabel = "Proportion of networks", xlabel = "ΔN(i)", label = "Network Depth: $(parameters.netDepth)", color = RGB(1-color_i/Ncolors,1-color_i/Ncolors,1-color_i/Ncolors), bins = 200)
    end
    return plt, succesfulMutants, sdΔN, meanΔN
end



parameterSets = [generateParams(2, 1), generateParams(4, 1), generateParams(6, 1)]
plt, counts, standDev, means= generateHistograms(parameterSets, 50000)
savefig(plt, "mutationEFfectHistogram.svg")
print(counts) ## prints the proportion of succesful vs silent mutations
print(standDev)
print(means)