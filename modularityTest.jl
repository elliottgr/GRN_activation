## Showing that modularity is related to regulation depth

using Distributed



@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
    Pkg.instantiate(); Pkg.precompile()
end


@everywhere begin
    using Graphs, SimpleWeightedGraphs, Plots, DataFrames, JLD2, Dates
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

    include("NetworkGraphFunctions.jl")

    function main(filestring)
        dateString = string(filestring, Dates.now(), ".jld2")
        N = 1000
        T = 10^6
        netDepth, netWidth = (5, 3)
        # regulationDepth = 5
        saveStep = 100000
        reps = 50
        Logistic(x, α = 1.0, β = 0.0, γ = 1.0) = γ/(1+exp(-α * (x - β)))
        α, β, γ = (1.0, 0.0, 1.0)
        activationScale = 1.0
        K = 5.0
        polyDegree = 3
        μ_size = 0.1
        pleiotropy = 1

        ## Width 5 
        parameterSets = [simParams( N,T,saveStep,reps,Logistic,α,β,γ,activationScale,K,
                polyDegree,netDepth,netWidth,regulationDepth,μ_size,pleiotropy    ) for regulationDepth in 1:5]

        ## Width 1
        [push!(parameterSets, simParams( N,T,saveStep,reps,Logistic,α,β,γ,activationScale,K,
        polyDegree,netDepth,1,regulationDepth,μ_size,pleiotropy)) for regulationDepth in 1:5]

        rawSimulationDataSets = pmap(simulateForModularity, parameterSets)
        SimulationData = vcat(DataFrame.(rawSimulationDataSets)...)
        selection = SimulationData[!, [:Networks, :netDepth, :netWidth, :regulationDepth]]
        # return selection
        NetworkGraphs = pmap(generateGraph,  Tuple.(eachrow(SimulationData[!, [:Networks, :netDepth, :netWidth, :regulationDepth]])))
        NetworkLabels = getfield.(map(label_propagation, NetworkGraphs), 1)
        NetworkModularities = [modularity(network, labels) for (network, labels) in zip(NetworkGraphs, NetworkLabels)]
        SimulationData[!, :Modularity] = NetworkModularities

        jldsave(dateString; SimulationData) 
        # return SimulationData

    end

    filestring = "ModularityTests"
end

@time main("ModularityTests")
