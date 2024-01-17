## this file calls simulations of various network sizes to compare how this influences adaptation
## Network parameters explored are both the total number of nodes, as well as the distribution of these nodes (width vs depth)
using Distributed

## Comparing different parameters for multi-processing
nprocs()

@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
    Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin
    using  JLD2, Dates, DataFrames
    include("Networks.jl")
    function generateSimulations(minNetSize = 1, maxNetSize = 30, minNetWidth = 1, maxNetWidth = 30, netSizeStep = 5, N = 1000, T = 1000, SaveStep = 1000, reps = 10, regulationDepth = 100, filestring = "GRN_Adaptation_Comparisons_")
        
        ##Global Parameters for all simulations
        dateString = string(filestring, Dates.now(), ".jld2")

        ## activation functions are Le Nagard's Exp (inverse Gaussian), the Gaussian Function, the Logistic function, and the Binary Step function
        ## α is the steepness of the response 
        ## β is the x-axis offset
        ## γ is the maximum value of responses 
        
        LeNagardExp(x, α=sqrt(1/2), β=0.0, γ = 1.0) = 1 - (γ * exp(-((x-β)^2/(2*(α^2)))))
        Gaussian(x, α=sqrt(1/2), β=0.0, γ = 1.0) = (γ * exp(-((x-β)^2/(2*(α^2)))))
        Logistic(x, α = 1.0, β = 0.0, γ = 1.0) = γ/(1+exp(-α * (x - β)))
        function BinaryStep(x, α, β, γ)
            if x < 0
                return 0.0
            else
                return 1.0
            end
        end
        activationFunctions = [LeNagardExp] ## add or remove as desired
        activationScale = 1.0
        α, β, γ, K = (1.0, 0.0, 1.0, 5.0)
        envChallenges = [1,2,3,4,5,6] ## Vector of each polynomial degree to check
        μ_size = .1

        ## Temporary containers
        simulationOutputs = Dict() ## Dictionary where the keys are parameters (environmental challenge)
        SimulationParameterSets = []
        networkSizes = []

        print("Beginning simulations with \n 
                maxNetSize : $maxNetSize \n 
                N : $N (Population size) \n 
                T : $T (Number of timesteps) \n 
                reps : $reps (number of replicates) \n")
                # nproc : $(nprocs()) (number of processes) \n"

        ## Main loop to create parameter sets
        for pleiotropyStrength in [1, 2, 4, 8]
            for regulationDepth in [1, (maxNetSize+1)]
                for a in [1.0]
                    for polyDegree in envChallenges
                        for activationFunction in activationFunctions
                            for i in minNetSize:netSizeStep:maxNetSize
                                push!(SimulationParameterSets, simParams(N, T, SaveStep, reps, activationFunction, a, β, γ, activationScale, K, polyDegree, i, 1, regulationDepth, μ_size, pleiotropyStrength))
                                push!(networkSizes, (i, 1))
                            end
                            ## Need to account for the fact that the first layer doesn't process when determining active nodes
                            ## This generates all networks of the same "active" size, meaning they have the same number of nodes outside the input layer
                            for width in minNetWidth:maxNetWidth
                                if mod(maxNetSize, width) == 0 ## only iterating with valid network sizes
                                    netDepth = Int((maxNetSize/width)+1)
                                    push!(SimulationParameterSets, simParams(N, T, SaveStep, reps, activationFunction, a, β, γ, activationScale, K, polyDegree, netDepth, width, regulationDepth, μ_size, pleiotropyStrength))
                                    push!(networkSizes, (netDepth, width))
                                end
                            end
                        end    
                    end
                end
            end
        end

        print("Succesfully generated $(length(networkSizes)) parameter sets with $(reps * length(networkSizes)) replicates, testing network sizes (Depth, Width): \n")
        [print(x, ",  ") for x in unique(networkSizes)]
        print("\n Running simulations... \n")
        outputComparisons = pmap(simulate, SimulationParameterSets)
        print("...done! Merging dataframes and saving to $dateString \n")
        simulationOutputs = vcat(DataFrame.(outputComparisons)...)
        jldsave(dateString; simulationOutputs) 

        ## Seeing if we actually saved as many replicates as we started with
        print("Successfully saved $(length(unique(simulationOutputs[simulationOutputs.T .== maximum(simulationOutputs.T), :].replicateID))) replicates! \n")
    end
    
    minNetSize = 5
    minNetWidth = 1
    maxNetSize = 5
    maxNetWidth = 8
    netStepSize = 1
    regulationDepth = maxNetSize + 1
    N = 1000
    T = 10
    SaveStep = 1
    reps = 50
    filestring = "PleiotropyDebug"
    
end 

@time simulationOutputs = generateSimulations(minNetSize, maxNetSize, minNetWidth, maxNetWidth, netStepSize, N, T, SaveStep, reps, regulationDepth, filestring)