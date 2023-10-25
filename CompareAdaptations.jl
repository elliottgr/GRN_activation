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
    using  JLD2, Dates ## For violin plots
    include("Networks.jl")
    function generateSimulations(minNetSize = 1, maxNetSize = 30, minNetWidth = 1, maxNetWidth = 30, netSizeStep = 5, N = 1000, T = 1000, reps = 10, filestring = "GRN_Adaptation_Comparisons_")
        dateString = string(filestring, Dates.now(), ".jld2")
        ##Global Parameters for all simulations
        
        
        ## activation functions are Le Nagard's Exp (inverse Gaussian), the Gaussian Function, the Logistic function, and the Binary Step function
        LeNagardExp(x, α, β, γ) = 1 - exp(-x^2)
        Gaussian(x, α, β, γ) = exp(-x^2)
        Logistic(x, α, β, γ = 1) = γ/(1+exp(-α * (x - β)))
        function BinaryStep(x, α, β, γ)
            if x < 0
                return 0.0
            else
                return 1.0
            end
        end
        activationFunctions = [Logistic]
        activationScale = 1.0
        K = 5.0
        envChallenges = [1,2,3,4,5] ## Vector of each polynomial degree to check
        μ_size = .1
        simulationOutputs = Dict() ## Dictionary where the keys are parameters (environmental challenge)
        print("Beginning simulations with \n 
                maxNetSize : $maxNetSize \n 
                N : $N (Population size) \n 
                T : $T (Number of timesteps) \n 
                reps : $reps (number of replicates) \n")
                # nproc : $(nprocs()) (number of processes) \n"
    
        SimulationParameterSets = []
        networkSizes = []
        for polyDegree in envChallenges
            for activationFunction in activationFunctions
                for i in minNetSize:netSizeStep:maxNetSize
                    push!(SimulationParameterSets, simParams(N, T, reps, activationFunction, α, β, γ, activationScale, K, polyDegree, i, 1, μ_size))
                    push!(networkSizes, (i, 1))
                end
                ## Need to account for the fact that the first layer doesn't process when determining active nodes
                ## This generates all networks of the same "active" size, meaning they have the same number of nodes outside the input layer
                for width in minNetWidth:maxNetWidth
                    if mod(maxNetSize, width) == 0 ## only iterating with valid network sizes
                        netDepth = Int((maxNetSize/width)+1)
                        push!(SimulationParameterSets, simParams(N, T, reps, activationFunction, α, β, γ, activationScale, K, polyDegree, netDepth, width, μ_size))
                        push!(networkSizes, (netDepth, width))
                    end
                end
            end    
        end
        print("Succesfully generated $(length(networkSizes)) parameter sets, testing network sizes (Depth, Width): \n")
        [print(x, ",  ") for x in unique(networkSizes)]
        print("\n Running simulations... \n")
        outputComparisons = pmap(simulate, SimulationParameterSets)
        print("...done! Merging dataframes and saving to $dateString \n")

        simulationOutputs = mergewith(vcat, outputComparisons...)
        
        jldsave(dateString; simulationOutputs)
        
    end
    minNetSize = 1
    minNetWidth = 1
    maxNetSize = 4
    maxNetWidth = 4
    netStepSize = 1
    N = 100
    T = 1000
    reps = 3
    filestring = "PmapTesting"
    
end 

@time simulationOutputs = generateSimulations(minNetSize, maxNetSize, minNetWidth, maxNetWidth, netStepSize, N, T, reps, filestring)