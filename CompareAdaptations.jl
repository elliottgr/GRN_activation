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
        LeNagardExp(x) = 1 - exp(-x^2)
        Gaussian(x) = exp(-x^2)
        Logistic(x) = 1/(1+exp(-x))
        function BinaryStep(x)
            if x < 0
                return 0.0
            else
                return 1.0
            end
        end
        activationFunctions = [LeNagardExp, Gaussian, Logistic, BinaryStep]
        activationFunction = LeNagardExp ## setting default to avoid breaking backwards compatibility
        activationScale = 1.0
        K = 5.0
        envChallenges = [1] ## Vector of each polynomial degree to check
        μ_size = .1
        simulationOutputs = Dict() ## Dictionary where the keys are parameters (environmental challenge)
        print("Beginning simulations with \n 
                maxNetSize : $maxNetSize \n 
                N : $N (Population size) \n 
                T : $T (Number of timesteps) \n 
                reps : $reps (number of replicates) \n")
                ## nproc : $(nprocs()) (number of processes) \n"
    
        for polyDegree in envChallenges
            print("Now testing Legendre Polynomials of degree $polyDegree \n")
    
            
            ## testing how the size of the network influences the final evolved fitness
            ## only varying network depth (number of layers) and keeping the number of nodes per layer the same
            networkDepthComparisons = []
    
            ## generates fitness histories for all networks of a given size
            ## only tests networks that have the same number of total nodes, but with different depths / widths
            networkWidthComparisons = []

            ## compares the activation functions using the preset network width and depth (I dont want to iterate over all of the possible network sizes, but want to preserve file structure to reconstruct tabular data later)
            ## ELG (note to self): THIS ORDER MUST BE MAINTAINED TO PRESERVE TABULAR DATA STRUCTURES (PLEASE DO NOT BREAK IT WITHOUT UPDATING GeneratePlots.jl AS WELL!!!!!!!!!!!!!!!)
            ActivationFunctionTestDepth = 6
            ActivationFunctionTestWidth = 6
            LeNagardComparisons = [simParams(N, T, reps, LeNagardExp, activationScale, K, polyDegree, ActivationFunctionTestDepth, ActivationFunctionTestWidth, μ_size)]
            GaussianComparisons = [simParams(N, T, reps, Gaussian, activationScale, K, polyDegree, ActivationFunctionTestDepth, ActivationFunctionTestWidth, μ_size)]
            LogisticComparisons = [simParams(N, T, reps, Logistic, activationScale, K, polyDegree, ActivationFunctionTestDepth, ActivationFunctionTestWidth, μ_size)]
            BinaryStepComparisons = [simParams(N, T, reps, BinaryStep, activationScale, K, polyDegree, ActivationFunctionTestDepth, ActivationFunctionTestWidth, μ_size)]

            for i in minNetSize:netSizeStep:maxNetSize
                push!(networkDepthComparisons, simParams(N, T, reps, activationFunction, activationScale, K, polyDegree, i, 1, μ_size))
            end
            
            ## Need to account for the fact that the first layer doesn't process when determining active nodes
            ## This generates all networks of the same "active" size, meaning they have the same number of nodes outside the input layer
            for width in minNetWidth:maxNetWidth
                if mod(maxNetSize, width) == 0 ## only iterating with valid network sizes
                    netDepth = Int((maxNetSize/width)+1)
                    push!(networkWidthComparisons, simParams(N, T, reps, activationFunction, activationScale, K, polyDegree, netDepth, width, μ_size))
                end
            end

            outputDepthComparisons = pmap(simulate, networkDepthComparisons)
            print("Depth comparisons of degree $polyDegree completed! \n")
            outputWidthComparisons = pmap(simulate, networkWidthComparisons)
            print("Width comparisons of degree $polyDegree completed! \n")
            outputLeNagardComparisons = pmap(simulate, LeNagardComparisons)
            print("LeNagard (inverse Gaussian) activation comparisons of degree $polyDegree completed! \n")
            outputGaussianComparisons = pmap(simulate, GaussianComparisons)
            print("Gaussian activation comparisons of degree $polyDegree completed! \n")
            outputLogisticComparisons = pmap(simulate, LogisticComparisons)
            print("Logistic activation comparisons of degree $polyDegree completed! \n")
            outputBinaryStepComparisons = pmap(simulate, BinaryStepComparisons)
            print("Binary Step activation comparisons of degree $polyDegree completed! \n")

            simulationOutputs[polyDegree] = [outputDepthComparisons, 
                                            outputWidthComparisons, 
                                            outputLeNagardComparisons, 
                                            outputGaussianComparisons,
                                            outputLogisticComparisons,
                                            outputBinaryStepComparisons]
        end
        jldsave(dateString; simulationOutputs)
        
    end
    minNetSize = 12
    minNetWidth = 12
    maxNetSize = 12
    maxNetWidth = 12
    netStepSize = 1
    N = 1000
    T = 5000
    reps = 3
    filestring = "ActivationFunctionComparisons"
end 

@time simulationOutputs = generateSimulations(minNetSize, maxNetSize, minNetWidth, maxNetWidth, netStepSize, N, T, reps, filestring)