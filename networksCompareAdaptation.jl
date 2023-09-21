## this file calls simulations of various network sizes to compare how this influences adaptation
## Network parameters explored are both the total number of nodes, as well as the distribution of these nodes (width vs depth)
using Distributed

maxNetSize = 3
maxNetWidth = 10
N = 1000
T = 10000
reps = 2

## Comparing different parameters for multi-processing
nprocs()

@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
    Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin
    using  JLD2, Dates ## For violin plots
    include("networksInvasionProbability.jl")
    function generateSimulations(maxNetSize = 30, maxNetWidth = 30, netSizeStep = 5, N = 1000, T = 1000, reps = 10)
        dateString = string("GRN_Adaptation_Comparisons_",Dates.now(), ".jld2")
    
        ##Global Parameters for all simulations
    
        activationFunction = (f(x) = (1-exp(-x^2)))
        activationScale = 1.0
        K = 5.0
        envChallenges = [1, 2, 3, 5, 7] ## Vector of each polynomial degree to check
        # envChallenges = [3]
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
    
            ## Rewritten to use multi-processing
            
            ## testing how the size of the network influences the final evolved fitness
            ## only varying network depth (number of layers) and keeping the number of nodes per layer the same
            networkDepthComparisons = []
    
            ## generates fitness histories for all networks of a given size
            ## only tests networks that have the same number of total nodes, but with different depths / widths
            networkWidthComparisons = []
    
            for i in 1:netSizeStep:maxNetSize
                push!(networkDepthComparisons, simParams(N, T, reps, activationFunction, activationScale, K, polyDegree, i, 1, μ_size))
            end
            
            ## Need to account for the fact that the first layer doesn't process when determining active nodes
            ## This generates all networks of the same "active" size, meaning they have the same number of nodes outside the input layer
            for width in 1:maxNetWidth
                if mod(maxNetSize, width) == 0 ## only iterating with valid network sizes
                    netDepth = Int((maxNetSize/width)+1)
                    push!(networkWidthComparisons, simParams(N, T, reps, activationFunction, activationScale, K, polyDegree, netDepth, width, μ_size))
                end
            end
            
            outputDepthComparisons = pmap(simulate, networkDepthComparisons)
            print("Depth comparisons of degree $polyDegree completed! \n")
            outputWidthComparisons = pmap(simulate, networkWidthComparisons)
            print("Width comparisons of degree $polyDegree completed! \n")
            simulationOutputs[polyDegree] = [outputDepthComparisons, outputWidthComparisons]
        end
        jldsave(dateString; simulationOutputs)
        # return simulationOutputs
        
    end
    
    maxNetSize = 12
    maxNetWidth = 12
    N = 1000
    T = 100000
    reps = 50

end 

@time simulationOutputs = generateSimulations(maxNetSize, maxNetWidth, 1, N, T, reps)