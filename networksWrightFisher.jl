## Builds the networks from scratch using code from the social FFN project
## arranges them into populations, and assesses their evolution
## Based on the individual model from Le Nagard 2011 that uses Wright-Fisher dynamics

## Plots are for visualization
## The rest are used to run the RNG for Wright-Fisher Selection
using  Plots, Random, Distributions
include("networksFuncs.jl") ## loading formulas

## Define population of N networks
function simulate(N = 10, T = 10, reps = 1, activationFunction = (f(x) = (1-exp(-x^2))), activationScale = 1.0, K = 5.0, polyDegree = 1, netDepth = 5, netWidth=6, μ = 0.01, μ_size = .1)

    ##Iterating over all replicates
    meanFitnessReps = fill([], reps)
    varFitnessReps = fill(0.0, reps)
    finalNetworks = [generateFilledNetwork(netDepth, netWidth, 0.0) for _ in 1:reps]
    for r in 1:reps
        parentNetwork = generateNetwork(netDepth, netWidth)
        population = []
        for _ in 1:N
            push!(population, parentNetwork)
        end
        meanFitness = zeros(Float64, T)
        varFitness = zeros(Float64, T)
        for t in 1:T
            population, fitnessScores = timestep(population, activationFunction, activationScale, K, polyDegree, μ, μ_size)
            meanFitness[t] = mean(fitnessScores)
            varFitness = var(meanFitness)
        end
        meanFitnessReps[r] = meanFitness
        varFitnessReps[r] = varFitness
        
        ## sorting the networks and finding the highest fitness network in the population to return
        maxFitness = 0
        for network in population
            if fitness(activationFunction, activationScale, K, polyDegree, network) > maxFitness
                finalNetworks[r] = network
            end
        end
    end
    return [meanFitnessReps, varFitnessReps, finalNetworks]
end

function timestep(population, activationFunction, activationScale, K, polyDegree, μ, μ_size)

    N = length(population)
    netDepth, netWidth = size(population[1])

    ## find fitness of each member of population
    fitnessScores = zeros(Float64, N)
    for i in 1:N
        fitnessScores[i] = fitness(activationFunction, activationScale, K, polyDegree, population[i])
    end
  
    ## Code for simulating reproduction
    ## Should weight selection based on relative fitness
    ## Iterating over a blank population, seems to produce less errors? 

    # newPop = fill([fill(fill(0.0, (netDepth, netWidth)), (netDepth, netWidth)), zeros(Float64, (netDepth, netWidth))], N)
    newPop = fill(generateFilledNetwork(netDepth, netWidth, 0.0), N)
    for i in 1:N
        newPop[i] = population[wsample(collect(1:N), fitnessScores)]
    end

    ## Mutate based on some parameter
    
    ## Le Nagard Method
    ## samples a random weight and shifts it
    for i in 1:N
        if rand() <= μ
            newPop[i] = mutateNetwork(μ_size, copy(newPop[i]))
        end
    end

    ## update the population, return measured fitnesses

    return [newPop, fitnessScores]

end

## Testing the network adaptation to the response curves 
N = 10 ## N (population size)
T = 250 ## T (simulation length)
reps = 5 ## number of replicates
activationFunction = (f(x) = (1 - exp(-x^2))) ## Le Nagard's activation function
# activationFunction = (f(x) = (1 / (1 + exp(-x)))) ## Logistic / sigmoid
# activationFunction = (f(x) = x) ## Linear activation
# activationFunction = (f(x) = maximum([0.0, x])) ## ReLU
activationScale = 1.0 ## activationScale (activation coefficient)
K = 5.0 ## K (strength of selection)
polyDegree = 2 ## degree of the Legendre Polynomial
netDepth = 5 ## Size of the networks
netWidth = 6
μ_size = .1 ## standard deviation of mutation magnitude

simResults = simulate(N, T, reps, activationFunction, activationScale, K, polyDegree, netDepth, netWidth, μ_size)
plotReplicatesFitness(simResults)
plotResponseCurves(activationFunction, activationScale, polyDegree, simResults)