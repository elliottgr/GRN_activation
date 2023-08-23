## Builds the networks from scratch using code from the social FFN project
## arranges them into populations, and assesses their evolution
## Based on the individual model from Le Nagard 2011 that uses Wright-Fisher dynamics

## Plots are for visualization
## The rest are used to run the RNG for Wright-Fisher Selection
using  Plots, Random, Distributions
include("networksFuncs.jl") ## loading formulas

## Define population of N networks
function simulate(N = 10, T = 10, reps = 1, Φ = (f(x) = (1-exp(-x^2))), α = 1.0, K = 5.0, polyDegree = 1, netSize = 5, μ = 0.01, μ_size = .1)

    ##Iterating over all replicates
    meanFitnessReps = fill([], reps)
    varFitnessReps = fill(0.0, reps)
    finalNetworks = [[fill(0.0, (netSize, netSize)), fill(0.0, netSize)] for _ in 1:reps]
    for r in 1:reps
        parentNetwork = generateNetwork(netSize)
        population = []
        for _ in 1:N
            push!(population, parentNetwork)
        end
        meanFitness = zeros(Float64, T)
        varFitness = zeros(Float64, T)
        for t in 1:T
            population, fitnessScores = timestep(population, Φ, α, K, polyDegree, μ, μ_size)
            meanFitness[t] = mean(fitnessScores)
            varFitness = var(meanFitness)
        end
        meanFitnessReps[r] = meanFitness
        varFitnessReps[r] = varFitness
        
        ## sorting the networks and finding the highest fitness network in the population to return
        maxFitness = 0
        for network in population
            if fitness(Φ, α, K, polyDegree, network) > maxFitness
                finalNetworks[r] = network
            end
        end
    end
    return [meanFitnessReps, varFitnessReps, finalNetworks]
end

function timestep(population, activation_function, activation_scale, K, polyDegree, μ, μ_size)

    N = length(population)
    netSize = size(population[1][2])[1]

    ## find fitness of each member of population
    fitnessScores = zeros(Float64, N)
    for i in 1:N
        fitnessScores[i] = fitness(activation_function, activation_scale, K, polyDegree, population[i])
    end
  
    ## Code for simulating reproduction
    ## Should weight selection based on relative fitness
    ## Iterating over a blank population, seems to produce less errors? 

    newPop = fill([fill(0.0, netSize, netSize), fill(0.0, netSize)], N)
    for i in 1:N
        newPop[i] = population[wsample(collect(1:N), fitnessScores)]
    end

    ## Mutate based on some parameters
    
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

## Plotting specific functions

function plotReplicatesFitness(simulationResults)
    netSize = length(simulationResults[3][1][2])
    numReps = length(simulationResults[3])
    titleStr = string("Fitness of $numReps replicates for network size $netSize")

    fitnessPlot = plot(1:length(simulationResults[1][1]), simulationResults[1], title = titleStr)
end

function plotResponseCurves(activation_function, activation_scale, polyDegree, simulationResults)
    ## Plotting the polynomial curve
    plt = plot(-1:0.02:1, collect([PlNormalized(i, polyDegree, 0, 1) for i in -1:0.02:1]), label = "Target")
    
    ## Updating it with the fitness of each replicate
    for network in simulationResults[3]
        valueRange = collect(-1:0.02:1) ## the range of input values the networks measure against
        responseCurveValues = []
        for i in valueRange
            LayerOutputs = zeros(Float64, size(network[2])) ## size of the bias vector
            push!(responseCurveValues, last(iterateNetwork(activation_function, activation_scale, i, network, LayerOutputs)))
        end
        plt = plot!(-1:0.02:1, responseCurveValues)
    end
    return plt
end

## Testing the network adaptation to the response curves 
N = 100 ## N (population size)
T = 250 ## T (simulation length)
reps = 10 ## number of replicates
Φ = (f(x) = (1 - exp(-x^2))) ## Le Nagard's activation function
# Φ = (f(x) = (1 / (1 + exp(-x)))) ## Logistic / sigmoid
# Φ = (f(x) = x) ## Linear activation
# Φ = (f(x) = maximum([0.0, x])) ## ReLU
α = 1.0 ## α (activation coefficient)
K = 5.0 ## K (strength of selection)
polyDegree = 2 ## degree of the Legendre Polynomial
netSize = 10 ## Size of the networks
μ_size = .1 ## standard deviation of mutation magnitude

simResults = simulate(N, T, reps, Φ, α, K, polyDegree, netSize, μ_size)
plotReplicatesFitness(simResults)
plotResponseCurves(Φ, α, polyDegree, simResults)