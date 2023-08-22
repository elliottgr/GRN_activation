## Builds the networks from scratch using code from the social FFN project
## arranges them into populations, and assesses their evolution
## Based on the individual model from Le Nagard 2011 that uses Wright-Fisher dynamics

include("networksFuncs.jl") ## loading formulas


## Plots are for visualization
## The rest are used to run the RNG for Wright-Fisher Selection
using  Plots, Random, Distributions



## Define population of N networks
function simulate(N=10, T = 10, reps = 1)
    ## Capture 
    ## W_m = weight matrices for node connection weights
    ## W_b = weight vector for node biases
    ## prev_out = running total of the iterations on the node
    ## j = current layer of iteration
    ## activation_scale = population level scaling parameter of inputs to influence behavior response
    ## activation_function = the thing we're testing :)

    ## test params
    Φ(x) = (1 - exp(-x^2)) ## Identity function to test activation
    α = 1.0 ## activation scale
    K = 1.0
    polynomialDegree = 4
    μ = 0.04 ## per capita chance of mutation
    μ_trait = 0.2 ## chance that a given weight in the network changes
    μ_size = 0.1
    NetSize = 4

    
    ##Iterating over all replicates
    meanFitnessReps = fill([], reps)
    varFitnessReps = fill(0.0, reps)

    for r in 1:reps
        parentNetwork = [rand(Float64, (NetSize, NetSize)), rand(Float64, NetSize)]
        population = []
        for i in 1:N
            push!(population, parentNetwork)
        end
        meanFitness = zeros(Float64, T)
        varFitness = zeros(Float64, T)
        for t in 1:T
            population, fitnessScores = timestep(population, Φ, α, K, polynomialDegree, μ, μ_trait, μ_size)
            meanFitness[t] = mean(fitnessScores)
            varFitness = var(meanFitness)
        end
        meanFitnessReps[r] = meanFitness
        varFitnessReps[r] = varFitness
    end
    return meanFitnessReps, varFitnessReps
end

function timestep(population, activation_function, activation_scale, K, polynomialDegree, μ, μ_trait, μ_size)

    N = length(population)
    NetSize = size(population[1][2])[1]

    ## find fitness of each member of population
    fitnessScores = zeros(Float64, N)
    for i in 1:N
        fitnessScores[i] = fitness(activation_function, activation_scale, K, polynomialDegree, population[i])
    end
  

    ## Code for simulating reproduction
    ## Should weight selection based on relative fitness

    ## Iterating over a blank population, seems to produce less errors? 

    newPop = fill([fill(0.0, NetSize, NetSize), fill(0.0, NetSize)], N)
    for i in 1:N
        newPop[i] = population[wsample(collect(1:N), fitnessScores)]
    end

    ## Mutate based on some parameters
    
    ## Le Nagard Method
    ## samples a random weight and shifts it
    for i in 1:N
        if rand() <= μ
            newPop[i] = mutateNetwork(μ_size, copy(newPop[i]))
            # weightID = sample(1:(NetSize^2 + NetSize))
            # if weightID <= NetSize^2
            #     newPop[i][1][weightID] += randn()*μ_size
            # else
            #     newPop[i][2][weightID-(NetSize^2)] += randn()*μ_size
            # end
        end
    end


    ## Social Evolution / JVC Method
    ## Each weight has a chance to mutate
    # for i in 1:N
    #     if rand() <= μ

    #     end
    # end

    ## update the population, return measured fitnesses

    return [newPop, fitnessScores]

end


function plotReplicates(N = 10, T = 10, reps = 1)
    plot(1:T, simulate(N, T, reps)[1])
end


## To do:

## check whether PL degree 1 agrees with network activation if I set it to maximum