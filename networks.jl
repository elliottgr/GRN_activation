## Builds the networks from scratch using code from the social FFN project
## arranges them into populations, and assesses their evolution

using LegendrePolynomials, Statistics, Random, Distributions, StatsBase

function calcOj(activation_function::Function, activation_scale::Float64, j::Int, prev_out, Wm, Wb)
    ##############################
    ## Iterates a single layer of the Feed Forward network
    ##############################
    ## dot product of Wm and prev_out, + node weights. Equivalent to x = dot(Wm[1:j,j], prev_out[1:j]) + Wb[j]
    ## doing it this way allows scalar indexing of the static arrays, which is significantly faster and avoids unnecessary array invocation
    x = 0
    for i in 1:j-1
        x += (Wm[i, j] * prev_out[i]) 
    end
    x += Wb[j]
    return(activation_function(activation_scale * x)) 
end

## Define g(i), the network response to some input
## is the gradient iterator, g(i) = -1 + (2 * i/100)
## Here I'm just using it as an input range 
## N(g(i)) is just the network evaluation at some g(i) 
## Still calling the function iterateNetwork for clarity

function iterateNetwork(activation_function::Function, activation_scale, input, network, prev_out)
    ##############################
    ## Calculates the total output of the network,
    ## iterating over calcOj() for each layer
    ##############################
    Wm, Wb = network ## for clarity in the future
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = calcOj(activation_function, activation_scale, j, prev_out, Wm, Wb)
    end
    return prev_out
end


## Define R(g(i)), the target the network is training towards
## R is just some Legendre Polynomial, which I'm defining
## as R(i, n)  with i as value and with n as degree. 
## Using the "LegendrePolynomial" package to generate these
##  

## Fitness Evaluation of network
## Need to generate N(g(i)) - R(g(i))

## this function measures the fit of the network versus the chosen legendre polynomial 
function measureNetwork(activation_function, activation_scale, polynomialDegree, network)
    x = 0
    for i in -1:0.02:1
        LayerOutputs = zeros(Float64, size(network[2])) ## size of the bias vector
        x += (last(iterateNetwork(activation_function, activation_scale, i, network, LayerOutputs)) - Pl(i, polynomialDegree))
    end
    return x
end

## Defining the fitness function

function fitness(activation_function, activation_scale, K, polynomialDegree, network)
    Wm, Wb = network
    Var_F = var([Pl(i, polynomialDegree) for i in -1:0.02:1])
    return exp((-K * (measureNetwork(activation_function, activation_scale, polynomialDegree, network))^2) / (100*Var_F))

end



## Define population of N networks
function simulate(N, T = 10)
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
    K = 5.0
    polynomialDegree = 3
    μ = 0.1 ## per capita chance of mutation
    μ_trait = 0.1 ## chance that a given weight in the network changes
    μ_size = 0.01
    NetSize = 3


    population = []
    for i in 1:N
        push!(population, [rand(Float64, (NetSize, NetSize)), rand(Float64, NetSize)])
    end
    meanFitness = zeros(Float64, T)
    
    for t in 1:T
        population, fitnessScores = timestep(population, Φ, α, K, polynomialDegree, μ, μ_trait, μ_size)
        meanFitness[t] = mean(fitnessScores)
    end
    return meanFitness
end

function timestep(population, activation_function, activation_scale, K, polynomialDegree, μ, μ_trait, μ_size)

    N = length(population)
    NetSize = size(population[1][2])[1]
    network = [fill(0.0, NetSize, NetSize), fill(0.0, NetSize)]

    ## find fitness of each member of population
    fitnessScores = zeros(Float64, N)
    for i in 1:N
        fitnessScores[i] = fitness(activation_function, activation_scale, K, polynomialDegree, population[i])
    end

    ## normalizing the fitness scores
    for i in 1:N
        fitnessScores[i] = fitnessScores[i]/maximum(fitnessScores)
    end
    # print(mean(fitnessScores), "\n")
    ## Reproduce based on relative fitness

    reproductionIndex = sample(1:N, ProbabilityWeights(fitnessScores), N, replace = true)

    newPop = fill([fill(0.0, NetSize, NetSize), fill(0.0, NetSize)], N)
    for i in 1:N
        newPop[i] = population[reproductionIndex[i]]
    end

    ## Mutate based on some parameters
    for i in 1:N
        if rand() <= μ
            newPop[i][1] += rand(Binomial(1, μ_trait), (NetSize, NetSize)) .* rand(Normal(0, μ_size), (NetSize, NetSize))
            newPop[i][2] += rand(Binomial(1, μ_trait), NetSize) .* rand(Normal(0, μ_size),NetSize)
        end
    end

    ## update the population
    return [population, fitnessScores]
end

## To do:

## check whether PL degree 1 agrees with network activation if I set it to maximum