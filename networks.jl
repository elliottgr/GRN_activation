## Builds the networks from scratch using code from the social FFN project
## arranges them into populations, and assesses their evolution

using LegendrePolynomials, Plots, Statistics, Random, Distributions, StatsBase

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
        # print((last(iterateNetwork(activation_function, activation_scale, i, network, LayerOutputs)) - Pl(i, polynomialDegree) ))
        x += (last(iterateNetwork(activation_function, activation_scale, i, network, LayerOutputs)) - Pl(i, polynomialDegree))
    end
    return x
end

## Testing
# W_m = fill(0.0, 2, 2)
# W_m = [0 0   
#        0 0]
# W_b = [0.0, 0.0]
# LayerOutputs = fill(0.0, 2)
# Φ(x) = (1 - exp(-x^2))

# iterateNetwork(Φ, 1.0, 0, [W_m, W_b], LayerOutputs) ## returns expected value (0) for a zero network
# measureNetwork(Φ, 1.0, 0, [W_m, W_b]) ## returns the expected value (-101.0) for a zero network measured against the null polynomial
# Pl(0, 0)
## Defining the fitness function

function fitness(activation_function, activation_scale, K, polynomialDegree, network)
    Wm, Wb = network
    Var_F = var(collect([Pl(i, polynomialDegree) for i in -1:0.02:1]))
    return exp((-K * (measureNetwork(activation_function, activation_scale, polynomialDegree, network))^2) / (100*Var_F))

end


# ## Testing
# W_m = fill(0.0, (2,2))
# W_b = [0.0, 0.0]
# Φ(x) = (1 - exp(-x^2))
# K = 5.0
# Var_F = var(collect([Pl(i, 1) for i in -1:0.02:1]))

# testFitness = exp( (-K * (2.6645352591003757e-15)^2) / (100* Var_F) )


# iterateNetwork(Φ, 1.0, 1, [W_m, W_b], LayerOutputs) ## returns expected value (0) for a zero network
# measureNetwork(Φ, 1.0, 1, [W_m, W_b]) ## returns the expected value (-101.0) for a zero network measured against the null polynomial
# fitness(Φ, 1.0, K, 1, [W_m, W_b])

## Checking the fitness function


## Define population of N networks
function simulate(N=10, T = 10, reps = 5)
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
    polynomialDegree = 1
    μ = 0.01 ## per capita chance of mutation
    μ_trait = 0.2 ## chance that a given weight in the network changes
    μ_size = 0.1
    NetSize = 2
    parentNetwork = [rand(Float64, (NetSize, NetSize)), rand(Float64, NetSize)]
    
    ##Iterating over all replicates
    meanFitnessReps = fill([], reps)
    for r in 1:reps
    
        population = []
        for i in 1:N
            push!(population, parentNetwork)
        end
        meanFitness = zeros(Float64, T)
        
        for t in 1:T
            population, fitnessScores = timestep(population, Φ, α, K, polynomialDegree, μ, μ_trait, μ_size)
            meanFitness[t] = mean(fitnessScores)
        end
        meanFitnessReps[r] = meanFitness
    end
    return meanFitnessReps
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
    rawFitnessScores = copy(fitnessScores)
    ## normalizing the fitness scores
    # for i in 1:N
    #     fitnessScores[i] = fitnessScores[i]/maximum(fitnessScores)
    # end
    # print("Mean per capita fitness: ", mean(fitnessScores), "\n")
    # print("Total fitness of all individuals: ", sum(fitnessScores), "\n")
    ## Reproduce based on relative fitness


    ## Code for simulating reproduction
    ## Should weight selection based on relative fitness
    
    reproductionIndex = collect(1:N)
    for i in 1:N
        reproductionIndex[i] = wsample(collect(1:N), fitnessScores)
    end

    # reproductionIndex = sample(1:N, ProbabilityWeights(fitnessScores), N, replace = true)
    # reproductionIndex = sample(1:N, N, replace = true)
    newPop = fill([fill(0.0, NetSize, NetSize), fill(0.0, NetSize)], N)
    for i in 1:N
        newPop[i] = population[reproductionIndex[i]]
    end

    ## Mutate based on some parameters
    
    ## Le Nagard Method
    ## samples a random weight and shifts it
    for i in 1:N
        if rand() <= μ
            weightID = sample(1:(NetSize^2 + NetSize))
            if weightID <= NetSize^2
                newPop[i][1][weightID] += randn()
            else
                newPop[i][2][weightID-(NetSize^2)] += randn()
            end
        end
    end


    ## Social Evolution / JVC Method
    ## Each weight has a chance to mutate
    # for i in 1:N
    #     if rand() <= μ
    #         newPop[i][1] += rand(Binomial(1, μ_trait), (NetSize, NetSize)) .* rand(Normal(0, μ_size), (NetSize, NetSize))
    #         newPop[i][2] += rand(Binomial(1, μ_trait), NetSize) .* rand(Normal(0, μ_size),NetSize)
    #     end
    # end

    ## update the population
    return [newPop, rawFitnessScores]
end

plt = plot(1:50, simulate(10,50, 5))


## To do:

## check whether PL degree 1 agrees with network activation if I set it to maximum