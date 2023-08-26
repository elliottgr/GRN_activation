## Using the probability of fixation from Le Nagard 2011

using Plots
include("networksFuncs.jl") ## taking formulas 

## Equivalent to Eq. 5, P(f_0 -> f_i), in le Nagard (2011)
function invasionProbability(activation_function, activation_scale, K, polynomialDegree, N, residentNetwork, mutantNetwork)

    resFitness = fitness(activation_function, activation_scale, K, polynomialDegree, residentNetwork)
    mutFitness = fitness(activation_function, activation_scale, K, polynomialDegree, mutantNetwork)
    fitnessRatio =  resFitness / mutFitness

    ## A few conditional statements broken down for debugging and dodging NaNs
    if mutFitness == 0.0
        out = 0.0
    else
        num = 1 - (fitnessRatio)^2
        den = 1 - ((fitnessRatio)^(2*N))
        out = num/den
         ## Debugging, will flood the console but prints all fitness tests
        # print("resFit : $resFitness  |  mutFit : $mutFitness  | fitnessRatio :  $fitnessRatio  |  Num : $num  |  Den : $den  |  Prob : $out \n")
    end
    if isnan(out) == true
        return 0.0
    else
        return out
    end
end

## Parameters:

## T = length of simulation / number of timesteps
## N = population size
## reps = replicates

function simulate(N = 10, T = 10, reps = 1, Φ = (f(x) = (1-exp(-x^2))), α = 1.0, K = 5.0, polyDegree = 1, netDepth = 5, netWidth = 6, μ_size = .1)

    ## Generates a random network, then mutates it
    fitnessHistories = [fill(0.0, T) for _ in 1:reps]
    invasionProbabilities = [fill(0.0, T) for _ in 1:reps]
    finalNetworks = [generateFilledNetwork(netDepth, netWidth, 0.0) for _ in 1:reps]
    for r in 1:reps
        resNet = generateNetwork(netDepth, netWidth) ## Initial resident network
        ## Main timestep loop
        for t in 1:T
            mutNet = mutateNetwork(μ_size, copy(resNet))
            invasionProb = invasionProbability(Φ, α, K, polyDegree, N, resNet, mutNet)
            invasionProbabilities[r][t] = invasionProb
            if rand() <= invasionProb
                resNet = mutNet
            end
            fitnessHistories[r][t] = fitness(Φ, α, K, polyDegree, resNet)
        end

        ## have to copy each index because of array rules
        finalNetwork = generateFilledNetwork(netDepth, netWidth, 0.0)
        for i in eachindex(resNet)
            finalNetwork[i] = copy(resNet[i])
        end
        finalNetworks[r] = resNet
    end
    return [fitnessHistories, invasionProbabilities, finalNetworks]
end




## Testing the network adaptation to the response curves 
N = 1000 ## N (population size)
T = 50000 ## T (simulation length)
reps = 2 ## number of replicates
Φ = (f(x) = (1 - exp(-x^2))) ## Le Nagard's activation function
# Φ = (f(x) = (1 / (1 + exp(-x)))) ## Logistic / sigmoid
# Φ = (f(x) = x) ## Linear activation
# Φ = (f(x) = maximum([0.0, x])) ## ReLU
α = 1.0 ## α (activation coefficient)
K = 5.0 ## K (strength of selection)
polyDegree = 3 ## degree of the Legendre Polynomial
netDepth = 3 ## Size of the networks
netWidth = 2
μ_size = .1 ## standard deviation of mutation magnitude

simResults = simulate(N, T, reps, Φ, α, K, polyDegree, netDepth, netWidth, μ_size)
plotReplicatesFitness(simResults)
plotResponseCurves(Φ, α, polyDegree, simResults)