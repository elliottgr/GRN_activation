## Using the probability of fixation from Le Nagard 2011

using Plots
include("networksFuncs.jl") ## taking formulas 

## Equivalent to Eq. 5, P(f_0 -> f_i), in le Nagard (2011)
function invasionProbability(activation_function, activation_scale, K, polynomialDegree, N, residentNetwork, mutantNetwork)

    resFitness = fitness(activation_function, activation_scale, K, polynomialDegree, residentNetwork)
    mutFitness = fitness(activation_function, activation_scale, K, polynomialDegree, mutantNetwork)
    fitnessRatio =  resFitness / mutFitness
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

## Testing
## Generating random phenotypes and showing that they'll have some invasion chance
function debugInvasionProbability(trials, NetSize)
    resNet = [rand(Float64, (NetSize, NetSize)), rand(Float64, NetSize)]
    testHistory = fill(0.0, trials)
    Φ(x) = (1 - exp(-x^2))
    α = 1.0
    K = 1.0
    polyDegree = 1
    N = 100
    for t in 1:trials
        mutNet = [rand(Float64, (NetSize, NetSize)), rand(Float64, NetSize)]
        testHistory[t] = invasionProbability(Φ, α, K, 1, N, resNet, mutNet)
    end
    return testHistory
end

## Parameters:

## T = length of simulation / number of timesteps
## N = population size

function simulate(N = 10, T = 10, reps = 1)
    ## Generates a random network, then mutates it

    Φ(x) = (1 - exp(-x^2))
    α = 1.0
    K = 5.0
    polyDegree = 1
    NetSize = 5
    μ_size = .1
    fitnessHistories = [fill(0.0, T) for _ in 1:reps]
    invasionProbabilities = [fill(0.0, T) for _ in 1:reps]
    for r in 1:reps
        resNet = [rand(Float64, (NetSize, NetSize)), rand(Float64, NetSize)] ## Initial resident network

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
    end
    return [fitnessHistories, invasionProbabilities]
end

function plotReplicates(N = 50, T = 50, reps = 5)
    simulationResults = simulate(N, T, reps)
    fitnessPlot = plot(1:T, simulationResults[1], legend = :none)
    invasionProbPlot = plot(1:T, simulationResults[2], legend = :none)
    plot(fitnessPlot, invasionProbPlot, layout = (2,1), sharex=true)
end