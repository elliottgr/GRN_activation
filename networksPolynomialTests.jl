## This file is testing whether the degree of the Legendre polynomial
## has significant effects on the population. Using Le Nagard's fitness equation
## polynomials of even degree should see lower fitness (hypothesis)

## Using the invasion probability functions for quicker comparisons
include("networksInvasionProbability.jl")

## running some loops to generate datasets
## runs replicates on each of the Legendre Polynomials in the indicated range

function iterateOverPolynomials(polyMin = 0, polyMax = 5, N = 100, T = 500, reps = 25, Φ = (f(x) = (1-exp(-x^2))), α = 1.0, K = 5.0,  netSize = 5, μ_size = .1)
    simulationResults = []
    for i in polyMin:polyMax
        push!(simulationResults, simulate(N, T, reps, Φ, α, K, i, netSize, μ_size))
    end
    return simulationResults
end

function printFitness(simulationResults)
    meanFitnesses = []
    for i in simulationResults ## iterating over polynomials
        meanFitness = []
        for r in i[1] ## iterating over replicates in polynomial sets
            push!(meanFitness, last(r[1]))
        end
        push!(meanFitnesses, mean(meanFitness))
    end
    # return meanFitnesses
    for i in 1:length(meanFitnesses) ## I do not care that this is bad Julia code
        print("")
end