## Using the probability of fixation from Le Nagard 2011

include("networksFuncs.jl") ## taking formulas 

## Equivalent to Eq. 5, P(f_0 -> f_i), in le Nagard (2011)
function invasionProbability(activation_function, activation_scale, K, polynomialDegree, N, residentNetwork, mutantNetwork)
    num = 1 - (fitness(activation_function, activation_scale, K, polynomialDegree, residentNetwork) / fitness(activation_function, activation_scale, K, polynomialDegree, mutantNetwork))^2
    den = 1 - (fitness(activation_function, activation_scale, K, polynomialDegree, residentNetwork) / fitness(activation_function, activation_scale, K, polynomialDegree, mutantNetwork))^(2*N)
    return num/den
end

## Testing
## Generating random phenotypes and showing that they'll have some invasion chance
# function debugInvasionProbability(NetSize)
#     resNet = [rand(Float64, (NetSize, NetSize)), rand(Float64, NetSize)]
#     mutNet = [rand(Float64, (NetSize, NetSize)), rand(Float64, NetSize)]
#     Φ(x) = (1 - exp(-x^2))
#     α = 1.0
#     K = 1.0
#     polyDegree = 1
#     N = 100
#     return invasionProbability(Φ, α, K, 1, N, resNet, mutNet)
# end



function simulate()

    ## Generates a random network, then mutates it

    residentNetwork = 

    ## samples a random weight and shifts it

