## Debug file Checking that the calcOj function returns the same results using the old and new versions

using Plots ## I want to look at some of this :)
include("networksFuncs.jl") ## Importing current version

function oldCalcOj(activation_function::Function, activation_scale::Float64, j::Int, prev_out, Wm, Wb)
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

# ## This generates an N node network with no weights and calculates the calcOj of it
## Does not include all parameters of the later version  
function testOldCalcOj(activation_function::Function, activation_scale::Float64, netDepth = 2, netWeights = 0.0, testInput = 0.0, calcLayer = 1)
    W_m = fill(netWeights, (netDepth,netDepth))
    W_b = fill(netWeights, netDepth)
    prev_out = fill(0.0, netDepth) ## blank N node input
    prev_out[1] = testInput ## probably a way to do this in one line lol
    oldCalcOj(activation_function, activation_scale, calcLayer, prev_out, W_m, W_b)
end

function testNewCalcOj(activation_function::Function, activation_scale::Float64, netDepth = 2, netWidth = 1, netWeights = 0.0, testInput = 0.0, calcLayer = 1, calcNode = 1)
    W_m, W_b = generateFilledNetwork(netDepth, netWidth, netWeights)
    activationMatrix = zeros(Float64, netWidth)
    testInputVector = zeros(Float64, netWidth)
    testInputVector[1] = testInput
    activationMatrix[1, 1:netWidth] = testInputVector
    calcOj(activation_function, activation_scale, calcNode, calcLayer, activationMatrix, W_m, W_b)
end

## Plots the output of a calcOj iteration for a range of inputs
function compareCalcOj(activation_function, activation_scale, netDepth = 2, netWidth = 1, netWeights = 0.0, calcLayer = 1, calcNode = 1, inputRange = -1:0.02:1)
    oldCalcOj_i = [testOldCalcOj(Φ, α, netDepth, netWeights, i, calcLayer) for i in inputRange]
    newCalcOj_i = [testNewCalcOj(Φ, α, netDepth, netWidth, netWeights, i, calcLayer, calcNode) for i in inputRange]
    # plot([oldCalcOj_i, newCalcOj_i], labels = ["Old", "New"])
    return oldCalcOj_i == newCalcOj_i
end

## testing calcOj over a range of weights
function compareAcrossWeightsCalcOj(activation_function, activation_scale, netDepth, netWidth, netWeights, calcLayer, calcNode, inputRange)
    check = true
    for i in inputRange
        if compareCalcOj(activation_function, activation_scale, netDepth, netWidth, netWeights, calcLayer, calcNode, inputRange) != true ## True if true for all tests
            check = false
        end
    end
    return check
end

function oldIterateNetwork(activation_function::Function, activation_scale, input, network, prev_out)
    ##############################
    ## Calculates the total output of the network,
    ## iterating over calcOj() for each layer
    ##############################
    Wm, Wb = network ## for clarity in the future
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = oldCalcOj(activation_function, activation_scale, j, prev_out, Wm, Wb)
    end
    return prev_out
end

Φ = (f(x) = (1 - exp(-x^2))) 
α = 1.0
netDepth = 5
netWidth = 1
netWeights = 0.0
testInput = 0.0
calcLayer = 1
calcNode = 1
inputRange = -1:0.2:1 ## just using the step function from Le Nagard
testOldCalcOj(Φ, α, netDepth, netWeights, testInput, calcLayer)
testNewCalcOj(Φ, α, netDepth, netWidth, netWeights, testInput, calcLayer, calcNode)
compareAcrossWeightsCalcOj(Φ, α, netDepth, netWidth, netWeights, calcLayer, calcNode, inputRange)

function testOldIterateNetwork(activation_function::Function, activation_scale, netDepth, netWidth, netWeights, testInput)
    W_m = fill(netWeights, (netDepth,netDepth))
    W_b = fill(netWeights, netDepth)
    prev_out = fill(0.0, netDepth) ## blank N node input
    prev_out[1] = testInput ## probably a way to do this in one line lol
    oldIterateNetwork(Φ, α, testInput, [W_m, W_b], prev_out)
end

function testNewIterateNetwork(activation_function, activation_scale, netDepth, netWidth, netWeights, testInput)
    W_m, W_b = generateFilledNetwork(netDepth, netWidth, netWeights)
    activationMatrix = zeros(Float64, (netDepth, netWidth))
    testInputVector = zeros(Float64, netWidth)
    testInputVector[1] = testInput
    activationMatrix[1, 1:netWidth] = testInputVector
    iterateNetwork(activation_function, activation_scale, testInputVector, [W_m, W_b], activationMatrix)
end
print(testOldIterateNetwork(Φ, α, netDepth, netWidth, netWeights, testInput) == vec(testNewIterateNetwork(Φ, α, netDepth, netWidth, netWeights, testInput)))