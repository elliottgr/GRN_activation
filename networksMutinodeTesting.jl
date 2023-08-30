## Debug file Checking that the calcOj function returns the same results using the old and new versions

using Plots ## I want to look at some of this :)
include("networksFuncs.jl") ## Importing current version

function oldCalcOj(activationFunction::Function, activationScale::Float64, j::Int, prev_out, Wm, Wb)
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
    return(activationFunction(activationScale * x)) 
end

# ## This generates an N node network with no weights and calculates the calcOj of it
## Does not include all parameters of the later version  
function testOldCalcOj(activationFunction::Function, activationScale::Float64, netDepth = 2, netWeights = 0.0, testInput = 0.0, calcLayer = 1)
    W_m = fill(netWeights, (netDepth,netDepth))
    W_b = fill(netWeights, netDepth)
    prev_out = fill(0.0, netDepth) ## blank N node input
    prev_out[1] = testInput ## probably a way to do this in one line lol
    oldCalcOj(activationFunction, activationScale, calcLayer, prev_out, W_m, W_b)
end

function testNewCalcOj(activationFunction::Function, activationScale::Float64, netDepth = 2, netWidth = 1, netWeights = 0.0, testInput = 0.0, calcLayer = 1, calcNode = 1)
    W_m, W_b = generateFilledNetwork(netDepth, netWidth, netWeights)
    activationMatrix = zeros(Float64, (netDepth, netWidth))
    testInputVector = zeros(Float64, netWidth)
    testInputVector[1] = testInput
    activationMatrix[1, 1:netWidth] = testInputVector
    calcOj(activationFunction, activationScale, calcNode, calcLayer, activationMatrix, W_m, W_b)
end

## Plots the output of a calcOj iteration for a range of inputs
function compareCalcOj(activationFunction, activationScale, netDepth = 2, netWidth = 1, netWeights = 0.0, calcLayer = 1, calcNode = 1, inputRange = -1:0.02:1)
    oldCalcOj_i = [testOldCalcOj(activationFunction, activationScale, netDepth, netWeights, i, calcLayer) for i in inputRange]
    newCalcOj_i = [testNewCalcOj(activationFunction, activationScale, netDepth, netWidth, netWeights, i, calcLayer, calcNode) for i in inputRange]
    # plot([oldCalcOj_i, newCalcOj_i], labels = ["Old", "New"])
    return oldCalcOj_i == newCalcOj_i
end

## testing calcOj over a range of weights
function compareAcrossWeightsCalcOj(activationFunction, activationScale, netDepth, netWidth, netWeights, calcLayer, calcNode, inputRange)
    check = true
    for i in inputRange
        if compareCalcOj(activationFunction, activationScale, netDepth, netWidth, i, calcLayer, calcNode, inputRange) != true ## True if true for all tests
            check = false
        end
    end
    return check
end

function oldIterateNetwork(activationFunction::Function, activationScale, input, network, prev_out)
    ##############################
    ## Calculates the total output of the network,
    ## iterating over calcOj() for each layer
    ##############################
    Wm, Wb = network ## for clarity in the future
    prev_out[1] = input
    for j in 2:length(Wb)
        prev_out[j] = oldCalcOj(activationFunction, activationScale, j, prev_out, Wm, Wb)
    end
    return prev_out
end


function testOldIterateNetwork(activationFunction::Function, activationScale, netDepth, netWidth, netWeights, testInput)
    W_m = fill(netWeights, (netDepth,netDepth))
    W_b = fill(netWeights, netDepth)
    prev_out = fill(0.0, netDepth) ## blank N node input
    prev_out[1] = testInput ## probably a way to do this in one line lol
    oldIterateNetwork(activationFunction, activationScale, testInput, [W_m, W_b], prev_out)
end

function testNewIterateNetwork(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput)
    W_m, W_b = generateFilledNetwork(netDepth, netWidth, netWeights)
    activationMatrix = zeros(Float64, (netDepth, netWidth))
    testInputVector = zeros(Float64, netWidth)
    testInputVector[1] = testInput
    activationMatrix[1, 1:netWidth] = testInputVector
    iterateNetwork(activationFunction, activationScale, testInputVector, [W_m, W_b], activationMatrix)
end

function compareIterateNetwork(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput)
    return testOldIterateNetwork(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput) == vec(testNewIterateNetwork(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput))
end

function compareAcrossWeightsIterateNetwork(activationFunction, activationScale, netDepth, netWidth, weightRange, inputRange)
    for input in inputRange
        for weight in weightRange
            if compareIterateNetwork(activationFunction, activationScale, netDepth, netWidth, weight, input) == false
                return false
            end
        end
    end
    return true
end

activationFunction = (f(x) = (1 - exp(-x^2))) 
activationScale = 1.0
netDepth = 5
netWidth = 1
netWeights = 10.0
testInput = 0.0
calcLayer = 1
calcNode = 1
inputRange = -1:0.2:1 ## just using the step function from Le Nagard

function oldMeasureNetwork(activationFunction, activationScale, polynomialDegree, network)
    x = 0
    for i in -1:0.02:1
        LayerOutputs = zeros(Float64, size(network[2])) ## size of the bias vector
        # print((last(iterateNetwork(activationFunction, activationScale, i, network, LayerOutputs)) - Pl(i, polynomialDegree) ))
        x += (last(iterateNetwork(activationFunction, activationScale, i, network, LayerOutputs)) - PlNormalized(i, polynomialDegree, 0, 1))
    end
    return x
end

oldMeasureNetwork(activationFunction)

## these checks should come back the same regardless of network dimension
samples = 10 ## Showing that a random sample of weights will return the same value in both cases
testOldCalcOj(activationFunction, activationScale, netDepth, netWeights, testInput, calcLayer) == testNewCalcOj(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput, calcLayer, calcNode)
testOldIterateNetwork(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput)[netDepth] == testNewIterateNetwork(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput)[netDepth, netWidth]
compareAcrossWeightsCalcOj(activationFunction, activationScale, netDepth, netWidth, netWeights, calcLayer, calcNode, collect(randn((samples))))


## These checks should come back true if the networks are of width = 1, but false otherwise
compareAcrossWeightsIterateNetwork(activationFunction, activationScale, netDepth, netWidth, collect(randn((samples))), collect(randn((samples))))
