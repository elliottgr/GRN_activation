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


## testing calcOj over a range of weights
function compareCalcOj(activationFunction, activationScale, netDepth, netWidth, calcLayer, calcNode, weightRange, inputRange)
    for w in weightRange
        for i in inputRange
            if testOldCalcOj(activationFunction, activationScale, netDepth, w, i, calcLayer) != testNewCalcOj(activationFunction, activationScale, netDepth, netWidth, w, i, calcLayer, calcNode)
                return false
            end 
        end
    end
    return true
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

function compareIterateNetwork(activationFunction, activationScale, netDepth, netWidth, weightRange, inputRange)
    for i in inputRange
        for w in weightRange
            if testOldIterateNetwork(activationFunction, activationScale, netDepth, netWidth, w, i) != vec(testNewIterateNetwork(activationFunction, activationScale, netDepth, netWidth, w, i))
                return false
            end
        end
    end
    return true
end

function oldMeasureNetwork(activationFunction, activationScale, polynomialDegree, network)
    x = 0
    for i in -1:0.02:1
        LayerOutputs = zeros(Float64, size(network[2])) ## size of the bias vector
        # print((last(iterateNetwork(activationFunction, activationScale, i, network, LayerOutputs)) - Pl(i, polynomialDegree) ))
        N_i = last(oldIterateNetwork(activationFunction, activationScale, i, network, LayerOutputs)) 
        R_i = PlNormalized(i, polynomialDegree, 0, 1)
        x += abs(N_i - R_i)
    end
    return x
end

function testMeasureNetwork(activationFunction, activationScale, polyDegree)
    for w in weightSamples
        W_m = fill(w, (netDepth,netDepth))
        W_b = fill(w, netDepth)
        old = oldMeasureNetwork(activationFunction, activationScale, polyDegree, [W_m, W_b] )
        new = measureNetwork(activationFunction, activationScale, polyDegree, generateFilledNetwork(netDepth, netWidth, w))
        if old != new
        # print("Weight : $w  ||  Old : $old  ||  New : $new \n")
            return false
        end
    end
    return true
end

activationFunction = (f(x) = (1 - exp(-x^2))) 
activationScale = 1.0
netDepth = 5
netWidth = 1
netWeights = 0.0
polyDegree = 3

## Dummy variables
testInput = 10.0
calcLayer = 1
calcNode = 1
inputRange = -1:0.2:1 ## just using the step function from Le Nagard
weightSamples = collect(randn(100))


## these checks should come back the same regardless of network dimension
samples = 10 ## Showing that a random sample of weights will return the same value in both cases
testOldCalcOj(activationFunction, activationScale, netDepth, netWeights, testInput, calcLayer) == testNewCalcOj(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput, calcLayer, calcNode)
testOldIterateNetwork(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput)[netDepth] == testNewIterateNetwork(activationFunction, activationScale, netDepth, netWidth, netWeights, testInput)[netDepth, netWidth]
compareCalcOj(activationFunction, activationScale, netDepth, netWidth, calcLayer, calcNode, collect(randn((samples))), collect(randn((samples))))


## These checks should come back true if the networks are of width = 1, but false otherwise
compareIterateNetwork(activationFunction, activationScale, netDepth, netWidth, collect(randn((samples))), collect(randn((samples))))
testMeasureNetwork(activationFunction, activationScale, polyDegree)