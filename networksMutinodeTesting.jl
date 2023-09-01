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


## Takes the generalized version of the networks used here and converts it back
## to the old linear version. Using the prefixes wide and narrow to keep things clear in this function
## if the wideNetwork has width >1, it merely takes the top-most node in each layer to generate the new network
function generateOldNetwork(wideNetwork)
    wideW_m, wideW_b = wideNetwork
    netDepth, netWidth = size(wideW_b)

    ## need to unpack new W_m format into old format
    narrowW_m = zeros(Float64, (netDepth, netDepth))
    narrowW_b = zeros(Float64, netDepth)
    for layer in 1:netDepth
        narrowW_b[layer] = wideW_b[layer][1] ## taking the top node in each layer
        for iterLayer in 1:netDepth
            narrowW_m[iterLayer, layer] = wideW_m[iterLayer][layer]
        end
    end
    return [narrowW_m, narrowW_b]
end

function testOldCalcOj(activationFunction::Function, activationScale::Float64, network, testInput = 0.0, calcLayer = 1)
    W_m, W_b = network
    netDepth = size(W_b)
    ## need to unpack new W_m format into old format
    prev_out = fill(0.0, netDepth) ## blank N node input
    prev_out[1] = testInput ## probably a way to do this in one line lol
    oldCalcOj(activationFunction, activationScale, calcLayer, prev_out, W_m, W_b)
end

function testNewCalcOj(activationFunction::Function, activationScale::Float64, network, testInput = 0.0, calcLayer = 1, calcNode = 1)
    W_m, W_b = network
    netDepth, netWidth = size(W_b)
    activationMatrix = zeros(Float64, (netDepth, netWidth))
    testInputVector = zeros(Float64, netWidth)
    testInputVector[1] = testInput
    activationMatrix[1, 1:netWidth] = testInputVector
    print(activationMatrix)
    calcOj(activationFunction, activationScale, calcNode, calcLayer, activationMatrix, W_m, W_b)
end


## testing calcOj over a range of weights
function compareCalcOj(activationFunction, activationScale, netDepth, netWidth, calcNode, inputRange, val)
    # generating randomized networks and reshaping them to see when there will be a difference
    if typeof(val) == Float64
        newNetwork = generateFilledNetwork(netDepth, netWidth, val)
    else
        newNetwork = generateNetwork(netDepth, netWidth)
    end
    oldNetwork = generateOldNetwork(newNetwork)
    print(newNetwork, "\n")
    print(oldNetwork, "\n")
    for layer in 1:netDepth
        for input in inputRange
            oldCalcOjVal = testOldCalcOj(activationFunction, activationScale, oldNetwork, input, layer)
            newCalcOjVal= testNewCalcOj(activationFunction, activationScale, newNetwork, input, layer, calcNode)
            # if oldCalcOjVal != oldCalcOjVal 
                print("Layer : $layer  ||  Input : $input \n  Old : $oldCalcOjVal  || New : $newCalcOjVal \n")
                # return false
            # end 
        end   
    end
    return true
end

netDepth = 2
netWidth = 1
compareCalcOj(activationFunction, activationScale, netDepth, netWidth, calcNode, [1], "no")

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


function testOldIterateNetwork(activationFunction::Function, activationScale, network, testInput)
    prev_out = fill(0.0, netDepth) ## blank N node input
    prev_out[1] = testInput ## probably a way to do this in one line lol
    oldIterateNetwork(activationFunction, activationScale, testInput, generateOldNetwork(network), prev_out)
end

function testNewIterateNetwork(activationFunction, activationScale, network, testInput)
    # W_m, W_b = generateFilledNetwork(netDepth, netWidth, netWeights)
    activationMatrix = zeros(Float64, (netDepth, netWidth))
    testInputVector = zeros(Float64, netWidth)
    testInputVector[1] = testInput
    activationMatrix[1, 1:netWidth] = testInputVector
    iterateNetwork(activationFunction, activationScale, testInputVector, network, activationMatrix)
end

function compareIterateNetwork(activationFunction, activationScale, netDepth, netWidth, val, inputRange)
    if typeof(val) == Float64
        network = generateFilledNetwork(netDepth, netWidth, val)
    else
        network = generateNetwork(netDepth, netWidth)
    end
    for i in inputRange
        oldIterateNetworkVal = testOldIterateNetwork(activationFunction, activationScale, network, i)
        newIterateNetworkVal = vec(testNewIterateNetwork(activationFunction, activationScale, network, i))
        print("Input : $i  ||  New : $newIterateNetworkVal  ||  Old : $oldIterateNetworkVal \n")
        if oldIterateNetworkVal != newIterateNetworkVal
            return false
        end
    end
    return true
end

compareIterateNetwork(activationFunction, activationScale, netDepth, netWidth, "f", collect(randn((samples))))

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


function generateResponseCurves(activationFunction, activationScale, netDepth, netWidth, curves = 5)
    plt = plot()
    inputRange = -1:0.02:1
    for _ in 1:curves
        newNetVals = []
        oldNetVals = []
        newNetwork = generateNetwork(netDepth, netWidth)
        oldNetwork = generateOldNetwork(newNetwork)
        for i in inputRange
            LayerOutputs = zeros(Float64, (netDepth, netWidth)) ## size of the bias vector
            input = fill(0.0, netWidth)
            input[1] = i
            prevOut = fill(0.0, netDepth)
            push!(newNetVals, last(iterateNetwork(activationFunction, activationScale, input, newNetwork, LayerOutputs)[netDepth]))
            push!(oldNetVals, last(oldIterateNetwork(activationFunction, activationScale, i, oldNetwork, prevOut)))
        end
        plot!(inputRange, newNetVals)
        plot!(inputRange, oldNetVals, linestyle = :dash)
        # end
    end
    return plt
end

generateResponseCurves(activationFunction, activationScale, netDepth, netWidth, 5)
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
compareIterateNetwork(activationFunction, activationScale, netDepth, netWidth, "f", collect(randn((samples))))
testMeasureNetwork(activationFunction, activationScale, polyDegree)