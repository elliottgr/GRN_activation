## Debug file Checking that the calcOj function returns the same results using the old and new versions
include("networksFuncs.jl") ## Importing current version

function old_calcOj(activation_function::Function, activation_scale::Float64, j::Int, prev_out, Wm, Wb)
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
    old_calcOj(activation_function, activation_scale, calcLayer, prev_out, W_m, W_b)
end
function testNewCalcOj(activation_function::Function, activation_scale::Float64, netDepth = 2, netWidth = 1, netWeights = 0.0, testInput = 0.0, calcLayer = 1, calcNode = 1)
    W_m, W_b = generateFilledNetwork(netDepth, netWidth, netWeights)
    activationMatrix = zeros(Float64, netWidth)
    testInputVector = zeros(Float64, netWidth)
    testInputVector[1] = testInput
    activationMatrix[1, 1:netWidth] = testInputVector
    calcOj(activation_function, activation_scale, calcNode, calcLayer, activationMatrix, W_m, W_b)
end

Φ = (f(x) = (1 - exp(-x^2))) 
α = 1.0
netDepth = 5
netWidth = 1
netWeights = 0.0
testInput = 0.0
calcLayer = 1
calcNode = 1
testOldCalcOj(Φ, α, netDepth, netWeights, testInput, calcLayer)
testNewCalcOj(Φ, α, netDepth, netWidth, netWeights, testInput, calcLayer, calcNode)
