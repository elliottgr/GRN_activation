using JLD2, StatsPlots, DataFrames
include("Networks.jl")

function loadSimulationResults(path = pwd())
    simulationData = []
    for file in readdir(path)
        if splitext(file)[2] == ".jld2"
            simulationFile = JLD2.load(string(path, file))["simulationOutputs"]
            simulationFile.filename = fill(file, length(simulationFile.fitness))
            push!(simulationData, simulationFile)
            print("Sucessfully loaded: $file with length $(length(simulationFile.fitness)) entries of $(length(unique(simulationFile[simulationFile.T .== maximum(simulationFile.T), :].replicateID))) replicates \n")
        end
    end
    return vcat(simulationData..., cols = :union)
end

df = loadSimulationResults("/better_scratch/elliott/GRN_activation/")
filenames = ["LogisticExtraTests2023-11-15T15:50:16.076.jld2",
            "LogisticTests2023-11-03T14:42:05.882.jld2",
            "InverseGaussianTests2023-11-03T14:39:55.714.jld2",
            "InvExpExtraTests2023-11-15T15:52:03.360.jld2"]
df = df[in.(df.filename, Ref(filenames)), :]

## making a LaTeX table :)
## defining groups
envChallenge = 3
## (depth, width)

networkSizes = [(3,1), (4,1), (4,4)] ## x axis
αs = [0.5, 1.0, 2.0]
Φs = ["LeNagardExp", "Logistic"]

## selecting correct datapoints
testdf = copy(df)
testdf.netSize = tuple.(testdf[!, :netDepth], testdf[!, :netWidth])
testdf = testdf[(testdf.regulationDepth .== maximum(testdf.regulationDepth)) .& (testdf.T .== maximum(testdf.T)) .& (testdf.envChallenge .== envChallenge), :]
testdf = testdf[(in.(testdf.netSize, Ref(networkSizes))) .& in.(testdf.α, Ref(αs)), :]

## generating final dataframe, truncating fitness for table display
testgroups = groupby(testdf, [:netSize, :activationFunction,  :α])
testcombine = combine(testgroups, :fitness=>mean)
testcombine.fitness_mean = round.(testcombine[:, :fitness_mean], digits = 3)
