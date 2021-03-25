using Distributed

@everywhere include("zygote_bug_mwe_def.jl")

function update!(opt, xs::Zygote.Params, gs::Dict{Int32, AbstractArray})
    for (k, v) in gs
        Flux.Optimise.update!(opt, xs[k], v)
    end
end

function accumulateGradients(gradsList::Vector{Dict{Int32, AbstractArray}})::Dict{Int32, AbstractArray}
    grads = Dict{Int32, AbstractArray}()
    for liteGrads in gradsList
        for (k, v) in liteGrads
            if haskey(grads, k)
                grads[k] += v
            else
                grads[k] = v
            end
        end
    end
    return grads
end

function main()
    # simulate creation of training data
    nDocs = 20
    docSizes = rand(40:100, nDocs)
    K₁ = 10 # the feature vector size of each word in the document
    K₂ = 5 # hidden feature vector size
    data = map(docSizes) do n # n represents the size of this document
        X = map(x -> rand(Float32, K₁), 1:n) # this generates the words of the document
        y = rand(Float32)
        return X, y
    end

    # create the neural network here
    numUnaryRules  = 10
    numBinaryRules = 20
    unaryRules  = map(x -> UnaryRule(K₁, K₂), 1:numUnaryRules)
    binaryRules = map(x -> BinaryRule(K₂), 1:numBinaryRules)

    # dynamic programming to create the binary trees
    

    opt = Flux.Optimise.ADAM(1e-3, (0.9, 0.999))
    while true
        outputs = let unaryRules=unaryRules, binaryRules=binaryRules
            pmap(data) do Xy
                X, y = Xy # X is an array of vectors, y is a single floating point
                _, tree = buildTree(X, 1, length(X), unaryRules, binaryRules)

                θ = params(unaryRules, binaryRules)
                grads = gradient(θ) do
                    ŷ, _ = forward(tree, unaryRules, binaryRules)
                    ŷ = ŷ |> first
                    (ŷ - y)^2
                end

                liteGrads = liteGradients(grads)
            end
        end

        grads = accumulateGradients(outputs)
        θ = params(unaryRules, binaryRules)
        update!(opt, θ, grads)
    end
end

main()
