using Flux
using Zygote
using Random

const rng = MersenneTwister()

glorot_uniform(dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(2.0f-2/sum(dims))
glorot_normal(dims...)  = randn(rng, Float32, dims...) .* sqrt(2.0f-2/sum(dims))
zeroFloats(dims...)     = zeros(Float32, dims)

struct TreeConvBlock{A,V}
    W₁::A # left 
    W₂::A # right
    b₀::V 
end
Flux.@functor TreeConvBlock

function TreeConvBlock(i::Integer, o::Integer; init=glorot_uniform)
    return TreeConvBlock(init(o, i), init(o, i), zeroFloats(o))
end

function (block::TreeConvBlock)(xLeft::AbstractArray, xRight::AbstractArray)
	W₁, W₂, b₀ = block.W₁, block.W₂, block.b₀
    h = gelu.(W₁ * xLeft .+ W₂ * xRight .+ b₀)
    return h
end

abstract type RuleNode end

struct UnaryRuleNode <: RuleNode
    i::Int
    x::Vector{Float64}
end

struct BinaryRuleNode <: RuleNode
    i::Int
    left::RuleNode
    right::RuleNode
end

abstract type Rule end

struct UnaryRule <: Rule
    fc₀::Dense
    fc₁::Dense # just for the score, need to keep this fixed

    UnaryRule(K₁::Int, K₂::Int) = new(
        Dense(K₁, K₂, gelu; initW=glorot_uniform),
        Dense(K₂, 1; initW=glorot_normal)
    )
end
Flux.@functor UnaryRule

function (rule::UnaryRule)(x::AbstractArray)
    h = rule.fc₀(x)
    y = rule.fc₁(h)
    return y, h
end

struct BinaryRule <: Rule
    m₀::TreeConvBlock
    m₁::Dense

    BinaryRule(K::Int) = new(
        TreeConvBlock(K, K),
        Dense(K, 1; initW=glorot_normal)
    )
end
Flux.@functor BinaryRule

function (rule::BinaryRule)(h₁::AbstractArray, h₂::AbstractArray)
    h = rule.m₀(h₁, h₂)
    y = rule.m₁(h)
    return y, h
end

function buildTree(X::Vector{Vector{Float32}}, i::Int, j::Int, 
    unaryRules::Vector{UnaryRule}, binaryRules::Vector{BinaryRule})::Tuple{Vector{Float32}, RuleNode}
    # base case
    if i == j
        x = X[i] # base vector
        k = rand(1:length(unaryRules))
        f = unaryRules[k]
        _, h = f(x)
        return h, UnaryRuleNode(k, x)
    else
        # there is a left and right component here

        # choose a random partition between i and j
        k = rand(i:j-1)
        hl, left = buildTree(X, i, k, unaryRules, binaryRules)
        hr, right = buildTree(X, k+1, j, unaryRules, binaryRules)

        # choose a random binaryRules
        k_ = rand(1:length(binaryRules))
        f = binaryRules[k_]
        _, h = f(hl, hr)
        return h, BinaryRuleNode(k_, left, right)
    end
end

function forward(ruleNode::UnaryRuleNode, unaryRules::Vector{UnaryRule}, binaryRules::Vector{BinaryRule})
    f = unaryRules[ruleNode.i]
    y, h = f(ruleNode.x)
    return y, h
end

function forward(ruleNode::BinaryRuleNode, unaryRules::Vector{UnaryRule}, binaryRules::Vector{BinaryRule})
    yl, hl = forward(ruleNode.left, unaryRules, binaryRules)
    yr, hr = forward(ruleNode.right, unaryRules, binaryRules)
    
    f = binaryRules[ruleNode.i]
    y, h = f(hl, hr)
    return y + yl + yr, h
end

function liteGradients(grads::Zygote.Grads)::Dict{Int32, AbstractArray}
    idxToGradients = Dict{Int32, AbstractArray}()
    for (i, p) in enumerate(grads.params)
        if grads[p] !== nothing
            idxToGradients[i] = grads[p]
        end
    end
    return idxToGradients
end
