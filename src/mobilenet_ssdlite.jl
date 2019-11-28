module mobilenet_ssdlite

using Flux;
using Flux: @functor

using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using MLDatasets: CIFAR10
# macro stride1_residual_block() end
# macro stride2_residual_block() end

function inverted_residual_block(ksize, s, t, k, k′)
    m = Chain(
        Conv((1, 1), k=>t*k, relu),
        DepthwiseConv(ksize, t*k=>t*k, relu; stride=(1, 1), pad=(length(ksize)-1,length(ksize)-1), groupcount=t*k),
        Conv((1,1), t*k=>k′)) |> gpu
    # if s==1 & k == k′
    #     sm = SkipConnection(m, (mx, x) -> mx + x) |> gpu
    #     return sm
    # end
    return m
end


struct InvertedResidual
    size
    s
    t
    k
    k′
end

(f::InvertedResidual)(x) = inverted_residual_block(f.size, f.s, f.t, f.k, f.k′)(x)

macro repeat(n, ex)
    a = eval(ex)
    for i in 1:n-1
        a = a, eval(ex)
    end
    return a
end

@functor InvertedResidual

model = Chain(
    Conv((3,3), 3=>32, relu, pad=1, stride=2) |> gpu,
    InvertedResidual((3,3), 1, 1, 32, 16),
    InvertedResidual((3,3), 2, 6, 16, 24),
    # InvertedResidual((3,3), 1, 6, 24, 24),
    InvertedResidual((3,3), 2, 6, 24, 32),
    # InvertedResidual((3,3), 1, 6, 32, 32),
    # InvertedResidual((3,3), 1, 6, 32, 32),
    InvertedResidual((3,3), 2, 6, 32, 64),
    # InvertedResidual((3,3), 1, 6, 64, 64),
    # InvertedResidual((3,3), 1, 6, 64, 64),
    # InvertedResidual((3,3), 1, 6, 64, 64),
    InvertedResidual((3,3), 1, 6, 64, 96),
    # InvertedResidual((3,3), 1, 6, 96, 96),
    # InvertedResidual((3,3), 1, 6, 96, 96),
    InvertedResidual((3,3), 2, 6, 96, 160),
    # InvertedResidual((3,3), 1, 1, 160, 160),
    # InvertedResidual((3,3), 1, 1, 160, 160),
    InvertedResidual((3,3), 1, 1, 160, 320),
    Conv((1,1), 320=>1280) |> gpu,
    MeanPool((14,14)) |> gpu,
    Conv((1,1), 1280=>10) |> gpu,
    softmax |> gpu
)



struct TrainIter
    count
end

Base.length(b::TrainIter) = b.count

struct ValidIter
    count
end

Base.length(b::ValidIter) = b.count

struct TestIter
    count
end

Base.length(b::TestIter) = b.count

batchsize=5

Base.iterate(t::TrainIter, state=1) = begin
    x_batch = []
    y_batch = []
    global batchsize
    for b in 1:batchsize
        i = rand(1:40000)
        if state > t.count
            return nothing
        end
        push!(x_batch, reshape(CIFAR10.convert2features(CIFAR10.traintensor(Float32, i)), (32,32,3,1)))
        push!(y_batch, Flux.onehot(CIFAR10.trainlabels(i), 0:9))
    end
    return ((cat(x_batch..., dims=4) |> gpu, cat(y_batch..., dims=4) |> gpu), state+1)
end

Base.iterate(t::ValidIter, state=1) = begin
    x_batch = []
    y_batch = []
    global batchsize
    for b in 1:batchsize
        i = rand(40001:50000)
        if state > t.count
            return nothing
        end
        push!(x_batch, reshape(CIFAR10.convert2features(CIFAR10.traintensor(Float32, i)), (32,32,3,1)))
        push!(y_batch, Flux.onehot(CIFAR10.trainlabels(i), 0:9))
    end
    return ((cat(x_batch..., dims=4) |> gpu, cat(y_batch..., dims=4) |> gpu), state+1)
end

Base.iterate(t::TestIter, state=1) = begin
    x_batch = []
    y_batch = []
    global batchsize
    for b in 1:batchsize
        i = rand(40001:50000)
        if state > t.count
            return nothing
        end
        push!(x_batch, reshape(CIFAR10.convert2features(CIFAR10.testtensor(Float32, i)), (32,32,3)))
        push!(y_batch, Flux.onehot(MLDataset.CIFAR10.testlabels(i), 0:9))
    end
    return ((cat(x_batch..., dims=4) |> gpu, cat(y_batch..., dims=4) |> gpu), state+1)
end

trainiter = TrainIter(100)
validiter = ValidIter(100)
testiter = TestIter(100)

# X = trainimgs(CIFAR10)
# imgs = [getarray(X[i].img) for i in 1:50000]
# labels = onehotbatch([X[i].ground_truth.class for i in 1:5000],1:10)

loss(x, y) = crossentropy(reshape(model(x), (10,batchsize)), reshape(y, (10, batchsize)))

accuracy(x, y) = mean(onecold(reshape(model(x), (:, batchsize)), 1:10) .== onecold(reshape(y, (:, batchsize)), 1:10))

# Defining the callback and the optimizer

evalcb = throttle(() -> @show(accuracy(take(validiter, 1))), 10)

opt = ADAM()

for i in trainiter
    @show size(i[1]), size(i[2])
    # @show model(i[1])
    break
end

# Starting to train models

Flux.train!(loss, params(model), trainiter, opt, cb=evalcb)

# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs


testimgs = [getarray(test[i].img) for i in 1:10000]
testY = onehotbatch([test[i].ground_truth.class for i in 1:10000], 1:10) |> gpu
testX = cat(testimgs..., dims = 4) |> gpu

# Print the final accuracy

@show(accuracy(testX, testY))

end # module
