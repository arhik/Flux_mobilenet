module mobilenetv1_ssdlite

using Flux;
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Statistics: mean
using MLDatasets: CIFAR10


model = Chain(
    Conv((3,3), 3=>32, selu, pad=(1,1), stride=1),
    BatchNorm(32, relu),
    DepthwiseConv((3,3), 32=>32, groupcount=32, selu),
    BatchNorm(32, relu),
    Conv((1,1), 32=>64, selu),
    BatchNorm(64, relu),
    DepthwiseConv((3,3), 64=>64, groupcount=64, selu),
    BatchNorm(64, relu),
    Conv((1,1), 64=>128, selu),
    BatchNorm(128, relu),
    DepthwiseConv((3,3), 128=>128, groupcount=128, selu),
    BatchNorm(128, relu),
    Conv((1,1), 128=>128, selu),
    BatchNorm(128, relu),
    DepthwiseConv((3,3), 128=>128, groupcount=128, selu),
    BatchNorm(128, relu),
    Conv((1,1), 128=>256, selu),
    BatchNorm(256, relu),
    DepthwiseConv((3,3), 256=>256, groupcount=256, selu),
    BatchNorm(256, relu),
    Conv((1,1), 256=>256, selu),
    BatchNorm(256, relu),
    DepthwiseConv((3,3), 256=>256, groupcount=256, stride=1, selu),
    BatchNorm(256, relu),
    Conv((1,1), 256=>512, selu),
    BatchNorm(512, relu),
    DepthwiseConv((3,3), 512=>512, groupcount=512, stride=1, selu),
    BatchNorm(512, relu),
    Conv((1,1), 512=>512, selu),
    BatchNorm(512, relu),
    DepthwiseConv((3,3), 512=>512, groupcount=512, selu),
    BatchNorm(512, relu),
    Conv((1,1), 512=>512, selu),
    BatchNorm(512, relu),
    DepthwiseConv((3,3), 512=>512, groupcount=512, selu),
    BatchNorm(512, relu),
    Conv((1,1), 512=>512, selu),
    BatchNorm(512, relu),
    DepthwiseConv((3,3), 512=>512, groupcount=512, selu),
    BatchNorm(512, relu),
    Conv((1,1), 512=>512, selu),
    BatchNorm(512, relu),
    DepthwiseConv((3,3), 512=>512, groupcount=512, selu),
    BatchNorm(512, relu),
    Conv((1,1), 512=>512, selu),
    BatchNorm(512, relu),
    DepthwiseConv((3,3), 512=>512, groupcount=512, selu),
    BatchNorm(512, relu),
    Conv((1,1), 512=>1024, selu),
    BatchNorm(1024, relu),
    DepthwiseConv((3,3), 1024=>1024, groupcount=1024, selu),
    BatchNorm(1024, relu),
    Conv((1,1), 1024=>1024, selu),
    BatchNorm(1024, relu),
    MeanPool((4,4)),
    Conv((1,1),1024=>10, selu),
    BatchNorm(10, relu),
    x -> reshape(x, :, size(x, 4)),
    softmax
) |> gpu



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

batchsize=20

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
    return ((cat(x_batch..., dims=4) |> gpu, cat(y_batch..., dims=2) |> gpu), state+1)
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
    return ((cat(x_batch..., dims=4) |> gpu, cat(y_batch..., dims=2) |> gpu), state+1)
end

Base.iterate(t::TestIter, state=1) = begin
    x_batch = []
    y_batch = []
    global batchsize
    for b in 1:batchsize
        i = rand(1:10000)
        if state > t.count
            return nothing
        end
        push!(x_batch, reshape(CIFAR10.convert2features(CIFAR10.testtensor(Float32, i)), (32,32,3)))
        push!(y_batch, Flux.onehot(CIFAR10.testlabels(i), 0:9))
    end
    return ((cat(x_batch..., dims=4) |> gpu, cat(y_batch..., dims=2) |> gpu), state+1)
end

trainiter = TrainIter(1000)
validiter = ValidIter(1000)
testiter = TestIter(1000)

# X = trainimgs(CIFAR10)
# imgs = [getarray(X[i].img) for i in 1:50000]
# labels = onehotbatch([X[i].ground_truth.class for i in 1:5000],1:10)

# loss(x, y) = crossentropy(reshape(model(x), (10,batchsize)), reshape(y, (10, batchsize)))
loss(x, y) = crossentropy(model(x), y)


# accuracy(x, y) = mean(onecold(reshape(model(x), (10, batchsize)), 1:10) .== onecold(reshape(y, (10, batchsize)), 1:10))
accuracy(x, y) = mean(onecold(model(x), 1:10) .== onecold(y, 1:10))

# Defining the callback and the optimize
Loss = 0.0

evalcb = throttle(() -> begin
        global Loss
        i = collect(Iterators.take(testiter, 1))
        acc = accuracy(i[1]...)
        Loss = (Loss+loss(i[1]...))/2
        @show(Loss, acc)
    end, 10)

opt = ADAM(1e-3)

# for i in trainiter
#     @show size(i[1]), size(i[2])
#     # @show model(i[1])
#     break
# end


# Starting to train models
using Flux: @epochs
# using Debugger
@epochs 100 Flux.train!(loss, params(model), trainiter, opt, cb=evalcb)

using BSON: @save

@save "mobilenetv1.bson" model
# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
# Print the final accuracy

@show(accuracy(collect(Iterators.take(testiter, 100))[1]...))

end # module
