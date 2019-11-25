module mobilenet_ssdlite

using Flux;
using Flux: @functor

# macro stride1_residual_block() end
# macro stride2_residual_block() end

function inverted_residual_block(size, s, t, k, k′)
    m = Chain(
        Conv((1, 1), k=>t*k, relu, pad=(1,1)),
        DepthwiseConv(size, t*k=>t*k, relu; stride=(s, s), pad=(1,1), groupcount=t*k),
        Conv((1,1), t*k=>k′)) |> gpu
    if s == 2
        sm = SkipConnection(m, (mx, x) -> mx + x) |> gpu
        return sm
    elseif s==1
        return m
    end
end

struct InvertedResidual
    size
    s
    t
    k
    k′
end

(f::InvertedResidual)(x) = inverted_residual_block(f.size, f.s, f.t, f.k, f.k′)(x)


# function stride1_block(x, size, s, t, k, k′)
#     y = ((@residual_block size s t k k′)...)(x)
#     y += x
# end

# function stride2_block(x, size, s, t, k, k′)
#
# end

macro repeat(n, ex)
    a = eval(ex)
    for i in 1:n-1
        a = a, eval(ex)
    end
    return a
end

@functor InvertedResidual

model = Chain(
    Conv((3,3), 3=>32, relu) |> gpu,
    InvertedResidual((3,3), 1, 1, 32, 16),
    # residual_block((3,3), 1, 1, 32, 16),
    # residual_block((3,3), 2, 6, 16, 24),
    # residual_block((3,3), 1, 6, 24, 24),
    # residual_block((3,3), 2, 6, 24, 32),
    # residual_block((3,3), 1, 6, 32, 32),
    # residual_block((3,3), 1, 6, 32, 32),
    # residual_block((3,3), 2, 6, 32, 64),
    # residual_block((3,3), 1, 6, 64, 64),
    # residual_block((3,3), 1, 6, 64, 64),
    # residual_block((3,3), 1, 6, 64, 64),
    # residual_block((3,3), 1, 6, 64, 96),
    # residual_block((3,3), 1, 6, 96, 96),
    # residual_block((3,3), 1, 6, 96, 96),
    # residual_block((3,3), 2, 6, 96, 160),
    # residual_block((3,3), 1, 1, 160, 160),
    # residual_block((3,3), 1, 1, 160, 160),
    # residual_block((3,3), 1, 1, 160, 320),
    Conv((1,1), 16=>12),
    MeanPool((7,7)),
    Conv((1,1), 12=>1)
)

x = rand(224, 224, 3, 1) |> gpu

y = model(x)

end # module
