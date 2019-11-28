using Makie
using Images
using LinearAlgebra
using ImageView
using Revise
using Debugger
using VideoIO
using ProgressMeter

# Create Gaussian which can modify the Images
function gaussianVal(μ, σ, p)
    return exp(-(((p[1] - μ[1])^2)/(2*σ[1]^2) + ((p[2] - μ[2])^2)/(2*σ[2]^2)))
end

function gaussianPaint(img, μ, σ)
    a = copy(img)
    s = size(a)
    for i in ((_i, _j) for _i in 1:s[1] for _j in 1:s[2])
        # if sqrt((μ[1] - i[1])^2 + (μ[2] - i[2])^2) < 10*σ
        if (σ[1] > abs(μ[1] - i[1])/10) && (σ[2] > abs(μ[2] - i[2])/10)
            val = gaussianVal(μ, σ, i)
            a[i[1], i[2]] = val
        end
    end
    return a
end

function gaussianNoisePaint(img, μ, σ)
    a = copy(img)
    s = size(a)
    for i in ((_i, _j) for _i in 1:s[1] for _j in 1:s[2])
        # if sqrt((μ[1] - i[1])^2 + (μ[2] - i[2])^2) < 10*σ
        if (σ[1] > abs(μ[1] - i[1])/10) && (σ[2] > abs(μ[2] - i[2])/10)
            val = (gaussianVal(μ, σ, i) + 0.3*rand())/2
            a[i[1], i[2]] = val
        end
    end
    return a
end


function square_cont(img, location, width)
    a = copy(img)
    s = size(a)
    for i in ((_i, _j) for _i in 1:s[1] for _j in 1:s[2])
        if (width[1] > abs(location[1] - i[1])) && (width[2] > abs(location[2] - i[2]))
            a[i[1], i[2]] = 1.0
        end
    end
    return a
end

function circle_cont(img, location, radius)
    a = copy(img)
    s = size(a)
    for i in ((_i, _j) for _i in 1:s[1] for _j in 1:s[2])
        if sqrt((location[1] - i[1])^2 + (location[2] - i[2])^2) < radius
            a[i[1], i[2]] = 1.0
        end
    end
    return a
end

mutable struct GaussianPaint
    img
    func
    μ
    σ
end

mutable struct GaussianNoisePaint
    img
    func
    μ
    σ
end

mutable struct SquarePaint
    img
    func
    location
    width
end

mutable struct CirclePaint
    img
    func
    location
    radius
end

struct Chain
    painters
end

Base.iterate(s::Chain, state=0) = begin
    img = nothing;
    for (idx, painter) in enumerate(s.painters)
        next_state = copy(state)
        try
            if idx < length(s.painters)
                (img, _next_state) = iterate(painter, state)
                nextpainter = s.painters[idx + 1]
                nextpainter.img = img
            else
                (img, next_state) = iterate(painter, state)
            end
        catch LoadError
            return nothing
        end
        state = next_state
    end
    # @show state
    if state ≥ 100.0
        return nothing
    end
    return (img, state)
end

Base.iterate(s::SquarePaint, state=0.01) = begin
    width = s.func(state)
    img = square_cont(s.img, s.location, (width, width))
    state += 0.1
    if state ≥ 100.0
        return nothing
    end
    return (img, state)
end

Base.iterate(s::CirclePaint, state=0.01) = begin
    radius = s.func(state)
    img = circle_cont(s.img, s.location, radius)
    state += 0.1
    if state ≥ 100.0
        return nothing
    end
    return (img, state)
end

Base.iterate(s::GaussianPaint, state=0.01) = begin
    σ = s.func(state)
    img = gaussianPaint(s.img, s.μ, (σ, σ))
    state += 0.1
    if state ≥ 100.0
        return nothing
    end
    return (img, state)
end

Base.iterate(sn::GaussianNoisePaint, state=0.01) = begin
    σ = sn.func(state)
    img = gaussianNoisePaint(sn.img, sn.μ, (σ, σ))
    state += 0.1
    if state ≥ 100.0
        return nothing
    end
    return (img, state)
end

function synthesizeData()
    framerate = 24.0;
    props = [:priv_data => ("crf"=>"23","preset"=>"ultrafast")]
    a = zeros(600, 600)
    encoder = prepareencoder(RGB{N0f8}.(Gray.(a)), framerate=framerate, AVCodecContextProperties=props, codec_name="libx264")
    filename = "gendataNoise.mp4";
    io = Base.open("temp.stream","w")
    p = Progress(1000, 1)

    # square_paint = SquarePaint(a, (x) -> 10*(1.0 + cos(2*x + 0.3) + sin(2*x)), (100, 100), (10, 10))
    # circle_paint = CirclePaint(square_paint.img, (x) -> 10*(1.0 + sin(4*x)), (400, 400), 10)
    gaussian_paint = GaussianNoisePaint(a, (x) -> 10*(3.0 +  0.5*cos(x + 10.0) + sin(x)), (400, 100), (1, 1))

    painters = Chain([gaussian_paint,])

    for (idx, i) in enumerate(painters)
        appendencode!(encoder, io, RGB{N0f8}.(Gray.(i)), idx)
        next!(p)
    end

    finishencode!(encoder, io)
    println("Done with encoding");
    close(io)
    mux("temp.stream",filename,framerate)
end

# if PROGRAM_FILE == @__FILE__
synthesizeData()

#    @enter synthesizeData()
