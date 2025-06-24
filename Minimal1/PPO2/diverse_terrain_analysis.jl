using LinearAlgebra
using IntervalSets
using StableRNGs
using SparseArrays
using FFTW
using PlotlyJS
using FileIO, JLD2
using Flux
using Random
using RL
using DataFrames
using Statistics
using JuMP
using Ipopt
using Optimisers
#using Blink


dim = 20
batch_size = 5
fun = gelu

# Create a deeper network with larger intermediate layers
a = Chain(
    Dense(2, dim, fun),
    Dense(dim, dim, fun),
    Dense(dim, 1)
)

# Use a higher learning rate to find more complex patterns
opt_state = Flux.setup(Optimisers.Adam(7e-3), a)

training_set_size = 70

input = randn(Float32, 2, training_set_size)
output = randn(Float32, 1, training_set_size)

for i in 1:10_000
    # create random batch indices
    rand_inds = shuffle!(Vector(1:training_set_size))
    batch_inds = rand_inds[1:batch_size]

    g = Flux.gradient(a) do aa
        Flux.mse(aa(input[:,batch_inds]), output[:,batch_inds])
    end

    Flux.update!(opt_state, a, g[1])
end


xx = collect(-2:0.05:2)
yy = xx  # Using the same range for y axis

# Create the grid points
xy_input = zeros(Float32, 2, length(xx), length(yy))
for (i, x) in enumerate(xx), (j, y) in enumerate(yy)
    xy_input[:, i, j] = [x, y]
end

z_data = a(xy_input)

p = plot(surface(x=xx, y=yy, z=z_data[1, :, :], colorscale="Viridis", showscale=false))
add_trace!(p, scatter3d(x=input[1,:], y=input[2,:], z=output[:], mode="markers",
    marker=attr(size=2, color="red", opacity=0.8)))
display(p)