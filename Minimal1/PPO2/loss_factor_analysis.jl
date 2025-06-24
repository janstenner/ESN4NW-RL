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
using Zygote:ignore
#using Blink


dim = 20
batch_size = 5
fun = gelu


a = Chain(
    Dense(2, dim, fun),
    Dense(dim, dim, fun),
    Dense(dim, 1)
)

b = Chain(
    Dense(2, dim, fun),
    Dense(dim, dim, fun),
    Dense(dim, 1)
)

# Use a higher learning rate to find more complex patterns
opt_state = Flux.setup(Optimisers.Adam(7e-3), a)
opt_state_2 = Flux.setup(Optimisers.Adam(7e-3), b)

training_set_size = 10
training_set_size_2 = 10

input = randn(Float32, 2, training_set_size)
output = randn(Float32, 1, training_set_size)
input_2 = randn(Float32, 2, training_set_size_2)
output_2 = randn(Float32, 1, training_set_size_2)

losses1 = Float32[]
losses2 = Float32[]

for i in 1:1_000
    # create random batch indices
    rand_inds = shuffle!(Vector(1:training_set_size))
    batch_inds = rand_inds[1:batch_size]
    
    rand_inds_2 = shuffle!(Vector(1:training_set_size_2))
    batch_inds_2 = rand_inds_2[1:batch_size]

    ga, gb = Flux.gradient(a, b) do aa, bb

        loss1 = Flux.mse(aa(input[:,batch_inds]), output[:,batch_inds])
        loss2 = Flux.mse(bb(input_2[:,batch_inds_2]), output_2[:,batch_inds_2])


        ignore() do 
            push!(losses1, loss1)
            push!(losses2, loss2)
        end

        loss = 20000 * loss1 + loss2
    end

    Flux.update!(opt_state, a, ga)
    Flux.update!(opt_state_2, b, gb)
end


xx = collect(-2:0.05:2)
yy = xx  # Using the same range for y axis

# Create the grid points
xy_input = zeros(Float32, 2, length(xx), length(yy))
for (i, x) in enumerate(xx), (j, y) in enumerate(yy)
    xy_input[:, i, j] = [x, y]
end

z_data = a(xy_input)
z_data_2 = b(xy_input)

p = plot(surface(x=xx, y=yy, z=z_data[1, :, :], colorscale="YlOrRd", showscale=false))
add_trace!(p, scatter3d(x=input[1,:], y=input[2,:], z=output[:], mode="markers", marker=attr(size=4, color="red", opacity=1.0)))
display(p)

p = plot(surface(x=xx, y=yy, z=z_data_2[1, :, :], colorscale="YlOrRd", showscale=false))
add_trace!(p, scatter3d(x=input_2[1,:], y=input_2[2,:], z=output_2[:], mode="markers", marker=attr(size=4, color="green", opacity=1.0)))
display(p)

loss_plot = plot([losses1 losses2])
display(loss_plot)