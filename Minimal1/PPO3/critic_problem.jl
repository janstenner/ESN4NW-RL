

using JLD2
using FileIO



const CRITIC_SNAPSHOT_PATH = joinpath(@__DIR__, "critic_snapshot.jld2")

function save_critic_snapshot!(path::AbstractString=CRITIC_SNAPSHOT_PATH)

    # code for saving a copy of the current critic NN
    critic = deepcopy(agent.policy.approximator.critic)
    optimizer_critic = deepcopy(agent.policy.approximator.optimizer_critic)
    critic_state_tree = deepcopy(agent.policy.approximator.critic_state_tree)
    critic_compare = deepcopy(critic)
    optimizer_critic_compare = Optimisers.AdamW(learning_rate_critic, betas)
    critic_compare_state_tree = Flux.setup(optimizer_critic_compare, critic_compare)

    global state_visits
    global state_returns
    global state_targets
    global state_values
    global multiple_day_trajectory
    global targets

    FileIO.save(
        path,
        "critic", critic,
        "optimizer_critic", optimizer_critic,
        "critic_state_tree", critic_state_tree,
        "critic_compare", critic_compare,
        "optimizer_critic_compare", optimizer_critic_compare,
        "critic_compare_state_tree", critic_compare_state_tree,
        "state_visit", state_visits,
        "state_returns", state_returns,
        "state_targets", state_targets,
        "state_values", state_values,
        "multiple_day_trajectory", multiple_day_trajectory,
        "targets", targets,
    )
    return nothing
end


function load_critic_snapshot!(path::AbstractString=CRITIC_SNAPSHOT_PATH)
    data = FileIO.load(path)
    global critic = data["critic"]
    global optimizer_critic = data["optimizer_critic"]
    global critic_state_tree = data["critic_state_tree"]
    global critic_compare = data["critic_compare"]
    global optimizer_critic_compare = data["optimizer_critic_compare"]
    global critic_compare_state_tree = data["critic_compare_state_tree"]
    global state_visit = data["state_visit"]
    global state_returns = data["state_returns"]
    global state_targets = data["state_targets"]
    global state_values = data["state_values"]
    global multiple_day_trajectory = data["multiple_day_trajectory"]
    global targets = data["targets"]
    return nothing
end


# Pre-activation residual block: LN -> GELU -> Dense -> GELU -> Dense + skip
struct ResMLPBlock
    ln1::LayerNorm
    d1::Dense
    ln2::LayerNorm
    d2::Dense
end
ResMLPBlock(width::Int) = ResMLPBlock(
    LayerNorm(width), Dense(width, width, gelu),
    LayerNorm(width), Dense(width, width, gelu),
)
(m::ResMLPBlock)(x) = x .+ m.d2(m.ln2(m.d1(m.ln1(x))))

# Critic: input 10-dim → widen → a few residual blocks → linear head
function make_critic(; in_dim=10, width=128, blocks=4)
    Chain(
        Dense(in_dim, width, gelu),              # widening stem
        (ResMLPBlock(width) for _ in 1:blocks)...,
        LayerNorm(width),                        # final pre-act norm stabilizes the head
        Dense(width, 1)                          # linear, unbounded value head
    )
end

critic_compare = make_critic(in_dim=10, width=128, blocks=4)
optimizer_critic_compare = Optimisers.AdamW(learning_rate_critic, betas)
critic_compare_state_tree = Flux.setup(optimizer_critic_compare, critic_compare)


function train_one(epochs = 5, batch_count = 100)
    states_array = multiple_day_trajectory[:state]

    rewards = collect(multiple_day_trajectory[:reward])
    terminal = collect(multiple_day_trajectory[:terminal])
    next_states = flatten_batch(multiple_day_trajectory[:next_state])
    next_values = reshape( critic( next_states ), 1, :)
    next_values_compare = reshape( critic_compare( next_states ), 1, :)

    γ = gamma
    global targets = lambda_truncated_targets(rewards, terminal, next_values, γ)[:]
    global targets_compare = lambda_truncated_targets(rewards, terminal, next_values_compare, γ)[:]

    global values_before = critic(states_array)
    global values_compare_before = critic_compare(states_array)

    n_states = size(states_array, 3)  # number of states recorded
    minibatch_size = Int(floor(n_states ÷ batch_count))

    for epoch in 1:epochs
        rand_inds = shuffle!(rng, collect(1:size(states_array, 3)))

        for i in 1:batch_count

            inds = rand_inds[(i-1)*minibatch_size+1:i*minibatch_size]

            g_critic, g_critic_compare = Flux.gradient(critic, critic_compare) do cr, cr_c
                v_cr = cr(states_array[:,:,inds])
                v_cr_c = cr_c(states_array[:,:,inds])

                loss_cr = mean((v_cr[:] .- targets[inds]).^2)
                loss_cr_c = mean((v_cr_c[:] .- targets_compare[inds]).^2)

                loss = loss_cr + loss_cr_c

                loss
            end

            Flux.update!(critic_state_tree, critic, g_critic)
            Flux.update!(critic_compare_state_tree, critic_compare, g_critic_compare)
        end
    end


    global values = critic(states_array)
    global values_compare = critic_compare(states_array)

    @show maximum(abs.(values_before - values))
    @show maximum(abs.(values_compare_before - values_compare))


    # Create bins for load_left
    min_load = 0.0
    load_bins = collect(LinRange(min_load,1,200))
    n_load_bins = length(load_bins)
    time_bins = 1:288

    global state_visits = zeros(Int, n_load_bins-1, 288)
    global state_values = ones(Float32, n_load_bins-1, 288) .* minimum(values)
    global state_values_compare = ones(Float32, n_load_bins-1, 288) .* minimum(values_compare)
    global state_targets = ones(Float32, n_load_bins-1, 288) .* minimum(targets)
    global state_targets_compare = ones(Float32, n_load_bins-1, 288) .* minimum(targets_compare)

    for i in 1:n_states
        load_left = states_array[1, 1, i]  # state[1] - load_left value
        time_step = round(Int, states_array[end, 1, i] * te / dt + 1)        # time step (1-288)
        
        # Find the appropriate load_left bin
        load_bin = searchsortedfirst(load_bins, load_left) - 1
        load_bin = clamp(load_bin, 1, n_load_bins-1)

        # Increment the visitation count
        if state_visits[load_bin, time_step] == 0
            first = true
        else
            first = false
        end

        state_visits[load_bin, time_step] += 1
        if first
            state_values[load_bin, time_step] = values[i]
            state_values_compare[load_bin, time_step] = values_compare[i]
            state_targets[load_bin, time_step] = targets[i]
            state_targets_compare[load_bin, time_step] = targets_compare[i]
        else
            state_values[load_bin, time_step] += values[i]
            state_values_compare[load_bin, time_step] += values_compare[i]
            state_targets[load_bin, time_step] += targets[i]
            state_targets_compare[load_bin, time_step] += targets_compare[i]
        end
    end

    state_values ./= state_visits .+ (state_visits .== 0)
    state_values_compare ./= state_visits .+ (state_visits .== 0)
    state_targets ./= state_visits .+ (state_visits .== 0)
    state_targets_compare ./= state_visits .+ (state_visits .== 0)


    colorscale2 = [[0.0, "rgb(5, 0, 5)"], [0.1, "rgb(200, 0, 0)"], [0.5, "rgb(210, 210, 0)"], [0.75, "rgb(0, 210, 0)"], [1.0, "rgb(140, 255, 255)"]]

    # Create subplots layout
    layout = Layout(
        grid=attr(rows=3, columns=2, pattern="independent", rowgap=0.01, colgap=0.01),
        showlegend=false,
        margin=attr(t=50, pad=0),
    )

    # Create all four heatmaps
    heatmap1 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=state_targets,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Values", x=0.95, y=0.2),
        name="State Values"
    ))

    heatmap2 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=state_targets_compare,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Values", x=0.95, y=0.2),
        name="State Values"
    ))

    heatmap3 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=state_values,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Values", x=0.95, y=0.2),
        name="State Values"
    ))

    heatmap4 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=state_values_compare,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Values", x=0.95, y=0.2),
        name="State Values"
    ))

    sub = abs.(state_values - state_targets)
    sub[state_visits .== 0] .= 0.0f0

    heatmap5 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=sub,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Values", x=0.95, y=0.2),
        name="State Values"
    ))

    sub_compare = abs.(state_values_compare - state_targets_compare)
    sub_compare[state_visits .== 0] .= 0.0f0

    heatmap6 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=sub_compare,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Values", x=0.95, y=0.2),
        name="State Values"
    ))

    # Create and display the combined plot
    fig = [heatmap1 heatmap2; heatmap3 heatmap4; heatmap5 heatmap6]

    relayout!(fig, layout.fields)

    fig
end