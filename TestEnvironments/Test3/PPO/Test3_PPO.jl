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
using JSON
using UnicodePlots


scriptname = "Test3_PPO"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
end


include("./../Test3_env.jl")


seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false



# agent tuning parameters
memory_size = 0
nna_scale = 1.6
nna_scale_critic = 0.8
drop_middle_layer = false
drop_middle_layer_critic = false
fun = gelu
use_gpu = false
actionspace = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.99f0
gamma = y

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 800


learning_rate = 1e-4
learning_rate_critic = 3e-4
n_epochs = 5
n_microbatches = 10
logσ_is_network = false
max_σ = 1.0f0
entropy_loss_weight = 0#.1agen
clip_grad = 0.02
target_kl = 0.1
clip1 = false
start_logσ = -0.6
tanh_end = false
clip_range = 0.05f0
clip_range_vf = 0.08

betas = (0.9, 0.99)
noise = nothing#"perlin"


normalize_advantage = true

reward_shaping = false

wind_only = false





function initialize_setup(;use_random_init = false)

    global env = GeneralEnv(do_step = do_step, 
                reward_function = reward_function,
                featurize = featurize,
                prepare_action = prepare_action,
                y0 = y0,
                te = te, t0 = t0, dt = dt, 
                sim_space = sim_space, 
                action_space = actionspace,
                max_value = 1.0,
                check_max_value = "nothing")


        
        dim = 15

        logσ = Chain(
            Dense(state_dim, dim, relu, bias = false),
            Dense(dim, dim, relu, bias = false),
            Dense(dim, 1, identity, bias = false)
        )

        logσ.layers[1].weight[:] .*= 0.2
        logσ.layers[2].weight[:] .*= 0.2
        logσ.layers[2].weight[:] = -(abs.(logσ.layers[2].weight[:]))

        approximator = ActorCritic(
            actor = GaussianNetwork(
                μ = Chain(
                    Dense(state_dim, 20, fun),
                    Dense(20, 12, fun),
                    Dense(12, 6, fun),
                    Dense(6, 1)
                ),
                logσ = [-0.5],
                logσ_is_network = false,
                max_σ = max_σ,
            ),
            critic = Chain(
                Dense(state_dim, 20, fun),
                Dense(20, 12, fun),
                Dense(12, 6, fun),
                Dense(6, 1)
            ),
            optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
            optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
        )

        global agent = create_agent_ppo(
                # approximator = approximator,
                action_space = actionspace,
                state_space = env.state_space,
                use_gpu = use_gpu, 
                rng = rng,
                y = y, p = p,
                update_freq = update_freq,
                learning_rate = learning_rate,
                learning_rate_critic = learning_rate_critic,
                nna_scale = nna_scale,
                nna_scale_critic = nna_scale_critic,
                drop_middle_layer = drop_middle_layer,
                drop_middle_layer_critic = drop_middle_layer_critic,
                fun = fun,
                clip1 = clip1,
                n_epochs = n_epochs,
                n_microbatches = n_microbatches,
                logσ_is_network = logσ_is_network,
                max_σ = max_σ,
                entropy_loss_weight = entropy_loss_weight,
                clip_grad = clip_grad,
                target_kl = target_kl,
                start_logσ = start_logσ,
                tanh_end = tanh_end,
                clip_range = clip_range,
                clip_range_vf = clip_range_vf,
                betas = betas,
                noise = noise,
                normalize_advantage = normalize_advantage,)


    global hook = GeneralHook(min_best_episode = min_best_episode,
                            collect_NNA = false,
                            generate_random_init = generate_random_init,
                            collect_history = false,
                            collect_rewards_all_timesteps = false,
                            early_success_possible = true)
end


initialize_setup()





function render_run(; exploration = false, return_plot = false)
    positions = Float64[]
    times = Float64[]
    zone_traces = [Float64[] for _ in 1:3]

    reset!(env)
    generate_random_init()


    while !env.done
        action = exploration ? agent(env) : prob(agent.policy, env).μ
        env(action)
        push!(positions, Float64(env.y[1]))
        push!(times, Float64(env.y[2]))
        push!(zone_traces[1], Float64(env.y[3]))
        push!(zone_traces[2], Float64(env.y[4]))
        push!(zone_traces[3], Float64(env.y[5]))
    end

    time_axis = times

    layout = Layout(
        plot_bgcolor = "white",
        xaxis = attr(title = "Time"),
        yaxis = attr(title = "Position / Zones", range = [0, 1]),
        showlegend = true,
    )

    colors = [
        (255, 215, 0),   # yellow
        (0, 180, 0),     # green
        (220, 20, 60),   # red
    ]

    plot_data = AbstractTrace[]

    # zone bands
    for i in 1:3
        lower = zone_traces[i] .- delta_zone
        upper = zone_traces[i] .+ delta_zone
        fillcol = "rgba($(colors[i][1]), $(colors[i][2]), $(colors[i][3]), 0.25)"

        push!(plot_data, scatter(
            x = time_axis,
            y = lower,
            mode = "lines",
            line_color = "rgba(0,0,0,0)",
            showlegend = false
        ))
        push!(plot_data, scatter(
            x = time_axis,
            y = upper,
            mode = "lines",
            fill = "tonexty",
            fillcolor = fillcol,
            line_color = "rgba(0,0,0,0)",
            name = "Zone $i"
        ))
    end

    # agent position
    push!(plot_data, scatter(
        x = time_axis,
        y = positions,
        mode = "lines+markers",
        name = "Position",
        line_color = "rgb(50,50,200)"
    ))

    plt = plot(plot_data, layout)

    if return_plot
        return plt
    else
        display(plt)
    end
end
