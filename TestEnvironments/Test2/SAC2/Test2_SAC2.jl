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


scriptname = "Test2_SAC2"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
    println(io, "training.mp4")
end


include("./../Test2_env.jl")


seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false



# agent tuning parameters
nna_scale = 1.6
nna_scale_critic = 0.8
drop_middle_layer = false
drop_middle_layer_critic = false
fun = gelu
logσ_is_network = false
tanh_end = false
use_gpu = false
actionspace = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
gamma = y
a = 3f-2 #0.2f0
t = 0.02f0
target_entropy = -0.9f0
use_popart = false


learning_rate = 3e-4
learning_rate_critic = 3e-4
trajectory_length = 1_000_000
batch_size = 32
update_after = 100_000
update_freq = 50
update_loops = 1
clip_grad = 0.5
start_logσ = -1.5
automatic_entropy_tuning = true
on_policy_critic_update_freq = 300
λ_targets = 0.9f0
lr_alpha = 1e-2
fear_factor = 1.0f0
target_frac = 0.1f0

verbose = false

reward_shaping = false

wind_only = false

betas = (0.9, 0.9)





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


        
        

        global agent = create_agent_sac2(
                action_space = actionspace,
                state_space = env.state_space,
                rng = rng,
                y = y,
                a = a,
                t = t,
                use_gpu = use_gpu,
                update_after = update_after,
                update_freq = update_freq,
                update_loops = update_loops,
                trajectory_length = trajectory_length,
                batch_size = batch_size,
                learning_rate = learning_rate,
                learning_rate_critic = learning_rate_critic,
                nna_scale = nna_scale,
                nna_scale_critic = nna_scale_critic,
                drop_middle_layer = drop_middle_layer,
                drop_middle_layer_critic = drop_middle_layer_critic,
                fun = fun,
                logσ_is_network = logσ_is_network,
                clip_grad = clip_grad,
                start_logσ = start_logσ,
                tanh_end = tanh_end,
                automatic_entropy_tuning = automatic_entropy_tuning,
                target_entropy = target_entropy,
                use_popart = use_popart,
                on_policy_critic_update_freq = on_policy_critic_update_freq,
                λ_targets = λ_targets,
                lr_alpha = lr_alpha,
                betas = betas,
                fear_factor = fear_factor,
                target_frac = target_frac,
                verbose = verbose,
                )


    global hook = GeneralHook(min_best_episode = min_best_episode,
                            collect_NNA = false,
                            generate_random_init = generate_random_init,
                            collect_history = false,
                            collect_rewards_all_timesteps = false,
                            early_success_possible = true)
end



initialize_setup()





function render_run(; exploration = false, return_plot = false)
    rewards = Float64[]
    actions_taken = Float64[]

    reset!(env)
    generate_random_init()


    while !env.done
        action = exploration ? agent(env) : agent.policy.actor.μ(env.state)
        env(action)
        push!(rewards, env.reward[1])
        p_val = env.p isa AbstractArray ? env.p[1] : env.p
        push!(actions_taken, Float64(p_val))
    end

    time_axis = (0:length(rewards)-1) .* dt

    layout = Layout(
        plot_bgcolor = "white",
        xaxis = attr(title = "Time"),
        yaxis = attr(title = "Reward"),
        yaxis2 = attr(
            overlaying = "y",
            side = "right",
            title = "Action",
        ),
        showlegend = true,
    )

    plot_data = [
        scatter(x = time_axis, y = rewards, mode = "lines+markers", name = "Reward per step"),
        scatter(x = time_axis, y = actions_taken, mode = "lines+markers", name = "Action (env.p)", yaxis = "y2"),
    ]

    plt = plot(plot_data, layout)

    if return_plot
        return plt
    else
        display(plt)
    end
end
