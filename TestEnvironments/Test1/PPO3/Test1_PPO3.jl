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


scriptname = "Test1"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
    println(io, "training.mp4")
end


include("./../Test1_env.jl")


seed = Int(floor(rand()*100000))
#seed = 23038

gpu_env = false



# agent tuning parameters
memory_size = 0
nna_scale = 6.4
nna_scale_critic = 4.2
drop_middle_layer = false
drop_middle_layer_critic = false
fun = gelu
use_gpu = false
actionspace = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.0f0
gamma = y

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 6000

critic_frozen_update_freq = 4
actor_update_freq = 2


learning_rate = 6e-5
learning_rate_critic = 2e-4
n_epochs = 5
n_microbatches = 100
actorbatch_size = 10
logσ_is_network = true
max_σ = 1.0f0
entropy_loss_weight = 0.02f0
clip_grad = 0.5
target_kl = 0.01
clip1 = false
start_logσ = -0.3
tanh_end = true
clip_range = 0.05f0
clip_range_vf = 0.1f0

λ_targets = 0.9f0
n_targets = 100

betas = (0.9, 0.99)
noise = nothing #"perlin"
noise_scale = 20
normalize_advantage = true
fear_scale = 0.4
new_loss = false#true
adaptive_weights = true
critic2_takes_action = true
use_popart = false
critic_frozen_factor = 0.3f0
use_exploration_module = false
use_whole_delta_targets = true
use_critic3 = false

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


        
        

        global agent = create_agent_ppo3(
                # approximator = approximator,
                action_space = actionspace,
                state_space = env.state_space,
                use_gpu = use_gpu, 
                rng = rng,
                y = y, p = p,
                update_freq = update_freq,
                critic_frozen_update_freq = critic_frozen_update_freq,
                actor_update_freq = actor_update_freq,
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
                actorbatch_size = actorbatch_size,
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
                noise_scale = noise_scale,
                normalize_advantage = normalize_advantage,
                fear_scale = fear_scale,
                new_loss = new_loss,
                adaptive_weights = adaptive_weights,
                critic2_takes_action = critic2_takes_action,
                use_popart = use_popart,
                critic_frozen_factor = critic_frozen_factor,
                λ_targets = λ_targets,
                n_targets = n_targets,
                use_critic3 = use_critic3,
                use_exploration_module = use_exploration_module,
                use_whole_delta_targets = use_whole_delta_targets,
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
        action = exploration ? agent(env) : prob(agent.policy, env).μ
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