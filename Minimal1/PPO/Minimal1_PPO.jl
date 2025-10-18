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


scriptname = "Minimal1_PPO"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
end


include("./../Minimal1_env.jl")


seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false



# agent tuning parameters
memory_size = 0
nna_scale = 6.4
nna_scale_critic = 3.2
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

update_freq = 60_000


learning_rate = 1e-4
learning_rate_critic = 3e-4
n_epochs = 5
n_microbatches = 100
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

trajectories_file = "optimal_trajectories.jld2"
trajectories = FileIO.load(trajectories_file, "trajectories")
optimal_trajectory = trajectories["PPO"]["with_RS"]





function render_run(; plot_optimal = false, steps = 6000, show_training_episode = false, show_σ = false, exploration = false, return_plot = false, gae = false, plot_values = true)
    global history_steps

    if show_training_episode
        training_episode = length(hook.rewards)
    end


    # if use_best
    #     copyto!(agent.policy.behavior_actor, hook.bestNNA)
    # end

    # temp_noise = agent.policy.act_noise
    # agent.policy.act_noise = 0.0

    # temp_start_steps = agent.policy.start_steps
    # agent.policy.start_steps  = -1
    
    # temp_update_after = agent.policy.update_after
    # agent.policy.update_after = 100000


    # agent.policy.update_step = 0
    global rewards = Float64[]
    reward_sum = 0.0

    #w = Window()

    xx = collect(dt/60:dt/60:te/60)

    global results_run = Dict("rewards" => [], "loadleft" => [])

    for k in 1:n_windCORES
        results_run["hpc$k"] = []
        results_run["σ$k"] = []
    end

    values = []
    next_values = []

    reset!(env)
    generate_random_init()

    while !env.done

        if exploration
            action = agent(env)
            σ = agent.policy.last_sigma
        else
            prob_temp = prob(agent.policy, env)
            action = prob_temp.μ
            σ = prob_temp.σ
        end
        

        #action = agent(env)

        push!(values, agent.policy.approximator.critic(env.state)[1])

        env(action; reward_shaping = reward_shaping)

        push!(next_values, agent.policy.approximator.critic(env.state)[1])

        for k in 1:n_windCORES
            push!(results_run["hpc$k"], env.p[k])
            push!(results_run["σ$k"], σ[k])
        end
        push!(results_run["rewards"], env.reward[1])
        push!(results_run["loadleft"], env.y[1])

        # println(mean(env.reward))

        reward_sum += mean(env.reward)
        # push!(rewards, mean(env.reward))

        
    end

    # if use_best
    #     copyto!(agent.policy.behavior_actor, hook.currentNNA)
    # end

    # agent.policy.start_steps = temp_start_steps
    # agent.policy.act_noise = temp_noise
    # agent.policy.update_after = temp_update_after

    println(reward_sum)


    colorscale = [[0, "rgb(255, 0, 0)"], [0.5, "rgb(255, 255, 255)"], [1, "rgb(0, 255, 0)"], ]

    layout = Layout(
                    plot_bgcolor = "white",
                    font=attr(
                        family="Arial",
                        size=16,
                        color="black"
                    ),
                    showlegend = true,
                    legend=attr(x=0.5, y=-0.1, orientation="h", xanchor="center"),
                    xaxis = attr(gridcolor = "#E0E0E0FF",
                                linecolor = "#888888"),
                    yaxis = attr(gridcolor = "#E0E0E0FF",
                                linecolor = "#888888",
                                range=[0,1]),
                    yaxis2 = attr(
                        overlaying="y",
                        side="right",
                        titlefont_color="orange",
                        #range=[-1, 1]
                    ),
                )

    if show_training_episode
        layout.title = "Evaluation Episode after $(training_episode) Training Episodes"
    end

    

    to_plot = AbstractTrace[]
    
    if show_σ
        for k in 1:n_windCORES
            push!(to_plot, scatter(x=xx, y=results_run["σ$k"], name="σ$k", yaxis = "y2"))
        end
    elseif gae
        global y, p
        advantages, returns = generalized_advantage_estimation(
            results_run["rewards"],
            values,
            next_values,
            y,
            p
        )

        push!(to_plot, scatter(x=xx, y=advantages, name="Advantage", yaxis = "y2",
            mode="lines+markers",
            marker=attr(
                color=advantages,               # array of numbers
                cmin = -0.01,
                cmid = 0.0,
                cmax = 0.01,
                colorscale=colorscale,
                showscale=false
            ),
            line=attr(color = "rgba(200, 200, 200, 0.3)")))
    else
        push!(to_plot, scatter(x=xx, y=results_run["rewards"], name="Reward", yaxis = "y2"))
    end

    if plot_values
        push!(to_plot, scatter(x=xx, y=values, name="Critic Value", yaxis = "y2"))
        #push!(to_plot, scatter(x=xx, y=returns, name="Return", yaxis = "y2"))
    end

    push!(to_plot, scatter(x=xx, y=results_run["loadleft"], name="Load Left"))
    push!(to_plot, scatter(x=xx, y=grid_price[history_steps:end], name="Grid Price"))


    for k in 1:n_windCORES
        push!(to_plot, scatter(x=xx, y=results_run["hpc$k"], name="WindCORE utilization $k"))
    end


    for k in 1:n_turbines
        push!(to_plot, scatter(x=xx, y=wind[k][history_steps:end], name="Wind Power $k"))
    end
    

    if plot_optimal
        global optimal_actions = optimize_day(steps)
        global optimal_rewards = evaluate(optimal_actions; collect_rewards = true)

        for k in 1:n_windCORES
            push!(to_plot, scatter(x=xx, y=optimal_actions[k,:], name="Optimal HPC$k"))
        end
        push!(to_plot, scatter(x=xx, y=optimal_rewards, name="Optimal Reward", yaxis = "y2"))


        println("")
        println("--------------------------------------------")
        println("AGENT:   $reward_sum")
        println("IPOPT:   $(sum(optimal_rewards))")
        println("--------------------------------------------")
    end

    plott = plot(Vector(to_plot), layout)

    if return_plot
        return plott
    else
        display(plott)
    end
end


