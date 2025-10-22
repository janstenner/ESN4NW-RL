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


scriptname = "Minimal1"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
    println(io, "training.mp4")
end


include("./../Minimal1_env.jl")


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


learning_rate = 2e-4
learning_rate_critic = 4e-4
n_epochs = 5
n_microbatches = 100
actorbatch_size = 1000000
logσ_is_network = true
max_σ = 1.0f0
entropy_loss_weight = 0.3f0
clip_grad = 0.5
target_kl = 0.01
clip1 = false
start_logσ = -0.3
tanh_end = true
clip_range = 0.1f0
clip_range_vf = 0.2f0

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

trajectories_file = "optimal_trajectories.jld2"
trajectories = FileIO.load(trajectories_file, "trajectories")
optimal_trajectory = trajectories["PPO2"]["with_RS"]





function render_run(; plot_optimal = false, steps = 6000, show_training_episode = false, show_σ = false, exploration = false, return_plot = false, gae = true, plot_values = true, plot_critic2 = false, critic2_diagnostics = false, new_day = true,)
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

    global values = []
    global values2 = []
    global values3 = []
    global states = []
    global mus = []
    global terminals = []

    if new_day
        reset!(env)
        generate_random_init()
    else
        reset!(env)

        y0 = create_state(; generate_day = false)

        env.y0 = deepcopy(y0)
        env.y = deepcopy(y0)
        env.state = env.featurize(; env = env)

        global day_trajectory = CircularArrayTrajectory(;
                capacity = 288,
                state = Float32 => (size(env.state_space)[1], 1),
                action = Float32 => (size(env.action_space)[1], 1),
                action_log_prob = Float32 => (1),
                reward = Float32 => (1),
                explore_mod = Float32 => (1),
                terminal = Bool => (1,),
                next_state = Float32 => (size(env.state_space)[1], 1),
        )
    end

    while !env.done

        if exploration
            action = agent(env)
            μ = agent.policy.last_mu[1]
            σ = agent.policy.last_sigma
        else
            prob_temp = prob(agent.policy, env)
            action = prob_temp.μ
            μ = prob_temp.μ[1]
            σ = prob_temp.σ

            if ndims(action) == 2
                log_p = vec(sum(normlogpdf(μ, σ, action), dims=1))
            else
                log_p = normlogpdf(μ, σ, action)
            end

            agent.policy.last_action_log_prob = log_p
        end
        

        #action = agent(env)

        value = agent.policy.approximator.critic(env.state)[1]
        push!(values, value)

        critic2_input = critic2_takes_action ? vcat(env.state, action) : env.state
        value2 = agent.policy.approximator.critic2(critic2_input)[1]
        push!(values2, value2)

        if use_critic3
            value3 = agent.policy.approximator.critic3(env.state)[1]
            push!(values3, value3)
        end

        
        push!(states, env.state)
        push!(mus, μ)

        temp_state = deepcopy(env.state)
        env(action; reward_shaping = reward_shaping)

        push!(terminals, env.done)


        for k in 1:n_windCORES
            push!(results_run["hpc$k"], clamp((action[1]+1)*0.5, 0, 1)) #env.p[k])
            push!(results_run["σ$k"], σ[k])
        end
        push!(results_run["rewards"], env.reward[1])
        push!(results_run["loadleft"], env.y[1])


        if !new_day
            push!(day_trajectory;
                state = temp_state,
                action = action,
                action_log_prob = agent.policy.last_action_log_prob,
                reward = env.reward[:],
                explore_mod = 1.0f0,
                terminal = env.done,
                next_state = env.state,
            )
        end

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
                    ),#range=[0,1]),
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

        #deltas = results_run["rewards"] .+ values2 #.* (1 .- terminals) .- values

        offsets = agent.policy.approximator.critic2(vcat(reduce(hcat, states), mus'))[:]

        deltas = values2 #- offsets

        global y, p
        global advantages

        
        
        if use_whole_delta_targets
                if use_critic3
                    deltas = values2 - values3
                else
                    mean_c2 = RL.antithetic_mean(agent.policy.approximator.actor, agent.policy.approximator.critic2, reduce(hcat, states))[:]

                    deltas = values2 - mean_c2
                end
            advantages, returns = generalized_advantage_estimation(
                deltas,
                zeros(Float32, size(deltas)),
                zeros(Float32, size(deltas)),
                y,
                p;
                terminal=terminals
            )
        else
            advantages, returns = generalized_advantage_estimation(
                results_run["rewards"],
                values,
                values2,
                y,
                p;
                terminal=terminals
            )
        end
        

        #advantages = next_values #- values

        if normalize_advantage
            advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
        end


        

        push!(to_plot, scatter(x=xx, y=results_run["hpc1"], name="Advantage",
            mode="markers",
            marker=attr(
                color=advantages,               # array of numbers
                cmin = -1.0,
                cmid = 0.0,
                cmax = 1.0,
                colorscale=colorscale,
                showscale=false
            ),
            line=attr(color = "rgba(200, 200, 200, 0.3)")))
        
        
        if use_critic3
            push!(to_plot, scatter(x=xx, y=values2-values3, name="Values2-Values3", yaxis = "y2"))
            push!(to_plot, scatter(x=xx, y=values2, name="Values2", yaxis = "y2"))
            push!(to_plot, scatter(x=xx, y=values3, name="Values3", yaxis = "y2"))
        else
            push!(to_plot, scatter(x=xx, y=values2-mean_c2, name="Values2-mean_c2", yaxis = "y2"))
            push!(to_plot, scatter(x=xx, y=values2, name="Values2", yaxis = "y2"))
            push!(to_plot, scatter(x=xx, y=mean_c2, name="mean_c2", yaxis = "y2"))
        end
        #push!(to_plot, scatter(x=xx, y=results_run["rewards"], name="Reward", yaxis = "y2"))
    else
        push!(to_plot, scatter(x=xx, y=values, name="Values", yaxis = "y2"))
        push!(to_plot, scatter(x=xx, y=results_run["rewards"], name="Reward", yaxis = "y2"))
    end

    if plot_values
        #push!(to_plot, scatter(x=xx, y=offset_values, name="Critic Value", yaxis = "y2"))
        #push!(to_plot, scatter(x=xx, y=returns, name="Return", yaxis = "y2"))
    end

    push!(to_plot, scatter(x=xx, y=results_run["loadleft"], name="Load Left"))
    push!(to_plot, scatter(x=xx, y=grid_price[history_steps:end], name="Grid Price"))


    for k in 1:n_windCORES
        push!(to_plot, scatter(x=xx, y=results_run["hpc$k"], name="WindCORE utilization $k",
        line=attr(color = "rgba(200, 200, 200, 0.3)")))
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

    if plot_critic2

        colorscale2 = [[0.0, "rgb(50, 0, 50)"], [0.25, "rgb(200, 0, 0)"], [0.5, "rgb(210, 210, 0)"], [0.75, "rgb(0, 210, 0)"], [1.0, "rgb(140, 255, 255)"]]

        layout = Layout(
                plot_bgcolor="#f1f3f7",
                coloraxis = attr(cmid = 0, colorscale = colorscale2),
            )

        actions = collect(-1:0.02:1)

        if critic2_diagnostics
            global new_states = []
            temp_state = deepcopy(states[1])

            temp_state[2] = 1.0f0

            wind_index = 2 + include_history_steps - 1 + include_gradients + 2

            for i in 3:wind_index-2
                temp_state[i] = 0.0f0
            end

            temp_state[wind_index] = 0.0f0
            
            for i in wind_index+1:length(temp_state)-2
                temp_state[i] = 0.0f0
            end

            temp_state[end-1] = clamp(temp_state[6] - curtailment_threshold, 0.0f0, 1.0f0)

            push!(new_states, deepcopy(temp_state))

            xx = [0.0f0]

            for i in 1:288
                temp_state[wind_index] += 1.0f0 / 288.0f0
                temp_state[end-1] = clamp(temp_state[wind_index] - curtailment_threshold, 0.0f0, 1.0f0)
                push!(new_states, deepcopy(temp_state))
                push!(xx, temp_state[wind_index])
            end

            states = new_states
        end

        results_critic2 = zeros(Float32, length(actions), length(states))

        for (i,state) in enumerate(states)
            inputs = vcat(repeat(state, 1, length(actions)), actions')

            #mu = agent.policy.approximator.actor.μ(state)[:]
            #mu_value = agent.policy.approximator.critic2(vcat(state, mu))
            #mu_values = mu_value .* ones(length(actions))

            critic2_values = agent.policy.approximator.critic2(inputs)[:] #-1 first

            results_critic2[:,i] = critic2_values #- mu_values
        end

        #results_critic2 = (results_critic2 .- mean(results_critic2)) ./ clamp(std(results_critic2), 1e-8, 1000.0)

        min_val = - maximum(abs.(results_critic2))

        for (i,state) in enumerate(states)

            if critic2_diagnostics
                idx = clamp(searchsortedfirst(actions, state[end-1] * 2 - 1), 1, length(actions))

                idx2 = findmax(results_critic2[:,i])[2]
                results_critic2[idx2,i] = -min_val
            else
                idx = clamp(searchsortedfirst(actions, mus[i]), 1, length(actions))

                idx2 = findmax(results_critic2[:,i])[2]
                results_critic2[idx2,i] = -min_val
            end

            results_critic2[idx,i] = min_val
        end

        display(plot(PlotlyJS.heatmap(x = xx, y = actions, z=results_critic2, coloraxis="coloraxis"), layout))

    end
end




function get_state(x, y)
    st = Float32[0.8; 0.0; 0.0; 0.0; 0.0; 0.0; 0.4; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.2;;]

    grid_base = Float32[0.05310250000000005, 0.06105506000000005, 0.06905967000000002, 0.07711570000000001, 0.08522229999999997]
    wind_base = Float32[0.05310250000000005, 0.06105506000000005, 0.06905967000000002, 0.07711570000000001, 0.08522229999999997]

    st[2:6] .= grid_base .+ x * 0.9
    st[8:12] .= wind_base .+ y * 0.9

    st[13] = max(0.0, st[8] - curtailment_threshold)

    st
end

function plot_critic(; return_plot = false)
    xx = collect(0:0.025:1)
    yy = xx
    
    critic_values = zeros(Float32, length(xx), length(yy))

    for (i, _) in enumerate(xx), (j, _) in enumerate(yy)
        st = get_state(i, j)

        critic_value = agent.policy.approximator.critic(st)[1]

        critic_values[i, j] = critic_value
    end
    

    p = plot(surface(x=xx, y=yy, z=critic_values), Layout(
        scene = attr(
            xaxis_title="grid price",
            yaxis_title="wind power",
            zaxis_title="Critic Value"
        )
    ))

    if return_plot
        return p
    else
        display(p)
    end
end


function plot_trajectory()
    t = agent.trajectory
    AC = agent.policy.approximator
    states = collect(flatten_batch(t[:state]))
    actions = collect(flatten_batch(t[:action]))

    values = AC.critic(states)
    next_values = values + AC.critic2(vcat(states, actions))

    advantages, returns = generalized_advantage_estimation(
        t[:reward],
        values,
        next_values,
        y,
        p;
        dims=2,
        terminal=t[:terminal]
    )

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

    to_plot = AbstractTrace[]
    

    push!(to_plot, scatter(y=advantages[:], name="Advantage", yaxis = "y2",
            mode="lines+markers",
            marker=attr(
                color=advantages[:],               # array of numbers
                cmin = -0.01,
                cmid = 0.0,
                cmax = 0.01,
                colorscale=colorscale,
                showscale=false
            ),
            line=attr(color = "rgba(200, 200, 200, 0.3)")))
    
    push!(to_plot, scatter(y=t[:reward][:], name="Reward", yaxis = "y2"))
    


    push!(to_plot, scatter(y=values[:], name="Critic Values", yaxis = "y2"))
    push!(to_plot, scatter(y=next_values[:], name="Next Value", yaxis = "y2"))
    push!(to_plot, scatter(y=returns[:], name="Return", yaxis = "y2"))


    push!(to_plot, scatter(y=states[1,:], name="Load Left"))
    push!(to_plot, scatter(y=states[2,:], name="Grid Price"))

    push!(to_plot, scatter(y=t[:action][:], name="WindCORE utilization 1"))

    push!(to_plot, scatter(y=states[6,:], name="Wind Power 1"))

    push!(to_plot, scatter(y=Float32.(t[:terminal][:]), name="Terminal"))

    plott = plot(Vector(to_plot), layout)

    display(plott)
end

include("./same_day_trainer.jl")
include("./critic_problem.jl")