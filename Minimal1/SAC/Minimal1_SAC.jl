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


n_windCORES = 1
n_turbines = 1


scriptname = "Minimal1"




#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
end




# env parameters

seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false

te = 1440.0
dt = 5.0
t0 = 0.0
min_best_episode = 1




# action vector dim - contains the percentage of maximum power the HPC in the turbine will use for the duration of next time step
action_dim = n_windCORES


# Curtailment threshold
curtailment_threshold = 0.4

# state vector

# - amount of computation left (starts at 1.0 and goes to 0.0)
# - price of energy from the grid (last 5 steps)
# - current curtailment threshold
# - wind stituation at every turbine (last 5 steps) plus current wind power minus curtailment threshold
# - current time

history_steps = 5

function generate_wind()
    global history_steps, te, dt

    wind_steps = Int(te/dt) + history_steps

    wind_constant_day = rand()
    deviation = 1/5

    result = sign(randn()) * sin.(collect(LinRange(rand()*3+1, 4+rand()*4, wind_steps)))

    for i in 1:4
        result += sign(randn()) * sin.(collect(LinRange(rand()+4, 5+rand()*i*4, wind_steps)))
    end

    result .-= minimum(result)
    result ./= maximum(result)
    result .*= deviation

    day_wind = sign(randn()) * sin.(collect(LinRange(wind_constant_day*2*pi, 2+wind_constant_day*2*pi, wind_steps)))
    day_wind .+= 1.0
    day_wind ./= 4
    day_wind .+= 0.25


    result .+= day_wind

    clamp!(result, -1.0, 1.0)

    result
end

function generate_grid_price()
    global history_steps, te, dt

    grid_price_steps = Int(te/dt) + history_steps

    factor = 1.0;
    factor = 0.6;

    gp = (-sin.(collect(LinRange(rand()*1.5*factor, 2+rand()*2.5*factor, grid_price_steps))) .+(1+(rand()*factor)))

    clamp!(gp, -1, 1)

    #gp = ones(grid_price_steps)

    return gp
end

include_history_steps = 1
include_gradients = 2

function create_state(; env = nothing, compute_left = 1.0, step = 0, generate_day = true)
    global wind, grid_price, curtailment_threshold, history_steps, dt, include_history_steps, include_gradients


    if isnothing(env)
        y = [1.0]

        if generate_day
            wind = [generate_wind() for i in 1:n_turbines]
            grid_price = generate_grid_price()
        end

        time = 0.0

    else
        y = [compute_left]

        step = env.steps + 1

        time = (env.time + dt) / env.te

    end


    #test
    # y = []
    # for i in 1:n_turbines
    #     for j in history_steps:-1:(1 + (history_steps - include_history_steps))
    #         push!(y, wind[i][j+step])
    #     end
    # end
    # push!(y, time)
    # return Float32.(y)


    for i in history_steps:-1:(1 + (history_steps - include_history_steps))
        push!(y, grid_price[i+step])
    end

    if include_gradients > 0
        g1 = (grid_price[history_steps+step] - grid_price[history_steps+step-1])/dt
        push!(y, g1)
        if include_gradients > 1
            g2 = (grid_price[history_steps+step] - 2*grid_price[history_steps+step-1] + grid_price[history_steps+step-2])/(dt^2)
            push!(y, g2)
        end
    end


    push!(y, curtailment_threshold)

    for i in 1:n_turbines

        for j in history_steps:-1:(1 + (history_steps - include_history_steps))
            push!(y, wind[i][j+step])
        end

        if include_gradients > 0
            g1 = (wind[i][history_steps+step] - wind[i][history_steps+step-1])/dt
            push!(y, g1)
            if include_gradients > 1
                g2 = (wind[i][history_steps+step] - 2*wind[i][history_steps+step-1] + wind[i][history_steps+step-2])/(dt^2)
                push!(y, g2)
            end
        end

        push!(y, max(0.0, wind[i][history_steps+step] - curtailment_threshold))
    end

    push!(y, time)


    Float32.(y)
end

y0 = create_state()
state_dim = length(y0)

sim_space = Space(fill(0..1, (state_dim)))



# agent tuning parameters
nna_scale = 1.6
nna_scale_critic = 0.8
drop_middle_layer = false
drop_middle_layer_critic = false
fun = leakyrelu
logσ_is_network = true
tanh_end = false
use_gpu = false
actionspace = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.9997f0
gamma = y
a = 0.2f0
t = 0.008f0
target_entropy = -0.8


learning_rate = 7e-5
trajectory_length = 1_000_000
batch_size = 256
update_after = 200_000
update_freq = 10
update_loops = 3
clip_grad = 0.3
start_logσ = -1.5
automatic_entropy_tuning = true

betas = (0.8, 0.98)



wind_only = false


include("./../Validation_Minimal1.jl")





optimal_episodes = 200

if !isdefined(Main, :optimal_trajectory)
    optimal_trajectory = nothing
end





function smoothedReLu(x)
    x *= 100_000

    if x <= 0.0
        result =  0.0
    elseif x <= 0.5
        result =   x^2
    else
        result =   x - 0.25
    end

    return result / 100_000
end


function softplus_shifted(x)
    factor = 700
    log( 1 + exp(factor * (x - 0.006)) ) / factor
end

# xx = collect(-1:0.001:1)
# plot(scatter(y=softplus_shifted.(xx), x=xx))

reward_scale_factor = 100

function calculate_day(action, env, step = nothing; reward_shaping = true)
    global curtailment_threshold, wind, grid_price, history_steps

    if !isnothing(env)
        global wind_only

        compute_left = env.y[1]
        step = env.steps
    else
        compute_left = nothing
        wind_only = false
    end

    step += history_steps

    compute_power = 0.0
    for i in 1:n_windCORES
        compute_power += action[i]*0.01/n_windCORES
    end

    compute_left_before = compute_left

    if !isnothing(env)
        # subtracting the computed load
        compute_power_used = min(compute_left, compute_power)
        compute_left -= compute_power
        compute_left = max(compute_left, 0.0)

        if compute_left == 0.0
            env.done = true
        end
    else
        compute_power_used = compute_power
    end

    compute_left_after = compute_left

    #normalizing
    compute_power_used *= 100/n_turbines

    # reward calculation
    if wind_only
        tempreward = 0.0
        power_for_free = 0.0

        for i in 1:n_turbines
            # curtailment energy onlny when wind is above 0.4
            temp_free_power = (wind[i][step-1] - curtailment_threshold)
            temp_free_power = max(0.0, temp_free_power)

            power_for_free += temp_free_power
        end

        tempreward += abs( sum(action) - power_for_free )
        reward = - tempreward/288.0
    else

        power_for_free = 0.0

        for i in 1:n_turbines

            # curtailment energy onlny when wind is above 0.4
            temp_free_power = (wind[i][step-1] - curtailment_threshold)
            temp_free_power = max(0.0, temp_free_power)

            power_for_free += temp_free_power
        end

        #special_reward = max(0.1 - abs(power_for_free - compute_power_used), 0)
        special_reward = 0

        compute_power_used -= power_for_free
        #compute_power_used = max(0.0, compute_power_used)
        compute_power_used = softplus_shifted(compute_power_used)

        #normalizing
        compute_power_used *= (n_turbines * 0.01)
        
        reward1 = compute_power_used * grid_price[step-1]
        reward = - reward1

        if !isnothing(env) 
            if (env.time + env.dt) >= env.te 
                reward -= compute_left * 1.0
                compute_left_after = 0
            end
        end
    end

    if reward_shaping
        # potential based reward shaping
        beta = 0.5
        reward += beta * (compute_left_before - compute_left_after - (gamma-1) * compute_left_after)

        reward *= reward_scale_factor
    end

    return reward, compute_left
end


function do_step(env; reward_shaping = true)
    global wind_only
    
    reward, compute_left = calculate_day(env.p, env; reward_shaping = reward_shaping)

    #env.reward = [ -(reward^2)]
    env.reward = [reward]
    
    y = create_state(; env = env, compute_left = compute_left, step = env.steps + 1)

    return y
end

function reward_function(env)
    return env.reward
end



function featurize(y0 = nothing, t0 = nothing; env = nothing)
    if isnothing(env)
        y = y0
    else
        y = env.y
    end

    return reshape(y, length(y), 1)
end

function prepare_action(action0 = nothing, t0 = nothing; env = nothing) 
    if isnothing(env)
        action =  action0
    else
        action = env.action
    end

    action = (action .+1) .*0.5

    clamp!(action, 0.0, 1.0)

    return action
end



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


        
        

        global agent = create_agent_sac(
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
                nna_scale = nna_scale,
                nna_scale_critic = nna_scale_critic,
                drop_middle_layer = drop_middle_layer,
                drop_middle_layer_critic = drop_middle_layer_critic,
                fun = fun,
                logσ_is_network = logσ_is_network,
                clip_grad = clip_grad,
                start_logσ = start_logσ,
                tanh_end = tanh_end,
                betas = betas,
                automatic_entropy_tuning = automatic_entropy_tuning,
                target_entropy = target_entropy,)


    global hook = GeneralHook(min_best_episode = min_best_episode,
                            collect_NNA = false,
                            generate_random_init = generate_random_init,
                            collect_history = false,
                            collect_rewards_all_timesteps = false,
                            early_success_possible = true)
end

function generate_random_init()
    y0 = create_state()

    env.y0 = deepcopy(y0)
    env.y = deepcopy(y0)
    env.state = env.featurize(; env = env)

    y0
end

initialize_setup()



function train(use_random_init = true; visuals = false, num_steps = 10_000, inner_loops = 3, optimal_trainings  = 0, outer_loops = 13360, only_wind_steps = 0, json = false)
    global wind_only, optimal_trajectory
    wind_only = false
    
    rm(dirpath * "/training_frames/", recursive=true, force=true)
    mkdir(dirpath * "/training_frames/")
    frame = 1

    if visuals
        colorscale = [[0, "rgb(34, 74, 168)"], [0.5, "rgb(224, 224, 180)"], [1, "rgb(156, 33, 11)"], ]
        ymax = 30
        layout = Layout(
                plot_bgcolor="#f1f3f7",
                coloraxis = attr(cmin = 0, cmid = 1, cmax = 2, colorscale = colorscale),
            )
    end

    if use_random_init
        hook.generate_random_init = generate_random_init
    else
        hook.generate_random_init = false
    end


    if optimal_trainings > 0 &&  isnothing(optimal_trajectory)
        fill_optimal_trajectory()
    end
    
    global logs = []
    global validation_scores = []
    global agent_save = nothing

    for j = 1:outer_loops


        for i in 1:optimal_trainings
            RL._update!(agent.policy, optimal_trajectory)
        end


        if only_wind_steps > 0
            println("")
            println("Starting only wind learning...")
            stop_condition = StopAfterEpisodeWithMinSteps(only_wind_steps)

            global grid_price
            grid_price = ones(size(grid_price))

            # run start
            hook(PRE_EXPERIMENT_STAGE, agent, env)
            agent(PRE_EXPERIMENT_STAGE, env)
            is_stop = false
            while !is_stop
                reset!(env)
                agent(PRE_EPISODE_STAGE, env)
                hook(PRE_EPISODE_STAGE, agent, env)



                while !is_terminated(env) # one episode
                    action = agent(env)

                    agent(PRE_ACT_STAGE, env, action)
                    hook(PRE_ACT_STAGE, agent, env, action)

                    env(action)

                    agent(POST_ACT_STAGE, env)
                    hook(POST_ACT_STAGE, agent, env)

                    if visuals
                        p = plot(heatmap(z=env.y[1,:,:], coloraxis="coloraxis"), layout)

                        savefig(p, dirpath * "/training_frames//a$(lpad(string(frame), 5, '0')).png"; width=1000, height=800)
                    end

                    frame += 1

                    if stop_condition(agent, env)
                        is_stop = true
                        break
                    end
                end # end of an episode

                if is_terminated(env)
                    agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
                    hook(POST_EPISODE_STAGE, agent, env)
                end
            end
            hook(POST_EXPERIMENT_STAGE, agent, env)
        end


        for i = 1:inner_loops
            println("")
            stop_condition = StopAfterEpisodeWithMinSteps(num_steps)


            # run start
            hook(PRE_EXPERIMENT_STAGE, agent, env)
            agent(PRE_EXPERIMENT_STAGE, env)
            is_stop = false
            while !is_stop
                reset!(env)
                agent(PRE_EPISODE_STAGE, env)
                hook(PRE_EPISODE_STAGE, agent, env)

                while !is_terminated(env) # one episode
                    action = agent(env)

                    agent(PRE_ACT_STAGE, env, action)
                    hook(PRE_ACT_STAGE, agent, env, action)

                    env(action)

                    agent(POST_ACT_STAGE, env)
                    hook(POST_ACT_STAGE, agent, env)

                    if visuals
                        p = plot(heatmap(z=env.y[1,:,:], coloraxis="coloraxis"), layout)

                        savefig(p, dirpath * "/training_frames//a$(lpad(string(frame), 5, '0')).png"; width=1000, height=800)
                    end

                    frame += 1

                    if stop_condition(agent, env)
                        is_stop = true
                        break
                    end
                end # end of an episode

                if is_terminated(env)
                    agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
                    hook(POST_EPISODE_STAGE, agent, env)

                    if json
                        push!(logs, Dict(
                            "episode" => length(hook.rewards),
                            "reward_ep" => hook.rewards[end],
                            "actor_loss" => agent.policy.last_actor_loss,
                            "critic1_loss" => agent.policy.last_critic1_loss,
                            "critic2_loss" => agent.policy.last_critic2_loss,
                            "log_alpha" => agent.policy.log_α[1],
                            "q1_mean" => agent.policy.last_q1_mean,
                            "q2_mean" => agent.policy.last_q2_mean,
                            "target_q_mean" => agent.policy.last_target_q_mean,
                            "mean_minus_log_pi" => agent.policy.last_mean_minus_log_pi,
                        ))
                    end
                end
            end
            hook(POST_EXPERIMENT_STAGE, agent, env)
            # run end


            println(hook.bestreward)

            if @isdefined(validate_agent)
                current_score = mean(validate_agent())

                if !isempty(validation_scores) && current_score > maximum(validation_scores)
                    agent_save = deepcopy(agent)
                end
                
                push!(validation_scores, current_score)
            end

            if !isempty(validation_scores)
                println(lineplot(validation_scores, title="Validation scores", xlabel="Episode", ylabel="Score", color=:cyan))
            end

            # hook.rewards = clamp.(hook.rewards, -3000, 0)

            
        end

        p1 = render_run(; exploration = true, plot_values = true)
        #p2 = plot_critic(; return_plot = true)
        #display([p1 p2])
        #display(p1)

    end

    if visuals && false
        rm(dirpath * "/training.mp4", force=true)
        run(`ffmpeg -framerate 16 -i $(dirpath * "/training_frames/a%05d.png") -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le $(dirpath * "/training.mp4")`)
    end

    #save()
end


#train()
#train(;num_steps = 140)
#train(;visuals = true, num_steps = 70)


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env")
    else
        global hook = FileIO.load(dirpath * "/saves/hook$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent$number.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env$number.jld2","env")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/hook.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env.jld2","env",env)
    else
        FileIO.save(dirpath * "/saves/hook$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent$number.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env$number.jld2","env",env)
    end
end



function render_run(; plot_optimal = false, steps = 6000, show_training_episode = false, show_σ = false, exploration = false, return_plot = false, plot_values = false, plot_critic2 = false, critic2_diagnostics = false, json = false)
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

    global results = Dict("rewards" => [], "loadleft" => [])

    for k in 1:n_windCORES
        results["hpc$k"] = []
        results["σ$k"] = []
    end

    q1 = []
    q2 = []
    states = []
    terminals = []

    reset!(env)
    generate_random_init()

    global run_logs = []

    while !env.done

        if exploration
            action = agent(env)
        else
            action = agent.policy.actor.μ(env.state)
        end
        

        #action = agent(env)
        step_dict = Dict()

        push!(q1, agent.policy.qnetwork1(vcat(env.state, action))[1])
        push!(q2, agent.policy.qnetwork2(vcat(env.state, action))[1])

        if json
            step_dict["step"] = env.steps
            step_dict["state"] = env.state
            step_dict["action"] = action
            step_dict["q1"] = q1[end]
            step_dict["q2"] = q2[end]
        end

        env(action)

        push!(terminals, env.done)

        if json
            step_dict["next_state"] = env.state
            step_dict["terminated"] = env.done
            step_dict["reward"] = env.reward[1]
            push!(run_logs, step_dict)
        end

        for k in 1:n_windCORES
            push!(results["hpc$k"], clamp((action[1]+1)*0.5, 0, 1)) #env.p[k])
            #push!(results["σ$k"], σ[k])
        end
        push!(results["rewards"], env.reward[1])
        push!(results["loadleft"], env.y[1])

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
            push!(to_plot, scatter(x=xx, y=results["σ$k"], name="σ$k", yaxis = "y2"))
        end
    else
        push!(to_plot, scatter(x=xx, y=results["rewards"], name="Reward", yaxis = "y2"))
    end

    if plot_values
        push!(to_plot, scatter(x=xx, y=q1, name="q1", yaxis = "y2"))
        push!(to_plot, scatter(x=xx, y=q2, name="q2", yaxis = "y2"))
    end

    push!(to_plot, scatter(x=xx, y=results["loadleft"], name="Load Left"))
    push!(to_plot, scatter(x=xx, y=grid_price[history_steps:end], name="Grid Price"))


    for k in 1:n_windCORES
        push!(to_plot, scatter(x=xx, y=results["hpc$k"], name="WindCORE utilization $k"))
        #line=attr(color = "rgba(200, 200, 200, 0.3)")))
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

        results = zeros(Float32, length(actions), length(states))

        for (i,state) in enumerate(states)
            inputs = vcat(repeat(state, 1, length(actions)), actions')

            mu = agent.policy.approximator.actor.μ(state)[:]
            mu_value = agent.policy.approximator.critic2(vcat(state, mu))
            mu_values = mu_value .* ones(length(actions))

            critic2_values = agent.policy.approximator.critic2(inputs)[:] #-1 first

            results[:,i] = critic2_values - mu_values
        end

        results = (results .- mean(results)) ./ clamp(std(results), 1e-8, 1000.0)

        min_val = - maximum(abs.(results))

        for (i,state) in enumerate(states)

            if critic2_diagnostics
                idx = clamp(searchsortedfirst(actions, state[end-1] * 2 - 1), 1, length(actions))

                idx2 = findmax(results[:,i])[2]
                results[idx2,i] = -min_val
            else
                idx = clamp(searchsortedfirst(actions, mus[i]), 1, length(actions))
            end

            results[idx,i] = min_val
        end

        display(plot(heatmap(x = xx, y = actions, z=results, coloraxis="coloraxis"), layout))

    end
end

# t1 = scatter(y=rewards1)
# t2 = scatter(y=rewards2)
# t3 = scatter(y=rewards3)
# plot([t1, t2, t3])



function optimize_day(steps = 3000; verbose = true)
    model = Model(Ipopt.Optimizer)

    if !verbose
        set_silent(model)
    end

    set_optimizer_attribute(model, "max_iter", steps)

    @variable(model, -1 <= x[1:n_windCORES, 1:Int(te/dt)] <= 1)

    @constraint(model, sum((x .+1) .*0.5) == 100.0)

    @objective(model, Max, evaluate((x .+1) .*0.5))

    optimize!(model)

    return value.(x)
end





# sum(actions) has to be 100

function evaluate(actions; collect_rewards = false, reward_shaping = false)
    step = 2

    reward_sum = 0.0
    global rewards = Float64[]

    for t in 1:Int(te/dt)

        reward, _ = calculate_day(actions[:,t], nothing, t-1; reward_shaping = reward_shaping)

        reward_sum += reward

        if collect_rewards
            push!(rewards, reward)
        end

        step += 1
    end

    if collect_rewards
        rewards
    else
        reward_sum
    end
end

# train(num_steps = 14300, inner_loops = 2, optimized_episodes = 20, outer_loops = 100)

function plot_rewards(smoothing = 30)
    to_plot = Float64[]
    for i in smoothing:length(hook.rewards)
        push!(to_plot, mean(hook.rewards[i+1-smoothing:i]))
    end

    p = plot(to_plot)
    display(p)
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