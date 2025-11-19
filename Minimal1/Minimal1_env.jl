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


validation_scores = []


te = 1440.0
dt = 5.0
t0 = 0.0
min_best_episode = 1


# default - will be overwritten in most training scripts
reward_shaping = true
reward_shaping_beta = 1.0f0



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

function generate_wind(; same_day = false)
    global history_steps, te, dt

    wind_steps = Int(te/dt) + history_steps

    if same_day
        rand1 = 0.6
        rand2 = 0.2
        rand3 = 0.3
        rand4 = 0.3
        rand5 = 0.3
        rand6 = 0.3
        rand7 = 0.3
        rand8 = 0.1
    else
        rand1 = rand()
        rand2 = randn()
        rand3 = rand()
        rand4 = rand()
        rand5 = randn()
        rand6 = rand()
        rand7 = rand()
        rand8 = randn()
    end

    #@show rand1, rand2, rand3, rand4, rand5, rand6, rand7, rand8

    wind_constant_day = rand1
    deviation = 1/5

    result = sign(rand2) * sin.(collect(LinRange(rand3*3+1, 4+rand4*4, wind_steps)))

    for i in 1:4
        result += sign(rand5) * sin.(collect(LinRange(rand6+4, 5+rand7*i*4, wind_steps)))
    end

    result .-= minimum(result)
    result ./= maximum(result)
    result .*= deviation

    day_wind = sign(rand8) * sin.(collect(LinRange(wind_constant_day*2*pi, 2+wind_constant_day*2*pi, wind_steps)))
    day_wind .+= 1.0
    day_wind ./= 4
    day_wind .+= 0.25


    result .+= day_wind

    clamp!(result, -1.0, 1.0)

    result
end

function generate_grid_price(; same_day = false)
    global history_steps, te, dt

    if same_day
        rand1 = 0.0
        rand2 = 0.2
        rand3 = 0.3
    else
        rand1 = rand()
        rand2 = rand()
        rand3 = rand()
    end

    #@show rand1, rand2, rand3

    grid_price_steps = Int(te/dt) + history_steps

    factor = 1.0;
    factor = 0.6;

    gp = (-sin.(collect(LinRange(rand1*1.5*factor, 2+rand2*2.5*factor, grid_price_steps))) .+(1+(rand3*factor)))

    clamp!(gp, -1, 1)

    #gp = ones(grid_price_steps)

    return gp
end

include_history_steps = 1
include_gradients = 2

function create_state(; env = nothing, compute_left = 1.0, step = 0, generate_day = true, same_day = false)
    global wind, grid_price, curtailment_threshold, history_steps, dt, include_history_steps, include_gradients


    if isnothing(env)
        y = [1.0]

        if generate_day
            wind = [generate_wind(; same_day = same_day) for i in 1:n_turbines]
            grid_price = generate_grid_price(; same_day = same_day)
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

include("./Validation_Minimal1.jl")



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

function calculate_day(action, env, step = nothing; reward_shaping = reward_shaping)
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
                compute_left_after = 0.0f0 
            end
        end
    end

    if reward_shaping
        # potential based reward shaping
        beta = reward_shaping_beta
        reward += beta * (compute_left_before - compute_left_after - (gamma-1) * compute_left_after)

        reward *= reward_scale_factor
    end

    return reward, compute_left_after
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



function generate_random_init(; same_day = false)
    
    y0 = create_state(; same_day = same_day)

    env.y0 = deepcopy(y0)
    env.y = deepcopy(y0)
    env.state = env.featurize(; env = env)

    y0
end



# IPOPT Score for same_day: -0.16820833350564535

function train(use_random_init = true; visuals = false, num_steps = 10_000, inner_loops = 1, optimal_trainings  = 0, outer_loops = 5000, only_wind_steps = 0, json = false, reward_shaping = reward_shaping, plot_runs = true, same_day = false)
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


    if optimal_trainings > 0
        @assert isdefined(@__MODULE__, :optimal_trajectory) && !isnothing(optimal_trajectory) "optimal_trajectory ist nicht initialisiert"
    end
    
    global logs = []
    global validation_scores
    global agent_save

    

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

                if same_day
                    env.y0 = generate_random_init(; same_day = true)
                    env.y = deepcopy(env.y0)

                    env.state = env.featurize(; env = env)
                else
                    hook(PRE_EPISODE_STAGE, agent, env)
                end

                while !is_terminated(env) # one episode
                    action = agent(env)

                    agent(PRE_ACT_STAGE, env, action)
                    hook(PRE_ACT_STAGE, agent, env, action)

                    env(action; reward_shaping = reward_shaping)

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

                println("Best validation score: $(maximum(validation_scores))")
            end

            # hook.rewards = clamp.(hook.rewards, -3000, 0)

            
        end

        if plot_runs
            p1 = render_run(; exploration = true, new_day = !same_day)#, plot_values = true)
            #p2 = plot_critic(; return_plot = true)
            #display([p1 p2])
            #display(p1)
        end

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


function load_agent(number = nothing)
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

function save_agent(number = nothing)
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
