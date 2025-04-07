using LinearAlgebra
using IntervalSets
using StableRNGs
using SparseArrays
using Conda
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
#using Blink

n_turbines = 1


scriptname = "Minimal1"




#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
end




# action vector dim - contains the percentage of maximum power the HPC in the turbine will use for the duration of next time step

action_dim = n_turbines

# state vector

# - amount of computation left (starts at 1.0 and goes to 0.0)
# - wind stituation at every turbine (gradient of power output, current power output and curtailment enegry)
# - current gradient and price of energy from the grid
# - current time

state_dim = 4 + 3*n_turbines



# env parameters

seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false

te = 1440.0
dt = 5.0
t0 = 0.0
min_best_episode = 1

sim_space = Space(fill(0..1, (state_dim)))

function generate_wind()
    wind_constant_day = rand()
    deviation = 1/5

    result = sign(randn()) * sin.(collect(LinRange(rand()*3+1, 4+rand()*4, Int(te/dt)+1)))

    for i in 1:4
        result += sign(randn()) * sin.(collect(LinRange(rand()+4, 5+rand()*i*4, Int(te/dt)+1)))
    end

    result .-= minimum(result)
    result ./= maximum(result)
    result .*= deviation

    day_wind = sign(randn()) * sin.(collect(LinRange(wind_constant_day*2*pi, 2+wind_constant_day*2*pi, Int(te/dt)+1)))
    day_wind .+= 1.0
    day_wind ./= 4
    day_wind .+= 0.25


    result .+= day_wind

    clamp!(result, -1.0, 1.0)

    result
end

function generate_grid_price()

    factor = 1.0;
    factor = 0.6;

    gp = (-sin.(collect(LinRange(rand()*1.5*factor, 2+rand()*2.5*factor, Int(te/dt)+1))) .+(1+(rand()*factor)))

    clamp!(gp, -1, 1)

    return gp
end

y0 = [1.0]

wind = [generate_wind() for i in 1:n_turbines]

# layout = Layout(
#                 plot_bgcolor="#f1f3f7",
#                 yaxis=attr(range=[0,1]),
#             )

# to_plot = [scatter(y=wind[i]) for i in 1:1]
# plot(Vector{AbstractTrace}(to_plot), layout)

grid_price = generate_grid_price()
plot(scatter(y=grid_price), Layout(yaxis=attr(range=[0,1])))

for i in 1:n_turbines
    push!(y0, wind[i][2] - wind[i][1])
    push!(y0, wind[i][2])
    push!(y0, max(0.0, wind[i][2] - 0.4))
end

push!(y0, grid_price[2] - grid_price[1])
push!(y0, grid_price[2])

push!(y0, 0.0)

y0 = Float32.(y0)


# agent tuning parameters
memory_size = 0
nna_scale = 20.0
nna_scale_critic = 11.0
drop_middle_layer = false
drop_middle_layer_critic = false
fun = gelu
use_gpu = false
actionspace = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.997f0
p = 0.95f0

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 100


learning_rate = 1e-4
n_epochs = 3
n_microbatches = 10
logσ_is_network = false
max_σ = 10000.0f0
entropy_loss_weight = 0.01
clip_grad = 0.3
target_kl = 0.1
clip1 = false
start_logσ = -0.5
tanh_end = false
clip_range = 0.05f0


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


function do_step(env)
    y = [ env.y[1] ]
    step = env.steps + 2

    compute_power = 0.0
    for i in 1:n_turbines
        compute_power += env.p[i]*0.01
    end

    # subtracting the computed load
    compute_power_used = min(y[1], compute_power)
    y[1] -= compute_power
    y[1] = max(y[1], 0.0)

    if y[1] == 0.0
        env.done = true
    end

    #normalizing
    compute_power_used *= 100/n_turbines

    # reward calculation
    power_for_free = 0.0
    for i in 1:n_turbines

        # curtailment energy onlny when wind is above 0.4
        temp_free_power = (wind[i][step-1] - 0.4)
        temp_free_power = max(0.0, temp_free_power)

        power_for_free += temp_free_power
    end
    power_for_free_used = min(power_for_free, compute_power_used)
    compute_power_used -= power_for_free
    # compute_power_used = max(0.0, compute_power_used)
    compute_power_used = softplus_shifted(compute_power_used)


    #normalizing
    compute_power_used *= (n_turbines * 0.01)
    

    reward1 = compute_power_used * grid_price[step-1]

    reward = - reward1

    if (env.time + env.dt) >= env.te 
        reward -= y[1] * 1
    else
        #reward shaping
        #reward = (-1) * abs((reward * 45))^2.2

        #delta_action punish
        # reward -= 0.002 * mean(abs.(env.delta_action))
        #clamp!(env.reward, -1.0, 0.0)
    end

    env.reward = [ -(reward^2)]
    

    
    for i in 1:n_turbines
        push!(y, wind[i][step] - wind[i][step-1])
        push!(y, wind[i][step])
        push!(y, max(0.0, wind[i][step] - 0.4))
    end

    push!(y, grid_price[step] - grid_price[step-1])
    push!(y, grid_price[step])

    push!(y, env.time / env.te)

    

    y = Float32.(y)

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

    clamp!(action, -1.0, 1.0)

    action = (action .+1) .*0.5

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

        global agent = create_agent_ppo(action_space = actionspace,
                state_space = env.state_space,
                use_gpu = use_gpu, 
                rng = rng,
                y = y, p = p,
                update_freq = update_freq,
                learning_rate = learning_rate,
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
                clip_range = clip_range)


    global hook = GeneralHook(min_best_episode = min_best_episode,
                            collect_NNA = false,
                            generate_random_init = generate_random_init,
                            collect_history = false,
                            collect_rewards_all_timesteps = false,
                            early_success_possible = true)
end

function generate_random_init()
    global wind_constant_day, wind, grid_price

    y0 = [1.0]

    wind = [generate_wind() for i in 1:n_turbines]

    grid_price = generate_grid_price()

    for i in 1:n_turbines
        push!(y0, wind[i][2] - wind[i][1])
        push!(y0, wind[i][2])
        push!(y0, max(0.0, wind[i][2] - 0.4))
    end
    
    push!(y0, grid_price[2] - grid_price[1])
    push!(y0, grid_price[2])
    
    push!(y0, 0.0)

    y0 = Float32.(y0)

    env.y0 = deepcopy(y0)
    env.y = deepcopy(y0)
    env.state = env.featurize(; env = env)

    y0
end

initialize_setup()

# plotrun(use_best = false, plot3D = true)

function train(use_random_init = true; visuals = false, num_steps = 10_000, inner_loops = 1, optimized_episodes  = 4, outer_loops = 10, steps = 2000, only_wind_steps = 0)
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
    

    for j = 1:outer_loops
        
        println("")
        println("Starting optimized episodes learning...")

        for i in 1:optimized_episodes

            # run start
            agent(PRE_EXPERIMENT_STAGE, env)
            is_stop = false
            while !is_stop
                println("Optimized Episode $(i)...")
                reset!(env)
                agent(PRE_EPISODE_STAGE, env)

                env.y0 = generate_random_init()
                env.y = deepcopy(env.y0)
                env.state = env.featurize(; env = env)

                # generate optimal actions
                optimal_actions = optimize_day(steps; verbose = false)
                n = 1

                while !is_terminated(env) # one episode
                    # action = agent(env)

                    if n <= size(optimal_actions)[2]
                        action = optimal_actions[:,n]
                    else
                        # just in case y[1] is not exactly 0.0 due to numerical errors
                        action = [ 0.001 ]
                    end

                    agent(PRE_ACT_STAGE, env, action)

                    env(action)

                    agent(POST_ACT_STAGE, env)

                    if visuals
                        p = plot(heatmap(z=env.y[1,:,:], coloraxis="coloraxis"), layout)

                        savefig(p, dirpath * "/training_frames//a$(lpad(string(frame), 5, '0')).png"; width=1000, height=800)
                    end

                    frame += 1
                    n += 1
                end # end of an episode

                if is_terminated(env)
                    agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
                end

                is_stop = true
            end
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
                end
            end
            hook(POST_EXPERIMENT_STAGE, agent, env)
            # run end


            println(hook.bestreward)

            # hook.rewards = clamp.(hook.rewards, -3000, 0)

            render_run() #;plot_optimal=true, steps=2000)
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


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook_rhoch2.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent_rhoch2.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env")
    else
        global hook = FileIO.load(dirpath * "/saves/hook_rhoch2_$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent_rhoch2_$number.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env$number.jld2","env")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/hook_rhoch2.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent_rhoch2.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env.jld2","env",env)
    else
        FileIO.save(dirpath * "/saves/hook_rhoch2_$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent_rhoch2_$number.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env$number.jld2","env",env)
    end
end



function render_run(use_best = false; plot_optimal = false, steps = 6000, show_training_episode = true)

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


    agent.policy.update_step = 0
    global rewards = Float64[]
    reward_sum = 0.0

    #w = Window()

    xx = collect(0:(1/12):23+(11/12))

    results = Dict("rewards" => [], "loadleft" => [])

    for k in 1:n_turbines
        results["hpc$k"] = []
    end

    global currentDF = DataFrame()

    reset!(env)
    generate_random_init()

    while !env.done
        action = prob(agent.policy, env).μ

        #action = env.y[6] < 0.27 ? [-1.0] : [1.0]

        env(action)

        for k in 1:n_turbines
            push!(results["hpc$k"], env.p[k])
        end
        push!(results["rewards"], env.reward[1])
        push!(results["loadleft"], env.y[1])

        # println(mean(env.reward))

        reward_sum += mean(env.reward)
        # push!(rewards, mean(env.reward))

        tmp = DataFrame()
        insertcols!(tmp, :timestep => env.steps)
        insertcols!(tmp, :action => [vec(env.action)])
        insertcols!(tmp, :p => [send_to_host(env.p)])
        insertcols!(tmp, :y => [send_to_host(env.y)])
        insertcols!(tmp, :reward => [reward(env)])
        append!(hook.currentDF, tmp)
    end

    # if use_best
    #     copyto!(agent.policy.behavior_actor, hook.currentNNA)
    # end

    # agent.policy.start_steps = temp_start_steps
    # agent.policy.act_noise = temp_noise
    # agent.policy.update_after = temp_update_after

    println(reward_sum)


    layout = Layout(
                    plot_bgcolor = "white",
                    font=attr(
                        family="Arial",
                        size=16,
                        color="black"
                    ),
                    showlegend = true,
                    legend=attr(x=0.5, y=1.1, orientation="h", xanchor="center"),
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

    

    to_plot = [scatter(x=xx, y=results["rewards"], name="Reward", yaxis = "y2"),
                scatter(x=xx, y=results["loadleft"], name="Load Left"),
                scatter(x=xx, y=grid_price, name="Grid Price")]

    for k in 1:n_turbines
        push!(to_plot, scatter(x=xx, y=results["hpc$k"], name="WindCORE utilization $k"))
        push!(to_plot, scatter(x=xx, y=wind[k], name="Wind Power $k"))
    end

    if plot_optimal
        global optimal_actions = optimize_day(steps)
        global optimal_rewards = evaluate(optimal_actions; collect_rewards = true)

        for k in 1:n_turbines
            push!(to_plot, scatter(x=xx, y=optimal_actions[k,:], name="Optimal HPC$k"))
        end
        push!(to_plot, scatter(x=xx, y=optimal_rewards, name="Optimal Reward", yaxis = "y2"))


        println("")
        println("--------------------------------------------")
        println("AGENT:   $reward_sum")
        println("IPOPT:   $(sum(optimal_rewards))")
        println("--------------------------------------------")
    end

    p = plot(Vector{AbstractTrace}(to_plot), layout)
    display(p)

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

    @variable(model, 0 <= x[1:n_turbines, 1:Int(te/dt)] <= 1)

    @constraint(model, sum(x) == 100.0)

    @objective(model, Max, evaluate(x))

    optimize!(model)

    return value.(x)
end





# sum(actions) has to be 100

function evaluate(actions; collect_rewards = false)
    step = 2

    reward_sum = 0.0
    global rewards = Float64[]

    for t in 1:Int(te/dt)

        compute_power = 0.0
        for i in 1:n_turbines
            compute_power += actions[i,t]
        end

        compute_power_used = compute_power

        #normalizing
        compute_power_used /= n_turbines

        # reward calculation
        power_for_free = 0.0
        for i in 1:n_turbines

            # curtailment energy onlny when wind is above 0.4
            temp_free_power = (wind[i][step-1] - 0.4)
            temp_free_power = max(0.0, temp_free_power)

            power_for_free += temp_free_power
        end
        power_for_free_used = min(power_for_free, compute_power_used)

        # Hack for the Optimizer
        #compute_power_used -= power_for_free_used - 0.0000000001
        #power_for_free_used += 0.0000000001

        compute_power_used -= power_for_free
        #compute_power_used = max(0.0000001, compute_power_used)
        compute_power_used = softplus_shifted(compute_power_used)
        #compute_power_used = smoothedReLu(compute_power_used)
        
        #normalizing
        compute_power_used *= (n_turbines * 0.01)

        reward1 = compute_power_used * grid_price[step-1]

        reward = - (reward1^2)

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