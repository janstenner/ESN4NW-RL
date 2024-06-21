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
# - wind stituation at every turbine (gradient of power output and current power output)
# - current gradtient and price of energy from the grid
# - current time

state_dim = 4 + 2*n_turbines



# env parameters

seed = Int(floor(rand()*1000))
seed = 178

gpu_env = false

te = 1440.0
dt = 5.0
t0 = 0.0
min_best_episode = 1

sim_space = Space(fill(0..1, (state_dim)))

function generate_wind()
    wind_constant_day = rand()
    deviation = 1/5

    result = sin.(collect(LinRange(rand()*3+1, 4+rand()*4, Int(te/dt)+1)))

    for i in 1:4
        result += sin.(collect(LinRange(rand()+4, 5+rand()*i*4, Int(te/dt)+1)))
    end

    result .-= minimum(result)
    result ./= maximum(result)
    result .*= deviation

    day_wind = sin.(collect(LinRange(wind_constant_day*2*pi, 2+wind_constant_day*2*pi, Int(te/dt)+1)))
    day_wind .+= 1.0
    day_wind ./= 4
    day_wind .+= 0.25


    result .+= day_wind

    clamp!(result, -1.0, 1.0)

    result
end

function generate_grid_price()

    gp = (-sin.(collect(LinRange(rand()*1.5, 2+rand()*1.5, Int(te/dt)+1))) .+(1+rand()))

    clamp!(gp, -1, 1)

    return gp
end

y0 = zeros(state_dim)
y0[1] = 1.0

wind = [generate_wind() for i in 1:n_turbines]

# layout = Layout(
#                 plot_bgcolor="#f1f3f7",
#                 yaxis=attr(range=[0,1]),
#             )

# to_plot = [scatter(y=wind[i]) for i in 1:3]
# plot(Vector{AbstractTrace}(to_plot), layout)

grid_price = generate_grid_price()
# plot(grid_price)

for i in 1:n_turbines
    y0[((i-1)*2)+2] = wind[i][2] - wind[i][1]
    y0[((i-1)*2)+3] = wind[i][2] 
end

y0[1 + n_turbines * 2 + 1] = grid_price[2] - grid_price[1]
y0[1 + n_turbines * 2 + 2] = grid_price[2]

y0 = Float32.(y0)


# agent tuning parameters
memory_size = 0
nna_scale = 5.0
nna_scale_critic = 5.0
drop_middle_layer = false
drop_middle_layer_critic = false
fun = leakyrelu
use_gpu = false
actionspace = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.95f0
batch_size = 10
start_steps = -1
start_policy = ZeroPolicy(actionspace)
update_after = 10
update_freq = 288
update_loops = 10
reset_stage = POST_EPISODE_STAGE
learning_rate = 0.0002
learning_rate_critic = 0.0005
act_limit = 1.0
act_noise = 1.2
trajectory_length = 500_000

advv = FileIO.load("adv.jld2","adv")

function do_step(env)
    y = env.y
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


    # reward calculation
    power_for_free = 0.0
    for i in 1:n_turbines

        # curtailment energy onlny when wind is above 0.4
        temp_free_power = (wind[i][step-1] - 0.4)*0.01
        temp_free_power = max(0.0, temp_free_power)

        power_for_free += temp_free_power
    end
    power_for_free_used = min(power_for_free, compute_power_used)
    compute_power_used -= power_for_free
    compute_power_used = max(0.0, compute_power_used)

    reward1 = (50 * compute_power_used)^0.9 * ((grid_price[step-1] + 0.2)^2) * 0.5 - 0.3 * compute_power_used * 70

    reward2 = - (37 * compute_power_used^1.2) * (1-grid_price[step-1]*2)

    #factor = clamp(grid_price[step-1] * 2 - 0.5, 0.0, 1.0)
    #factor = sigmoid(grid_price[step-1] * 9 - 4.0)
    factor = 1

    reward_free = (power_for_free_used * 40)^1.2 + (grid_price[step-1])^1.2 * power_for_free_used * 10

    reward = - (factor * reward1 + (1 - factor) * reward2) + reward_free

    if (env.time + env.dt) >= env.te 
        reward -= y[1] * 100
        env.reward = [reward]
    else
        #reward shaping
        #reward = (-1) * abs((reward * 45))^2.2

        #delta_action punish
        # reward -= 0.002 * mean(abs.(env.delta_action))
        env.reward = [reward]
        #clamp!(env.reward, -1.0, 0.0)
    end
    

    
    for i in 1:n_turbines
        y[((i-1)*2)+2] = wind[i][step] - wind[i][step-1]
        y[((i-1)*2)+3] = wind[i][step]
    end

    y[1 + n_turbines * 2 + 1] = grid_price[step] - grid_price[step-1]
    y[1 + n_turbines * 2 + 2] = grid_price[step]

    y[1 + n_turbines * 2 + 3] = env.time / env.te

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
                oversampling = 1,
                use_radau = false,
                max_value = 1.0,
                check_max_value = "nothing")

        global agent = create_agent_ppo(action_space = actionspace,
                state_space = env.state_space,
                use_gpu = use_gpu, 
                rng = rng,
                y = y, p = p,
                update_freq = update_freq,
                nna_scale = nna_scale,
                nna_scale_critic = nna_scale_critic,
                drop_middle_layer = drop_middle_layer,
                drop_middle_layer_critic = drop_middle_layer_critic,
                fun = fun,
                clip1 = true)

    # global agent = create_agent_ppo(mono = true,
    #                     action_space = actionspace,
    #                     state_space = env.state_space,
    #                     use_gpu = use_gpu, 
    #                     rng = rng,
    #                     y = y, p = p, batch_size = batch_size, 
    #                     start_steps = start_steps, 
    #                     start_policy = start_policy,
    #                     update_after = update_after, 
    #                     update_freq = update_freq,
    #                     update_loops = update_loops,
    #                     reset_stage = reset_stage,
    #                     act_limit = act_limit, 
    #                     act_noise = act_noise,
    #                     nna_scale = nna_scale,
    #                     nna_scale_critic = nna_scale_critic,
    #                     drop_middle_layer = drop_middle_layer,
    #                     drop_middle_layer_critic = drop_middle_layer_critic,
    #                     fun = fun,
    #                     memory_size = memory_size,
    #                     trajectory_length = trajectory_length,
    #                     learning_rate = learning_rate,
    #                     learning_rate_critic = learning_rate_critic)

    global hook = GeneralHook(min_best_episode = min_best_episode,
                            collect_NNA = false,
                            generate_random_init = generate_random_init,
                            collect_history = false,
                            collect_rewards_all_timesteps = false,
                            early_success_possible = true)
end

function generate_random_init()
    global wind_constant_day, wind, grid_price

    y0 = zeros(state_dim)
    y0[1] = 1.0

    wind = [generate_wind() for i in 1:n_turbines]

    grid_price = generate_grid_price()

    for i in 1:n_turbines
        y0[((i-1)*2)+2] = wind[i][2] - wind[i][1]
        y0[((i-1)*2)+3] = wind[i][2] 
    end
    
    y0[1 + n_turbines * 2 + 1] = grid_price[2] - grid_price[1]
    y0[1 + n_turbines * 2 + 2] = grid_price[2]

    y0 = Float32.(y0)

    env.y0 = deepcopy(y0)
    env.y = deepcopy(y0)
    env.state = env.featurize(; env = env)

    y0
end

initialize_setup()

# plotrun(use_best = false, plot3D = true)

function train(use_random_init = true; visuals = false, num_steps = 288, inner_loops = 1)
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
    

    
    outer_loops = 1

    for i = 1:outer_loops
        
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
        end
    end

    if visuals && false
        rm(dirpath * "/training.mp4", force=true)
        run(`ffmpeg -framerate 16 -i $(dirpath * "/training_frames/a%05d.png") -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le $(dirpath * "/training.mp4")`)
    end

    save()
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



function render_run(use_best = false; plot_optimal = false, steps = 6000)
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

    results = Dict("rewards" => [], "loadleft" => [])

    for k in 1:n_turbines
        results["hpc$k"] = []
    end

    global currentDF = DataFrame()

    reset!(env)
    generate_random_init()

    while !env.done
        action = agent(env)

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
                    plot_bgcolor="#f1f3f7",
                    yaxis=attr(range=[0,1]),
                    yaxis2 = attr(
                        overlaying="y",
                        side="right",
                        titlefont_color="orange",
                        #range=[-1, 1]
                    ),
                )

    

    to_plot = [scatter(y=results["rewards"], name="reward", yaxis = "y2"),
                scatter(y=results["loadleft"], name="load left"),
                scatter(y=grid_price, name="grid price")]

    for k in 1:n_turbines
        push!(to_plot, scatter(y=results["hpc$k"], name="hpc$k"))
        push!(to_plot, scatter(y=wind[k], name="wind$k"))
    end

    if plot_optimal
        optimal_actions = optimize_day(steps)
        optimal_rewards = evaluate(optimal_actions; collect_rewards = true)

        for k in 1:n_turbines
            push!(to_plot, scatter(y=optimal_actions[k,:], name="optimal_hpc$k"))
        end
        push!(to_plot, scatter(y=optimal_rewards, name="optimal_reward", yaxis = "y2"))


        println("")
        println("--------------------------------------------")
        println("AGENT:   $reward_sum")
        println("IPOPT:   $(sum(optimal_rewards))")
        println("--------------------------------------------")
    end

    plot(Vector{AbstractTrace}(to_plot), layout)

end

# t1 = scatter(y=rewards1)
# t2 = scatter(y=rewards2)
# t3 = scatter(y=rewards3)
# plot([t1, t2, t3])



function optimize_day(steps = 3000)
    model = Model(Ipopt.Optimizer)

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
            compute_power += actions[i,t]*0.01
        end

        compute_power_used = compute_power

        # reward calculation
        power_for_free = 0.0
        for i in 1:n_turbines

            # curtailment energy onlny when wind is above 0.4
            temp_free_power = (wind[i][step-1] - 0.4)*0.01
            temp_free_power = max(0.0, temp_free_power)

            power_for_free += temp_free_power
        end
        power_for_free_used = min(power_for_free, compute_power_used)

        # Hack for the Optimizer
        #compute_power_used -= power_for_free_used - 0.0000000001
        #power_for_free_used += 0.0000000001

        compute_power_used -= power_for_free
        compute_power_used = max(0.0000001, compute_power_used)
        
        

        reward1 = compute_power_used * grid_price[step-1]

        reward = - reward1 

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

# train(num_steps = 100000)


# test = Float32[]

# for i in 1:1000000
#     push!(test, agent.policy(env)[1])
# end

rew = [0.10732127f0  0.060921825f0  0.07570994f0  0.083187185f0  0.008762053f0  0.091079764f0  0.07731481f0  0.024121685f0  0.102529384f0  0.0f0  0.05778408f0  0.0f0  0.0f0  0.08123535f0  0.045586165f0  0.10352385f0  0.08792342f0  0.101683445f0  0.084832415f0  0.10904671f0  0.08644019f0  0.09642662f0  0.067798294f0  0.0f0  0.09320033f0  0.08067376f0  0.09178088f0  0.107111536f0  0.10788016f0  0.0948415f0  0.095951885f0  0.096998334f0  0.08646075f0  0.101857156f0  0.099912286f0  0.055821583f0  0.10208863f0  0.10263626f0  0.104310155f0  0.10494222f0  0.10568493f0  0.1082907f0  0.0014879947f0  0.0f0  0.10905807f0  0.0f0  0.095533125f0  0.072396055f0  0.11087006f0  0.116754845f0  0.05048954f0  0.11457929f0  0.118194096f0  0.11034039f0  0.117701255f0  0.117818825f0  0.12533313f0  0.12552255f0  0.122025825f0  0.1214319f0  0.0130274715f0  0.076382786f0  0.13226405f0  0.119797185f0  0.12941921f0  0.13514195f0  0.10258139f0  0.13780467f0  0.13887821f0  0.12831128f0  0.13135089f0  0.14201672f0  0.1430332f0  0.13712767f0  0.13234434f0  0.14598414f0  0.0f0  0.0833771f0  0.09239931f0  0.14119264f0  0.14924663f0  0.13582896f0  0.14299177f0  0.14662938f0  0.13801213f0  0.0f0  0.0f0  0.0051841335f0  0.15670037f0  0.008406107f0  0.15800083f0  0.15819493f0  0.15919775f0  0.15975639f0  0.0f0  0.08126905f0  0.15470476f0  0.1448125f0  0.14178191f0  0.16252968f0  0.14312695f0  0.05680671f0  0.15619676f0  0.0f0  0.14655118f0  0.1642695f0  0.16445607f0  0.16461276f0  0.14964524f0  0.16483623f0  0.15168487f0  0.124345966f0  0.15384787f0  0.0f0  0.11717145f0  0.16203569f0  0.16131462f0  0.15423328f0  0.15846413f0  0.10951602f0  0.14724949f0  0.14823954f0  0.0f0  0.14242387f0  0.046255432f0  0.0f0  0.061573263f0  0.13424447f0  0.13495442f0  0.16057152f0  0.15572417f0  0.15952636f0  0.14763133f0  0.0f0  0.15776642f0  0.026836455f0  0.14708085f0  0.12821762f0  0.12522435f0  0.15434836f0  0.15359609f0  0.13315424f0  0.024759777f0  0.12225744f0  0.11931705f0  0.11812707f0  0.14864303f0  0.13069105f0  0.13483061f0  0.08215189f0  0.046871614f0  0.11753221f0  0.0f0  0.0f0  0.13641313f0  0.13999248f0  0.04559451f0  0.045504212f0  0.11884982f0  0.0f0  0.13468334f0  0.11904649f0  0.13248017f0  0.043718074f0  0.098462544f0  0.057257563f0  0.10053224f0  0.12681262f0  0.10233642f0  0.00025017877f0  0.089807525f0  0.009819962f0  0.08055573f0  0.107627064f0  0.086703844f0  0.09496794f0  0.111247525f0  0.0521256f0  -0.11057137f0  0.0f0  0.011098211f0  -0.10905207f0  -0.07451287f0  0.0f0  0.043607984f0  -0.10641843f0  -0.058080524f0  -0.0262301f0  -0.046872545f0  -0.034911335f0  -0.05766258f0  -0.10112977f0  -0.06192495f0  0.0f0  -0.09606597f0  -0.09670094f0  -0.019365806f0  -0.016391754f0  -0.033792153f0  -0.023754247f0  -0.00012078399f0  -0.08871583f0  0.029967409f0  -0.05381781f0  -0.033489965f0  0.020833997f0  -0.080872566f0  -0.025650911f0  -0.048145138f0  0.055523522f0  0.013038636f0  0.025618002f0  0.07030558f0  -0.032803524f0  0.0f0  0.055908322f0  0.032037072f0  -0.033383917f0  0.03313846f0  -0.021027384f0  -0.010832807f0  0.049334172f0  0.04761556f0  0.012285697f0  0.0f0  0.0f0  0.019682666f0  0.07796108f0  0.095983945f0  0.08517445f0  0.06319497f0  0.06810151f0  0.0f0  0.05550325f0  0.0f0  0.056298554f0  0.06831675f0  0.060046643f0  0.054761767f0  0.0f0  0.0628833f0  0.01784828f0  0.08048193f0  0.09232988f0  0.11231151f0  0.08318449f0  0.083938755f0  0.08417364f0  0.06656298f0  0.09731897f0  0.09415067f0  0.043542106f0  0.11359111f0  0.0f0  0.10587896f0  0.04783804f0  0.12498349f0  0.12605923f0  0.11375316f0  0.0f0  0.11760954f0  0.017810483f0  0.086778514f0  0.123808816f0  0.14042732f0  0.12928696f0  0.14089365f0  0.10371455f0  0.1426414f0  0.13094424f0  0.14328061f0  0.13757423f0  0.14003745f0  0.0f0  0.052359212f0  0.10192079f0  0.14414372f0  0.14527063f0  0.14633535f0  0.0f0  0.14827837f0  0.1491568f0  0.15361346f0  0.15123892f0  0.16509977f0  0.15346892f0  0.15261996f0  0.15668602f0]

val = [0.15635887f0  0.15562537f0  0.15434036f0  0.15309332f0  0.15241288f0  0.15129445f0  0.14996248f0  0.14927673f0  0.14832635f0  0.14772126f0  0.1469972f0  0.14641252f0  0.14583804f0  0.14457732f0  0.14391765f0  0.14306636f0  0.14199065f0  0.1411736f0  0.14005113f0  0.13932905f0  0.1382323f0  0.13739222f0  0.13678996f0  0.13632768f0  0.13541397f0  0.13481933f0  0.13376626f0  0.1332403f0  0.13272962f0  0.13187435f0  0.13108353f0  0.13037677f0  0.1300155f0  0.12951288f0  0.12875156f0  0.12834111f0  0.12736404f0  0.1266142f0  0.12582673f0  0.124897204f0  0.124008626f0  0.12339771f0  0.12313679f0  0.122886315f0  0.122223645f0  0.121982455f0  0.12153728f0  0.12114238f0  0.1206786f0  0.11976029f0  0.119420275f0  0.11881365f0  0.11798135f0  0.11753297f0  0.116879836f0  0.11631694f0  0.11542046f0  0.114659406f0  0.11408525f0  0.1134526f0  0.11323844f0  0.112821445f0  0.1116578f0  0.11109697f0  0.11010094f0  0.10905136f0  0.10851456f0  0.107404746f0  0.1062912f0  0.10552326f0  0.10458787f0  0.10336663f0  0.10228515f0  0.10158825f0  0.101002544f0  0.10037834f0  0.10008727f0  0.09963008f0  0.09913887f0  0.09841014f0  0.09757022f0  0.097053036f0  0.09637562f0  0.09565735f0  0.095111646f0  0.094732806f0  0.094349116f0  0.09395435f0  0.09332651f0  0.092937216f0  0.09234102f0  0.09174077f0  0.09140556f0  0.09100329f0  0.09070647f0  0.090373926f0  0.089807555f0  0.0892684f0  0.088758044f0  0.088092156f0  0.08754931f0  0.08709745f0  0.08635649f0  0.08592948f0  0.085242495f0  0.08440173f0  0.08355209f0  0.082693554f0  0.08194266f0  0.08109036f0  0.080326766f0  0.07969277f0  0.078893274f0  0.07838428f0  0.07772963f0  0.0768437f0  0.07595244f0  0.07521114f0  0.07473288f0  0.074167505f0  0.07363919f0  0.07311703f0  0.07250872f0  0.07195241f0  0.07135252f0  0.07075493f0  0.070128426f0  0.06947734f0  0.06882012f0  0.068116896f0  0.06744407f0  0.06716777f0  0.066928f0  0.06651497f0  0.06632418f0  0.06588342f0  0.06556073f0  0.06517124f0  0.06476645f0  0.06460558f0  0.06444093f0  0.06417997f0  0.06379363f0  0.06353915f0  0.063248664f0  0.06298102f0  0.06301749f0  0.063048616f0  0.063116f0  0.062926255f0  0.06268464f0  0.06261793f0  0.062291525f0  0.061961893f0  0.062064126f0  0.062196657f0  0.06193442f0  0.061669637f0  0.061659135f0  0.06131433f0  0.061435692f0  0.06139964f0  0.061204538f0  0.060739603f0  0.060336545f0  0.05986944f0  0.059490748f0  0.059235707f0  0.058877528f0  0.05863158f0  0.058462813f0  0.058296878f0  0.05813089f0  0.05796326f0  0.05764094f0  0.05706031f0  0.056319863f0  0.1169692f0  0.116220884f0  0.11601004f0  0.11578208f0  0.114849165f0  0.11410105f0  0.11394291f0  0.11369717f0  0.11281732f0  0.11208938f0  0.11148715f0  0.110778876f0  0.110144064f0  0.10938664f0  0.10827135f0  0.10737911f0  0.1072672f0  0.10607713f0  0.10463967f0  0.10387016f0  0.10312705f0  0.102296606f0  0.10154359f0  0.10097581f0  0.09976559f0  0.099425055f0  0.09850118f0  0.0979282f0  0.09769513f0  0.09689481f0  0.09649472f0  0.09626544f0  0.09669418f0  0.09687015f0  0.09709785f0  0.09750293f0  0.097350076f0  0.09794079f0  0.098207f0  0.09831921f0  0.09788978f0  0.09799801f0  0.097883396f0  0.09796976f0  0.09862889f0  0.09913871f0  0.09949738f0  0.10023394f0  0.100971654f0  0.10128344f0  0.1016476f0  0.101988584f0  0.102343224f0  0.10282476f0  0.103299975f0  0.103918284f0  0.10398351f0  0.10445497f0  0.10447194f0  0.10455419f0  0.104242876f0  0.10370686f0  0.103965074f0  0.10351902f0  0.10372243f0  0.103421465f0  0.1032229f0  0.10315889f0  0.10275577f0  0.10276003f0  0.10223566f0  0.10221851f0  0.10171599f0  0.101082f0  0.101050526f0  0.100871556f0  0.10093522f0  0.10030929f0  0.100239806f0  0.09987728f0  0.0996188f0  0.099147215f0  0.099152446f0  0.09890103f0  0.09891501f0  0.09885647f0  0.09856187f0  0.09840585f0  0.09811983f0  0.09801915f0  0.0979455f0  0.09780157f0  0.0976716f0  0.09746462f0  0.097008675f0  0.09653581f0  0.096336365f0  0.096070096f0  0.095749125f0  0.09519324f0  0.09460683f0  0.09399987f0  0.093734905f0  0.093159996f0  0.092713505f0  0.09232889f0  0.09187438f0  0.09152457f0  0.09104228f0  0.09043237f0  0.08979321f0]

ter = [false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  true  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false]

generalized_advantage_estimation(
        rew,
        val,
        y,
        p;
        dims=2,
        terminal=ter
    )