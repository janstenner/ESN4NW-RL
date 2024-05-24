using LinearAlgebra
using ReinforcementLearning
using IntervalSets
using StableRNGs
using ReinforcementLearning:update!
using SparseArrays
using Conda
using FFTW
using PlotlyJS
using FileIO, JLD2
#using Blink

n_turbines = 1


scriptname = "Minimal1"

include(pwd() * "/src/PDEagent.jl")
include(pwd() * "/src/PDEenv.jl")
include(pwd() * "/src/PDEhook.jl")
include(pwd() * "/src/StopCondition.jl")

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

gpu_env = false

te = 1440.0
dt = 5.0
t0 = 0.0
min_best_episode = 1

sim_space = Space(fill(0..1, (state_dim)))

function generate_wind()
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

y0 = zeros(state_dim)
y0[1] = 1.0

wind_constant_day = rand()

wind = [generate_wind() for i in 1:n_turbines]

# layout = Layout(
#                 plot_bgcolor="#f1f3f7",
#                 yaxis=attr(range=[0,1]),
#             )

# to_plot = [scatter(y=wind[i]) for i in 1:3]
# plot(Vector{AbstractTrace}(to_plot), layout)

grid_price = (-sin.(collect(LinRange(rand()*1.5, 2+rand()*1.5, Int(te/dt)+1))) .+(1+rand()))
clamp!(grid_price, -1, 1)
# plot(grid_price)

for i in 1:n_turbines
    y0[((i-1)*2)+2] = wind[i][2] - wind[i][1]
    y0[((i-1)*2)+3] = wind[i][2] 
end

y0[1 + n_turbines * 2 + 1] = grid_price[2] - grid_price[1]
y0[1 + n_turbines * 2 + 2] = grid_price[2]


# agent tuning parameters
memory_size = 0
nna_scale = 2.3
nna_scale_critic = 2.3
drop_middle_layer = false
drop_middle_layer_critic = false
fun = leakyrelu
use_gpu = false
action_space = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.95f0
p = 0.995f0
batch_size = 10
start_steps = -1
start_policy = ZeroPolicy(action_space)
update_after = 10
update_freq = 1
update_loops = 10
reset_stage = POST_EPISODE_STAGE
learning_rate = 0.0002
learning_rate_critic = 0.0005
act_limit = 1.0
act_noise = 1.2
trajectory_length = 500_000



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

    reward1 = sqrt(50 * compute_power_used) * ((grid_price[step-1] + 0.2)^2) * 0.5

    reward2 = - (37 * compute_power_used^1.2) * (1-grid_price[step-1]*3)

    factor = clamp(grid_price[step-1] * 2 - 0.5, 0.0, 1.0)
    reward = - (factor * reward1 + (1 - factor) * reward2) + (power_for_free_used * 40)^1.2


    if (env.time + env.dt) >= env.te 
        reward -= y[1] * 1000
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


# PDEenv can also take a custom y0 as a parameter. Example: PDEenv(y0=y0_sawtooth, ...)
function initialize_setup(;use_random_init = false)

    global env = PDEenv(do_step = do_step, 
                reward_function = reward_function,
                featurize = featurize,
                prepare_action = prepare_action,
                y0 = y0,
                te = te, t0 = t0, dt = dt, 
                sim_space = sim_space, 
                action_space = action_space,
                oversampling = 1,
                use_radau = false,
                max_value = 1.0,
                check_max_value = "nothing")

    global agent = create_agent(mono = true,
                        action_space = action_space,
                        state_space = env.state_space,
                        use_gpu = use_gpu, 
                        rng = rng,
                        y = y, p = p, batch_size = batch_size, 
                        start_steps = start_steps, 
                        start_policy = start_policy,
                        update_after = update_after, 
                        update_freq = update_freq,
                        update_loops = update_loops,
                        reset_stage = reset_stage,
                        act_limit = act_limit, 
                        act_noise = act_noise,
                        nna_scale = nna_scale,
                        nna_scale_critic = nna_scale_critic,
                        drop_middle_layer = drop_middle_layer,
                        drop_middle_layer_critic = drop_middle_layer_critic,
                        fun = fun,
                        memory_size = memory_size,
                        trajectory_length = trajectory_length,
                        learning_rate = learning_rate,
                        learning_rate_critic = learning_rate_critic)

    global hook = PDEhook(min_best_episode = min_best_episode, use_random_init = use_random_init)
end

function generate_random_init()
    global wind_constant_day, wind, grid_price

    y0 = zeros(state_dim)
    y0[1] = 1.0

    wind_constant_day = rand()

    wind = [generate_wind() for i in 1:n_turbines]

    grid_price = (-sin.(collect(LinRange(rand()*1.5, 2+rand()*1.5, Int(te/dt)+1))) .+(1+rand()))
    clamp!(grid_price, -1, 1)

    for i in 1:n_turbines
        y0[((i-1)*2)+2] = wind[i][2] - wind[i][1]
        y0[((i-1)*2)+3] = wind[i][2] 
    end
    
    y0[1 + n_turbines * 2 + 1] = grid_price[2] - grid_price[1]
    y0[1 + n_turbines * 2 + 2] = grid_price[2]

    env.y0 = y0
    env.y = env.y0
    env.state = env.featurize(; env = env)

    y0
end

initialize_setup()

# plotrun(use_best = false, plot3D = true)

function train(use_random_init = true; visuals = false, num_steps = 4000, inner_loops = 5)
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

    hook.use_random_init = use_random_init
    

    
    outer_loops = 40

    for i = 1:outer_loops
        agent.policy.act_noise = act_noise
        for i = 1:inner_loops
            println("")
            println(agent.policy.act_noise)
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
            agent.policy.act_noise = agent.policy.act_noise * 0.4

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



function render_run(use_best = false)
    if use_best
        copyto!(agent.policy.behavior_actor, hook.bestNNA)
    end

    temp_noise = agent.policy.act_noise
    agent.policy.act_noise = 0.0

    temp_start_steps = agent.policy.start_steps
    agent.policy.start_steps  = -1
    
    temp_update_after = agent.policy.update_after
    agent.policy.update_after = 100000

    agent.policy.update_step = 0
    global rewards = Float64[]
    reward_sum = 0.0

    #w = Window()

    results = Dict("rewards" => [], "loadleft" => [])

    for k in 1:n_turbines
        results["hpc$k"] = []
    end

    global currentDF = DataFrame()

    RLBase.reset!(env)
    generate_random_init()

    while !env.done
        action = agent(env)

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

    if use_best
        copyto!(agent.policy.behavior_actor, hook.currentNNA)
    end

    agent.policy.start_steps = temp_start_steps
    agent.policy.act_noise = temp_noise
    agent.policy.update_after = temp_update_after

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
    plot(Vector{AbstractTrace}(to_plot), layout)

end

# t1 = scatter(y=rewards1)
# t2 = scatter(y=rewards2)
# t3 = scatter(y=rewards3)
# plot([t1, t2, t3])