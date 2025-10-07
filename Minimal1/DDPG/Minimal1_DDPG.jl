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


scriptname = "Minimal1_DDPG"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
end


include("./../Minimal1_env.jl")


seed = Int(floor(rand()*1000))
# seed = 578

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
gamma = y
p = 0.995f0
batch_size = 256
start_steps = -1
start_policy = ZeroPolicy(actionspace)
update_after = 200_000
update_freq = 10
update_loops = 3
reset_stage = POST_EPISODE_STAGE
learning_rate = 5e-5
learning_rate_critic = 1e-4
clip_grad = 0.5
act_limit = 1.0
act_noise = 0.05
trajectory_length = 1_000_000

reward_shaping = false










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


    global agent = create_agent(mono = true,
                        action_space = actionspace,
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
                        learning_rate_critic = learning_rate_critic,
                        clip_grad = clip_grad,)

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
optimal_trajectory = trajectories["SAC_DDPG"]["with_RS"]




function render_run(use_best = false; exploration = true)
    if use_best
        copyto!(agent.policy.behavior_actor, hook.bestNNA)
    end


    temp_start_steps = agent.policy.start_steps
    agent.policy.start_steps  = -1
    
    temp_update_after = agent.policy.update_after
    agent.policy.update_after = 100000

    agent.policy.update_step = 0
    global rewards = Float64[]
    reward_sum = 0.0

    #w = Window()

    results_run = Dict("rewards" => [], "loadleft" => [])

    for k in 1:n_turbines
        results_run["hpc$k"] = []
    end

    global currentDF = DataFrame()

    reset!(env)
    generate_random_init()

    while !env.done
        if exploration
            action = agent(env)
        else
            action = agent.policy.behavior_actor(env)
        end

        #action = env.y[6] < 0.27 ? [-1.0] : [1.0]

        env(action)

        for k in 1:n_turbines
            push!(results_run["hpc$k"], env.p[k])
        end
        push!(results_run["rewards"], env.reward[1])
        push!(results_run["loadleft"], env.y[1])

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

    to_plot = [scatter(y=results_run["rewards"], name="reward", yaxis = "y2"),
                scatter(y=results_run["loadleft"], name="load left"),
                scatter(y=grid_price, name="grid price")]
    for k in 1:n_turbines
        push!(to_plot, scatter(y=results_run["hpc$k"], name="hpc$k"))
        push!(to_plot, scatter(y=wind[k], name="wind$k"))
    end
    p = plot(Vector{AbstractTrace}(to_plot), layout)

    display(p)

end