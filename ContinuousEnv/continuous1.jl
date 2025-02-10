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
using Distributions
using UnicodePlots
using CircularArrayBuffers
#using Blink

n_turbines = 1


scriptname = "continuous1"




#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
end



# Global constant for the probability to spawn a new job in an empty slot.
JOB_SPAWN_PROB = 0.1


# Number of job slots available (e.g. 4)
n_jobs = 1


# Maximum compute slots per data center (e.g. 5)
max_slots = 5


# action vector dim - contains a flattened delta_allocation matrix od size n_turbines * n_jobs
action_dim = n_turbines * n_jobs


# wind model variables
wind_model_vars = rand(n_turbines, 5) # 5 variables between 0 and 1 per turbine
wind = Float64[]


# grid price model variables
grid_model_vars = rand(5) # 5 variables between 0 and 1
grid_price = 0.0


# Curtailment threshold model variables
curtailment_threshold = 0.4


# A job has an integer compute load, a deadline, and an associated penalty.
mutable struct Job
    id::Int               # A unique identifier
    load::Int             # Total compute units required
    remaining::Int        # Work still to be done
    arrival_time::Float64 # When the job becomes available
    deadline::Float64     # Latest time by which the job must be completed
    penalty::Float64      # If finished in time, add this as positive reward; if not, subtract it
end

# Job slots: each slot may or may not hold a job
job_slots::Vector{Union{Job, Nothing}} = [nothing for i in 1:n_jobs]


# Allocation matrix (n_turbines × n_jobs): allocation[i, j] gives the number
# of compute slots in data center i that are allocated to job j.
allocation::Matrix{Int} = zeros(Int, n_turbines, n_jobs)


# state vector
# - time of day
# - current gradient and price of energy from the grid
# - current curtailment_threshold
# - wind stituation at every turbine (gradient of power output, current power output and curtailment enegry)
# - for each job slot: remaining, time until deadline, and penalty
# - the flattened allocation matrix of the size n_turbines * n_jobs
state_dim = 4 + 3*n_turbines + 3*n_jobs + n_turbines * n_jobs



# env parameters

seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false

te = Inf
dt = 5/1440 # 5-minute steps as fraction of a day
t0 = 0.0
min_best_episode = 1

sim_space = Space(fill(0..1, (state_dim)))




######################################
# Helper functions: Signal generators#
######################################

# Generate a wind signal for each turbine. For now, we simply return random values in [0,1]
function generate_wind()
    global wind_model_vars
    temp_wind = Float64[]

    if isnothing(env)
        time = 0.0
    else
        time = env.time
    end

    for i in 1:n_turbines

        t_mod = mod(time, 2π)

        # Base wind
        base_wind = 0.5 + 0.5 * sin(t_mod + wind_model_vars[i,1] * 4π)

        # Add three additional sines based on the model vars 2,3 and 4
        base_wind += (wind_model_vars[i,2] * 0.3) * sin(4.5 * time + 1.0)
        base_wind += (wind_model_vars[i,3] * 0.2) * sin(6.2 * time + 1.2)
        base_wind += (wind_model_vars[i,4] * 0.2) * sin(8.3 * time + 1.7)

        # Add the additive grid momentum which is the last model var
        wind_val = base_wind + 0.8 * (wind_model_vars[5] - 0.3)

        # scale down wind_val
        wind_val = 0.5 + (wind_val-0.3)*0.7
        
        # Update wind model wvars in a momentum fashion where the change is determined by sines
        for j in eachindex(wind_model_vars[i,:])
            wind_model_vars[i,j] = clamp(0.8 * wind_model_vars[i,j] + 0.2 * sin(j*time), 0.0, 1.0)
        end
        
        
        # Clamp the resulting wind value to [0, 1].
        wind_val = clamp(wind_val, 0.0, 1.0)

        # Flip it (it was too in sync with grid price)
        wind_val = -1 * wind_val + 1.0

        push!(temp_wind, wind_val)
    end
    
    return temp_wind
end

#test plot
# test_wind = Float64[]
# for step in 1:(1440/5)*1
#     push!(test_wind, generate_wind()[1])
#     env.time += dt
# end
# plot(test_wind, Layout(yaxis_range=[0, 1]))




# Generate a grid price signal in [0,1]
env = nothing
function generate_grid_price()
    global grid_model_vars
    # t_day is the current time of day, in [0, 1). For example, env.time=0.25 represents 6:00 AM if 1.0 = 24 hours.
    if isnothing(env)
        time = 0.0
    else
        time = env.time
    end

    t_day = mod(time, 1.0)
    
    # Base grid price: using a cosine so that:
    # - At midnight (t_day = 0), cos(0)=1 and the price is 1.
    # - At noon (t_day = 0.5), cos(π)=-1 and the price is 0.
    base_price = 0.5 + 0.5 * cos(2π * t_day)

    # Scale the grid price according to the first two model vars
    scaled_base_price = (grid_model_vars[1] - 0.5) * 0.5 + ((grid_model_vars[2]*0.5) + 0.8) * base_price

    # Add two small additional sines based on the second two model vars
    scaled_base_price += (grid_model_vars[3] * 0.3) * sin(9.3 * time)
    scaled_base_price += (grid_model_vars[4] * 0.2) * sin(14.3 * time)

    # Add the additive grid momentum which is the last model var
    price_val = scaled_base_price + 0.25 * (grid_model_vars[5] - 0.5)
    
    # Update grid model wvars in a momentum fashion where the change is determined by sines
    for i in eachindex(grid_model_vars)
        grid_model_vars[i] = clamp(0.8 * grid_model_vars[i] + 0.2 * sin(i*time), 0.0, 1.0)
    end

    # scale down price_val
    price_val = 0.5 + (price_val-0.3)*0.7
    
    # Clamp the grid price to [0, 1].
    return clamp(price_val, 0.0, 1.0)
end

#test plot
# test_price = Float64[]
# for step in 1:(1440/5)*1
#     push!(test_price, generate_grid_price()[1])
#     env.time += dt
# end
# plot(test_price, Layout(yaxis_range=[0, 1]))


# Generate a curtailment threshold signal
function generate_curtailment_threshold()
    return 0.4
end

# Generate a new job given the current time.
id_counter = 1
function generate_job(current_time::Float64)::Job
    global id_counter
    new_id = id_counter

    id_counter += 1

    load = Int(max(floor(rand(Normal(50,10))),0.0))                 # Compute load in units (integer)
    window = rand(0.3:3.0)           # The allowed time window
    deadline = current_time + window
    penalty = max(rand(Normal(6,4)),0.0)+0.5         # Penalty that can become negative oder positive reward

    load = 30
    window = rand() * 20.5
    deadline = current_time + window
    penalty = 2.2

    return Job(new_id, load, load, current_time, deadline, penalty)
end






# y0 calculation

y0 = [0.0]

wind = generate_wind()

grid_price = generate_grid_price()

curtailment_threshold = generate_curtailment_threshold()


push!(y0, grid_price - grid_price) # start with derivative 0
push!(y0, grid_price)

push!(y0, curtailment_threshold)

for i in 1:n_turbines
    push!(y0, wind[i] - wind[i]) # start with derivative 0
    push!(y0, wind[i])
    push!(y0, max(0.0, wind[i] - curtailment_threshold))
end

for i in 1:n_jobs
    push!(y0, 0.0f0) # remaining units
    push!(y0, 0.0f0) # time left till deadline
    push!(y0, 0.0f0) # penalty
end

for i in 1:n_jobs*n_turbines
    push!(y0, 0.0f0)
end

y0 = Float32.(y0)







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
y = 0.997f0
p = 0.95f0

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 100


learning_rate = 1e-5
n_epochs = 3
n_microbatches = 4
logσ_is_network = false
max_σ = 10000.0f0
actor_loss_weight = 100.0
critic_loss_weight = 0.01
entropy_loss_weight = 0.1
clip_grad = 0.3
target_kl = 0.1
clip1 = false
start_logσ = 0.0
tanh_end = true
clip_range = 0.05f0



wind_only = false




function softplus_shifted(x)
    factor = 700
    log( 1 + exp(factor * (x - 0.006)) ) / factor
end


function do_step(env)
    global wind, grid_price, curtailment_threshold, job_slots, allocation, n_turbines, n_jobs


    # -- 1. Update the allocation based on the agent's action --
    # For each data center (turbine) and job slot:
    action = env.p

    for i in 1:n_turbines
        for j in 1:n_jobs
            # Only update allocation if there is an active job in this slot.
            if job_slots[j] !== nothing
                # allocation[i, j] = max(allocation[i, j] + action[i, j], 0)
                allocation[i, j] = action[i, j]
            else
                # If there is no job, make sure allocation remains 0.
                allocation[i, j] = 0
            end
        end

        # Enforce the maximum compute slots constraint.
        total_alloc = sum(allocation[i, :])
        if total_alloc > max_slots
            # Scale down each allocation proportionally.
            scale = max_slots / total_alloc
            scaled_alloc = [allocation[i, j] * scale for j in 1:n_jobs]
            
            # Use rounding to get integer allocations.
            new_alloc = [round(Int, x) for x in scaled_alloc]
            
            # After rounding, we may end up with a total allocation that is too high or too low.
            # First, if the sum is too high, iteratively reduce the allocation from the job slot with the highest value.
            while sum(new_alloc) > max_slots
                idx = argmax(new_alloc)  # index of the largest allocation
                new_alloc[idx] -= 1
            end
            # Conversely, if the sum is less than max_slots, add extra slots to the job slot with the smallest allocation.
            while sum(new_alloc) < max_slots
                idx = argmin(new_alloc)
                new_alloc[idx] += 1
            end
            
            # Finally, update the allocation for data center i.
            for j in 1:n_jobs
                allocation[i, j] = new_alloc[j]
            end
        end
    end




    # -- 2. Process compute slots and update job progress & cost --
    total_cost = 0.0
    job_reward = 0.0  # bonus rewards from completed jobs or penalties for missed deadlines
    for i in 1:n_turbines
        # Sum the allocated slots for this turbine.
        allocated = sum(allocation[i, :])
        
        # If no job is allocated, assume one idle slot is used.
        if allocated == 0
            allocated = 1  # idle consumption (without any job progress)
        end

        # Convert allocated slots into a proportion of max_slots.
        # For example, if allocated=4 and max_slots=5, effective_alloc becomes 0.8.
        effective_alloc = allocated / max_slots

        # Determine free (wind) energy available (for simplicity, using a threshold rule).
        free_energy = max(0.0, wind[i] - curtailment_threshold)
        
        # Calculate the grid energy needed as the shortfall (if any).
        grid_energy = max(0.0, effective_alloc - free_energy)
        cost = grid_energy * grid_price
        total_cost += cost

        # For each job slot in this turbine, update the job's progress.
        for j in 1:n_jobs
            slots_for_job = allocation[i, j]
            if slots_for_job > 0 && (job_slots[j] !== nothing)
                job = job_slots[j]
                # Each compute slot reduces the remaining load by one unit.
                job.remaining -= slots_for_job
                if job.remaining <= 0
                    # If a job finishes, add its penalty as a positive reward bonus.
                    job_reward += job.penalty
                    # Remove the job and clear its allocation from all data centers.
                    job_slots[j] = nothing
                    for k in 1:n_turbines
                        allocation[k, j] = 0
                    end
                end
            end
        end
    end





    # -- 3. Update time and external signals --
    wind = generate_wind()
    grid_price = generate_grid_price()
    curtailment_threshold = generate_curtailment_threshold()





    # Check deadlines for jobs and apply penalties if necessary.
    for j in 1:n_jobs
        if job_slots[j] !== nothing
            job = job_slots[j]
            if env.time > job.deadline && job.remaining > 0
                job_reward -= job.penalty
                job_slots[j] = nothing
                for i in 1:n_turbines
                    allocation[i, j] = 0
                end
            end
        end
    end

    # Possibly generate new jobs in empty job slots.
    for j in 1:n_jobs
        if job_slots[j] === nothing && rand() < JOB_SPAWN_PROB
            job_slots[j] = generate_job(env.time)
        end
    end

    # Total reward is the negative cost plus any job-based rewards.
    step_reward = -total_cost + job_reward
    env. reward = [step_reward]




    # Build and return the new state.
    y = [mod(env.time, 1.0)]
    
    push!(y, grid_price - env.y[2])
    push!(y, grid_price)

    push!(y, curtailment_threshold)

    for i in 1:n_turbines
        push!(y, wind[i] - env.y[3*i+1]) # start with derivative 0
        push!(y, wind[i])
        push!(y, max(0.0, wind[i] - curtailment_threshold))
    end

    for job in job_slots
        if job !== nothing
            push!(y, Float32(job.remaining))
            push!(y, Float32(max(0.0, job.deadline - env.time)))  # time left (could be normalized)
            push!(y, Float32(job.penalty))
        else
            # Use zero values if no job is scheduled in this slot.
            push!(y, 0.0f0)
            push!(y, 0.0f0)
            push!(y, 0.0f0)
        end
    end

    # Flatten the allocation matrix into the state.
    for i in 1:size(allocation, 1)
        for j in 1:size(allocation, 2)
            push!(y, Float32(allocation[i, j]))
        end
    end

    return Float32.(y)
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

    clamp!(action, 0.0, 1.0)

    # enhance a bit
    action = action .* 5.0

    # convert Float32 array action to Int delta_allocation matrix
    action = Int.(floor.(reshape(action, n_turbines, n_jobs)))

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
                actor_loss_weight = actor_loss_weight,
                critic_loss_weight = critic_loss_weight,
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
                            early_success_possible = true,
                            collect_bestDF = false)
end

function generate_random_init()
    global wind, grid_price, curtailment_threshold, job_slots, allocation, wind_model_vars, grid_model_vars


    job_slots = [nothing for i in 1:n_jobs]
    allocation = zeros(Int, n_turbines, n_jobs)


    # Here the model variables can be modified before the generators are called
    
    y0 = [0.0]


    wind_model_vars = rand(n_turbines, 5)
    grid_model_vars = rand(5)

    wind = generate_wind()

    grid_price = generate_grid_price()

    curtailment_threshold = generate_curtailment_threshold()


    push!(y0, grid_price - grid_price) # start with derivative 0
    push!(y0, grid_price)

    push!(y0, curtailment_threshold)

    for i in 1:n_turbines
        push!(y0, wind[i] - wind[i]) # start with derivative 0
        push!(y0, wind[i])
        push!(y0, max(0.0, wind[i] - curtailment_threshold))
    end

    for i in 1:n_jobs
        push!(y0, 0.0f0)
        push!(y0, 0.0f0)
        push!(y0, 0.0f0)
    end

    for i in 1:n_jobs*n_turbines
        push!(y0, 0.0f0)
    end

    y0 = Float32.(y0)

    env.y0 = deepcopy(y0)
    env.y = deepcopy(y0)
    env.state = env.featurize(; env = env)

    y0
end

initialize_setup()

train_rewards = Float64[]
temp_reward_queue::CircularArrayBuffer{Float64} = CircularArrayBuffer{Float64}(1)


function train(use_random_init = true; num_steps = 500_000, smoothing_window = 400, collect_every = 100, plot_every = 5000)
    frame = 1

    global train_rewards
    global temp_reward_queue

    temp_reward_queue = CircularArrayBuffer{Float64}(smoothing_window)


    if use_random_init
        hook.generate_random_init = generate_random_init
    else
        hook.generate_random_init = false
    end
    


    stop_condition = StopAfterStep(num_steps)

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

            frame += 1

            push!(temp_reward_queue, env.reward[1])

            if frame > smoothing_window && frame%collect_every == 0
                push!(train_rewards, mean(temp_reward_queue))
            end

            if frame%plot_every == 0
                plt = lineplot(train_rewards, title="Current smoothed rewards", xlabel="Steps", ylabel="Score")
                println(plt)
            end

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

    # hook.rewards = clamp.(hook.rewards, -3000, 0)


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



function render_run(steps = 864, make_deterministic = true)
    # if use_best
    #     copyto!(agent.policy.behavior_actor, hook.bestNNA)
    # end

    # temp_noise = agent.policy.act_noise
    # agent.policy.act_noise = 0.0

    # temp_start_steps = agent.policy.start_steps
    # agent.policy.start_steps  = -1
    
    # temp_update_after = agent.policy.update_after
    # agent.policy.update_after = 100000

    if !(agent.policy.approximator.actor.logσ_is_network) && make_deterministic
        temp_logσ = deepcopy(agent.policy.approximator.actor.logσ)
        agent.policy.approximator.actor.logσ[:,:] = -7 .* ones(n_jobs*n_turbines,1)
    end

    global rewards = Float64[]
    reward_sum = 0.0

    #w = Window()

    global results = Dict("rewards" => [], "grid_price" => [])

    for k in 1:n_turbines
        results["wind$(k)"] = []
    end

    for k in 1:n_jobs
        results["job$(k)_remaining"] = []
        results["job$(k)_penalty"] = []
        results["job$(k)_time_left"] = []
    end

    global currentDF = DataFrame()

    reset!(env)
    generate_random_init()

    # set random time
    env.time = rand()*10000

    for i in 1:steps
        action = agent(env)

        #action = env.y[6] < 0.27 ? [-1.0] : [1.0]

        env(action)

        for k in 1:n_turbines
            push!(results["wind$(k)"], env.y[3+k*3])
        end

        for k in 1:n_jobs
            push!(results["job$(k)_remaining"], env.y[4 + n_turbines * 3 + 1 + (k-1) * 3])
            push!(results["job$(k)_time_left"], env.y[4 + n_turbines * 3 + 2 + (k-1) * 3])
            push!(results["job$(k)_penalty"], env.y[4 + n_turbines * 3 + 3 + (k-1) * 3])
        end

        push!(results["rewards"], env.reward[1])
        push!(results["grid_price"], env.y[3])

        # println(mean(env.reward))

        reward_sum += mean(env.reward)
        # push!(rewards, mean(env.reward))


    end

    if !(agent.policy.approximator.actor.logσ_is_network) && make_deterministic
        agent.policy.approximator.actor.logσ[:,:] = temp_logσ
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

    

    p = make_subplots(rows=2+n_jobs, cols=1)

    # global to_plot = [scatter(y=results["rewards"], name="reward", yaxis = "y2"),
    #             scatter(y=results["grid_price"], name="grid price")]

    add_trace!(p, scatter(y=results["rewards"], name="reward", yaxis = "y2"), row = 1)
    add_trace!(p, scatter(y=results["grid_price"], name="grid price"), row = 2, col = 1)

    for k in 1:n_turbines
        #push!(to_plot, scatter(y=results["wind$(k)"], name="wind$(k)"))
        add_trace!(p, scatter(y=results["wind$(k)"], name="wind$(k)"), row = 2, col = 1)
    end
    
    for k in 1:n_jobs
        add_trace!(p, scatter(y=results["job$(k)_remaining"], name="job$(k) remaining"), row = 2+k, col = 1)
    end


    #relayout!(p, layout.fields)
    display(p)

end

