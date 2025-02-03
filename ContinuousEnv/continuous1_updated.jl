using Random

##############################
# Data Structures Definition #
##############################

# A job has an integer compute load, a deadline, and an associated penalty.
struct Job
    id::Int               # A unique identifier (could be generated randomly)
    load::Int             # Total compute units required
    remaining::Int        # Work still to be done
    arrival_time::Float64 # When the job becomes available
    deadline::Float64     # Latest time by which the job must be completed
    penalty::Float64      # If finished in time, add this as positive reward; if not, subtract it
end

# The continuous environment structure
mutable struct ContinuousEnv
    time::Float64               # Current simulation time
    dt::Float64                 # Time step
    n_turbines::Int             # Number of data centers (wind turbines)
    n_jobs::Int                 # Number of job slots available (e.g. 4)
    max_slots::Int              # Maximum compute slots per data center (e.g. 5)
    threshold::Float64          # Wind threshold for free (green) energy (e.g. 0.4)
    
    wind::Vector{Float64}       # Current wind signal per turbine (length n_turbines)
    grid_price::Float64         # Current grid price (assumed global here)
    
    # Job slots: each slot may or may not hold a job
    job_slots::Vector{Union{Job, Nothing}}  # length = n_jobs
    
    # Allocation matrix (n_turbines × n_jobs): allocation[i, j] gives the number
    # of compute slots in data center i that are allocated to job j.
    allocation::Matrix{Int}
end

#############################################
# Helper functions: Random signal generators#
#############################################

# Generate a wind signal for each turbine. For now, we simply return random values in [0,1]
function generate_wind(n_turbines::Int)
    return [rand() for i in 1:n_turbines]
end

# Generate a grid price signal in [0,1]
function generate_grid_price()
    return rand()
end

# Generate a new job given the current time.
# You can adjust the ranges as desired.
function generate_job(current_time::Float64)::Job
    new_id = rand(1:10^6)            # For example, a random unique id
    load = rand(1:5)                 # Compute load in units (integer)
    window = rand(10.0:50.0)           # The allowed time window (in the same time units as dt)
    deadline = current_time + window
    penalty = load * 2.0             # For example: penalty equals two times the load
    return Job(new_id, load, load, current_time, deadline, penalty)
end

#############################################
# Environment Initialization and State      #
#############################################

# Create and return a new continuous environment.
function reset_env(; dt=5.0, n_turbines=1, n_jobs=4, max_slots=5, threshold=0.4)
    env = ContinuousEnv(
        0.0,                   # time starts at zero
        dt,
        n_turbines,
        n_jobs,
        max_slots,
        threshold,
        generate_wind(n_turbines),  # initial wind for each turbine
        generate_grid_price(),      # initial grid price
        [nothing for i in 1:n_jobs], # no job scheduled initially
        zeros(Int, n_turbines, n_jobs)  # no compute slots allocated yet
    )
    return env
end

# Build the state vector (here as a flat vector of Float32).
function get_state(env::ContinuousEnv)
    state = Float32[]
    
    # Include (possibly normalized) time:
    push!(state, Float32(env.time))
    
    # Add wind readings (per turbine):
    for w in env.wind
        push!(state, Float32(w))
    end
    
    # Add grid price:
    push!(state, Float32(env.grid_price))
    
    # For each job slot, add job information:
    # Here we include: remaining, time until deadline, and penalty.
    for job in env.job_slots
        if job !== nothing
            push!(state, Float32(job.remaining))
            push!(state, Float32(max(0.0, job.deadline - env.time)))  # time left (could be normalized)
            push!(state, Float32(job.penalty))
        else
            # Use zero values if no job is scheduled in this slot.
            push!(state, 0.0f0)
            push!(state, 0.0f0)
            push!(state, 0.0f0)
        end
    end
    
    # Flatten the allocation matrix into the state.
    for i in 1:size(env.allocation, 1)
        for j in 1:size(env.allocation, 2)
            push!(state, Float32(env.allocation[i, j]))
        end
    end
    
    return state
end

#################################################
# The Step Function – Core Environment Update   #
#################################################

# Global constant for the probability to spawn a new job in an empty slot.
const JOB_SPAWN_PROB = 0.1

"""
    step!(env, action)

Takes an action (a matrix of integers of shape (n_turbines, n_jobs)) which represents the delta
(change) in compute slot allocation for each data center and job slot. Updates the environment:
  - Adjusts allocations (enforcing that each data center uses at most max_slots)
  - Uses the allocated compute slots to advance any assigned jobs by one unit per slot.
  - Computes the cost based on grid power consumption after free energy is used.
  - Updates job statuses (rewarding completion or penalizing missed deadlines).
  - Advances time and updates the wind and grid price signals.
Returns the new state vector and the reward for the step.
"""
function step!(env::ContinuousEnv, action::Matrix{Int})
    # -- 1. Update the allocation based on the agent's action --
    # For each data center (turbine) and job slot:
    @assert size(action) == size(env.allocation)
    for i in 1:env.n_turbines
        for j in 1:env.n_jobs
            # TODO 1:
            # Only update allocation if there is an active job in this slot.
            if env.job_slots[j] !== nothing
                env.allocation[i, j] = max(env.allocation[i, j] + action[i, j], 0)
            else
                # If there is no job, make sure allocation remains 0.
                env.allocation[i, j] = 0
            end
        end

        # Enforce the maximum compute slots constraint.
        total_alloc = sum(env.allocation[i, :])
        if total_alloc > env.max_slots
            # Scale down each allocation proportionally.
            scale = env.max_slots / total_alloc
            scaled_alloc = [env.allocation[i, j] * scale for j in 1:env.n_jobs]
            
            # Use rounding to get integer allocations.
            new_alloc = [round(Int, x) for x in scaled_alloc]
            
            # TODO 2:
            # After rounding, we may end up with a total allocation that is too high or too low.
            # First, if the sum is too high, iteratively reduce the allocation from the job slot with the highest value.
            while sum(new_alloc) > env.max_slots
                idx = argmax(new_alloc)  # index of the largest allocation
                new_alloc[idx] -= 1
            end
            # Conversely, if the sum is less than max_slots, add extra slots to the job slot with the smallest allocation.
            while sum(new_alloc) < env.max_slots
                idx = argmin(new_alloc)
                new_alloc[idx] += 1
            end
            
            # Finally, update the allocation for data center i.
            for j in 1:env.n_jobs
                env.allocation[i, j] = new_alloc[j]
            end
        end
    end

    # -- 2. Process compute slots and update job progress & cost --
    total_cost = 0.0
    job_reward = 0.0  # bonus rewards from completed jobs or penalties for missed deadlines
    for i in 1:env.n_turbines
        # Sum the allocated slots for this turbine.
        allocated = sum(env.allocation[i, :])
        
        # If no job is allocated, assume one idle slot is used.
        if allocated == 0
            allocated = 1  # idle consumption (without any job progress)
        end

        # TODO 3:
        # Convert allocated slots into a proportion of max_slots.
        # For example, if allocated=4 and max_slots=5, effective_alloc becomes 0.8.
        effective_alloc = allocated / env.max_slots

        # Determine free (wind) energy available (for simplicity, using a threshold rule).
        free_energy = max(0.0, env.wind[i] - env.threshold)
        
        # Calculate the grid energy needed as the shortfall (if any).
        grid_energy = max(0.0, effective_alloc - free_energy)
        cost = grid_energy * env.grid_price
        total_cost += cost

        # For each job slot in this turbine, update the job's progress.
        for j in 1:env.n_jobs
            slots_for_job = env.allocation[i, j]
            if slots_for_job > 0 && (env.job_slots[j] !== nothing)
                job = env.job_slots[j]
                # Each compute slot reduces the remaining load by one unit.
                job.remaining -= slots_for_job
                if job.remaining <= 0
                    # If a job finishes, add its penalty as a positive reward bonus.
                    job_reward += job.penalty
                    # Remove the job and clear its allocation from all data centers.
                    env.job_slots[j] = nothing
                    for k in 1:env.n_turbines
                        env.allocation[k, j] = 0
                    end
                end
            end
        end
    end

    # -- 3. Update time and external signals --
    env.time += env.dt
    env.wind = generate_wind(env.n_turbines)
    env.grid_price = generate_grid_price()

    # Check deadlines for jobs and apply penalties if necessary.
    for j in 1:env.n_jobs
        if env.job_slots[j] !== nothing
            job = env.job_slots[j]
            if env.time > job.deadline && job.remaining > 0
                job_reward -= job.penalty
                env.job_slots[j] = nothing
                for i in 1:env.n_turbines
                    env.allocation[i, j] = 0
                end
            end
        end
    end

    # Possibly generate new jobs in empty job slots.
    for j in 1:env.n_jobs
        if env.job_slots[j] === nothing && rand() < JOB_SPAWN_PROB
            env.job_slots[j] = generate_job(env.time)
        end
    end

    # Total reward is the negative cost plus any job-based rewards.
    step_reward = -total_cost + job_reward

    # Build and return the new state.
    new_state = get_state(env)
    return new_state, step_reward
end


#############################################
# Example Usage                             #
#############################################

# Create an environment instance.
env = reset_env(dt=5.0, n_turbines=1, n_jobs=4, max_slots=5, threshold=0.4)

# (For example, an agent might output a delta matrix for compute slot reallocation.)
# Here we just create a dummy action (no changes).
dummy_action = zeros(Int, env.n_turbines, env.n_jobs)

# Run one step.
state, reward = step!(env, dummy_action)
println("New state: ", state)
println("Reward: ", reward)
