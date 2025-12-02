using Random
using IntervalSets
using RL
using FileIO, JLD2

# core settings
t0 = 0.0f0
te = 1.0f0
dt = 0.01f0
episode_steps = 100  # 0..99
delta_zone = 0.03f0
n_zones = 3
segments = 5  # 5 segments of 20 steps each
steps_per_segment = 20

min_best_episode = 1

validation_scores = []
reward_shaping = false

action_dim = 1
state_dim = 5

# per-episode run stats
zone_centers = zeros(Float32, n_zones, segments)             # 3 x 5
zone_potential_rewards = zeros(Float32, n_zones, episode_steps)  # 3 x 100

function sample_zone_centers()
    centers = zeros(Float32, n_zones, segments)
    low = delta_zone
    high = 1.0f0 - delta_zone
    min_gap = 2f0 * delta_zone

    for seg in 1:segments
        # rejection sample three sorted centers with min spacing
        tries = 0
        while true
            tries += 1
            vals = sort(rand(Float32, n_zones) .* (high - low) .+ low)
            if all(diff(vals) .>= min_gap)
                centers[:, seg] = vals
                break
            end
            if tries > 100_000
                error("Failed to sample non-overlapping zone centers after $tries attempts")
            end
        end
    end

    centers
end

function sample_zone_potential_rewards()
    biases = (0.0f0, 0.05f0, -0.05f0)
    rewards = zeros(Float32, n_zones, episode_steps)
    for i in 1:n_zones
        rewards[i, :] .= randn(Float32, episode_steps) .* 0.1f0 .+ biases[i]
    end
    rewards
end

# state helpers
function current_segment(step_idx::Int)
    clamp(div(step_idx, steps_per_segment) + 1, 1, segments)
end

function build_state(x::Float32, step_idx::Int)
    seg = current_segment(step_idx)
    t = clamp(Float32(step_idx) * dt, 0.0f0, 1.0f0)
    return Float32[x, t, zone_centers[1, seg], zone_centers[2, seg], zone_centers[3, seg]]
end

sim_space = Space(fill(0..1, (state_dim)))
y0 = build_state(0.5f0, 0)

function featurize(y0 = nothing, t0 = nothing; env = nothing)
    y = isnothing(env) ? y0 : env.y
    return reshape(y, length(y), 1)
end

function prepare_action(action0 = nothing, t0 = nothing; env = nothing)
    action = isnothing(env) ? action0 : env.action
    a = clamp(action, -1.0, 1.0)
    return 0.1f0 .* a
end

function reward_function(env)
    return env.reward
end

function in_zone(x::Float32, center::Float32)
    return (center - delta_zone) <= x <= (center + delta_zone)
end

function do_step(env)
    # step index before transition (0-based)
    step_idx = env.steps

    dx = env.p isa AbstractArray ? env.p[1] : env.p
    x_new = clamp(Float32(env.y[1] + dx), 0.0f0, 1.0f0)

    seg = current_segment(step_idx)
    centers_now = zone_centers[:, seg]

    zone_idx = -1
    for i in 1:n_zones
        if in_zone(x_new, centers_now[i])
            zone_idx = i
            break
        end
    end

    reward = 0.0f0
    if 0 <= step_idx < episode_steps && zone_idx > 0
        reward = zone_potential_rewards[zone_idx, step_idx + 1]
    end
    env.reward = [reward]

    # advance time and step index
    new_time = min(env.time + env.dt, env.te)

    env.y[1] .= x_new

    # next state uses centers for the next segment (based on new step index)
    new_state = build_state(x_new, step_idx + 1)
    env.y = new_state

    if new_time >= env.te
        env.done = true
    end

    return new_state
end

function generate_random_init()
    global zone_centers = sample_zone_centers()
    global zone_potential_rewards = sample_zone_potential_rewards()

    x0 = 0.5f0
    step_idx0 = 0

    y_init = build_state(x0, step_idx0)
    env.y0 = deepcopy(y_init)
    env.y = deepcopy(y_init)
    env.state = env.featurize(; env = env)
    env.done = false
    env.time = t0

    return y_init
end

# validation utilities
function generate_validation_set(; n = 100)
    global validation_set = []
    for _ in 1:n
        push!(validation_set, (sample_zone_centers(), sample_zone_potential_rewards()))
    end
    return validation_set
end

# try to load existing set
try
    global validation_set = FileIO.load("./TestEnvironments/Test2/validation_set.jld2", "set")
catch
    global validation_set = generate_validation_set()
    FileIO.save("./TestEnvironments/Test2/validation_set.jld2", "set", validation_set)
end




function validate_agent()
    scores = Float32[]

    for (zc, zpr) in validation_set
        global zone_centers = zc
        global zone_potential_rewards = zpr

        reset!(env)
        
        x0 = 0.5f0
        step_idx0 = 0

        y_init = build_state(x0, step_idx0)
        env.y0 = deepcopy(y_init)
        env.y = deepcopy(y_init)
        env.state = env.featurize(; env = env)
        env.done = false
        env.time = t0

        total_reward = 0.0f0
        while !env.done
            action = if hasproperty(agent.policy, :actor)
                agent.policy.actor.μ(env.state)
            elseif hasproperty(agent.policy, :approximator)
                agent.policy.approximator.actor.μ(env.state)
            elseif hasproperty(agent.policy, :behavior_actor)
                agent.policy.behavior_actor(env.state)
            else
                zeros(Float32, action_dim)
            end

            env(action)
            total_reward += env.reward[1]
        end

        push!(scores, total_reward)
    end

    return scores
end
