using FileIO, JLD2
using RL

# File to store optimal trajectories
trajectories_file = "optimal_trajectories.jld2"



optimal_episodes = 200




# Load or initialize trajectories dictionary
if isfile(trajectories_file)
    trajectories = FileIO.load(trajectories_file, "trajectories")
    println("Loaded existing trajectories file")
else
    # Dictionary to store trajectories for each algorithm type and reward shaping configuration
    trajectories = Dict{String, Dict{String, Any}}()
    for alg in ["SAC_DDPG", "PPO", "PPO2"]
        trajectories[alg] = Dict{String, Any}()
    end
    println("Created new trajectories dictionary")
end


function generate_optimal_trajectories(; steps = 10_000)
    global env, optimal_episodes, action_dim, gamma

    # Initialize all trajectories with their specific structures
    n_envs = 1
    
    # SAC/DDPG trajectory
    sac_trajectory_no_rs = CircularArrayTrajectory(;
        capacity = Int(te / dt) * optimal_episodes,
        state = Float32 => (size(env.state_space)[1], n_envs),
        action = Float32 => (size(env.action_space)[1], n_envs),
        reward = Float32 => (n_envs),
        terminal = Bool => (n_envs,)
    )
    
    sac_trajectory_rs = CircularArrayTrajectory(;
        capacity = Int(te / dt) * optimal_episodes,
        state = Float32 => (size(env.state_space)[1], n_envs),
        action = Float32 => (size(env.action_space)[1], n_envs),
        reward = Float32 => (n_envs),
        terminal = Bool => (n_envs,)
    )

    # PPO trajectory
    ppo_trajectory_no_rs = CircularArrayTrajectory(;
        capacity = Int(te / dt) * optimal_episodes,
        state = Float32 => (size(env.state_space)[1], n_envs),
        action = Float32 => (size(env.action_space)[1], n_envs),
        action_log_prob = Float32 => (n_envs),
        reward = Float32 => (n_envs),
        terminal = Bool => (n_envs,),
        next_values = Float32 => (1, n_envs),
    )
    
    ppo_trajectory_rs = CircularArrayTrajectory(;
        capacity = Int(te / dt) * optimal_episodes,
        state = Float32 => (size(env.state_space)[1], n_envs),
        action = Float32 => (size(env.action_space)[1], n_envs),
        action_log_prob = Float32 => (n_envs),
        reward = Float32 => (n_envs),
        terminal = Bool => (n_envs,),
        next_values = Float32 => (1, n_envs),
    )

    # PPO2 trajectory
    ppo2_trajectory_no_rs = CircularArrayTrajectory(;
        capacity = Int(te / dt) * optimal_episodes,
        state = Float32 => (size(env.state_space)[1], n_envs),
        action = Float32 => (size(env.action_space)[1], n_envs),
        action_log_prob = Float32 => (n_envs),
        reward = Float32 => (n_envs),
        explore_mod = Float32 => (n_envs),
        terminal = Bool => (n_envs,),
        next_values = Float32 => (1, n_envs)
    )
    
    ppo2_trajectory_rs = CircularArrayTrajectory(;
        capacity = Int(te / dt) * optimal_episodes,
        state = Float32 => (size(env.state_space)[1], n_envs),
        action = Float32 => (size(env.action_space)[1], n_envs),
        action_log_prob = Float32 => (n_envs),
        reward = Float32 => (n_envs),
        explore_mod = Float32 => (n_envs),
        terminal = Bool => (n_envs,),
        next_values = Float32 => (1, n_envs)
    )

    global optimal_rewards = Float64[]

    # Single loop to collect all trajectories
    for i in 1:optimal_episodes
        println("Optimized Episode $(i)...")
        reset!(env)
        generate_random_init()

        # Generate optimal actions
        optimal_actions = optimize_day(steps; verbose = false)
        n = 1

        while !is_terminated(env)
            if n <= size(optimal_actions)[2]
                action = hcat(optimal_actions[:,n])
            else
                action = 0.001f0 .* ones(action_dim,1)
            end

            # Store pre-step state and action in all trajectories
            # SAC/DDPG trajectories
            push!(sac_trajectory_no_rs; state=env.state, action=action)
            push!(sac_trajectory_rs; state=env.state, action=action)

            # PPO trajectories (with action_log_prob)
            last_action_log_prob = zeros(Float32, n_envs)
            push!(ppo_trajectory_no_rs;
                state=env.state,
                action=action,
                action_log_prob=last_action_log_prob
            )
            push!(ppo_trajectory_rs;
                state=env.state,
                action=action,
                action_log_prob=last_action_log_prob
            )

            # PPO2 trajectories
            push!(ppo2_trajectory_no_rs;
                state=env.state,
                action=action,
                action_log_prob=last_action_log_prob,
                explore_mod=1.0f0,
            )
            push!(ppo2_trajectory_rs;
                state=env.state,
                action=action,
                action_log_prob=last_action_log_prob,
                explore_mod=1.0f0,
            )

            # Store compute_left for reward shaping
            compute_left_before = env.y[1]
            
            # Step environment without reward shaping
            env(action; reward_shaping=false)
            
            # Get both raw and shaped rewards
            r = env.reward[1]  # Raw reward
            compute_left_after = env.y[1]
            
            # Calculate shaped reward
            beta = 1.0
            r_shaped = r + beta * (compute_left_before - compute_left_after - (gamma-1) * compute_left_after)
            r_shaped *= reward_scale_factor

            # Store rewards and terminal states
            # SAC/DDPG
            push!(sac_trajectory_no_rs[:reward], [r])
            push!(sac_trajectory_rs[:reward], [r_shaped])
            push!(sac_trajectory_no_rs[:terminal], is_terminated(env))
            push!(sac_trajectory_rs[:terminal], is_terminated(env))

            # PPO
            push!(ppo_trajectory_no_rs[:reward], [r])
            push!(ppo_trajectory_rs[:reward], [r_shaped])
            push!(ppo_trajectory_no_rs[:terminal], is_terminated(env))
            push!(ppo_trajectory_rs[:terminal], is_terminated(env))
            push!(ppo_trajectory_no_rs[:next_values], agent.policy.approximator.critic(env.state))
            push!(ppo_trajectory_rs[:next_values], agent.policy.approximator.critic(env.state))

            # PPO2
            push!(ppo2_trajectory_no_rs[:reward], [r])
            push!(ppo2_trajectory_rs[:reward], [r_shaped])
            push!(ppo2_trajectory_no_rs[:terminal], is_terminated(env))
            push!(ppo2_trajectory_rs[:terminal], is_terminated(env))
            push!(ppo2_trajectory_no_rs[:next_values], agent.policy.approximator.critic(env.state))
            push!(ppo2_trajectory_rs[:next_values], agent.policy.approximator.critic(env.state))

            n += 1
        end
    end

    # Store all trajectories in the dictionary
    trajectories["SAC_DDPG"]["no_RS"] = sac_trajectory_no_rs
    trajectories["SAC_DDPG"]["with_RS"] = sac_trajectory_rs
    trajectories["PPO"]["no_RS"] = ppo_trajectory_no_rs
    trajectories["PPO"]["with_RS"] = ppo_trajectory_rs
    trajectories["PPO2"]["no_RS"] = ppo2_trajectory_no_rs
    trajectories["PPO2"]["with_RS"] = ppo2_trajectory_rs

    # Save all trajectories
    FileIO.save(trajectories_file, "trajectories", trajectories)
    println("\nAll optimal trajectories generated and saved to $trajectories_file")
end

# Generate trajectories if they don't exist
if isempty(trajectories["SAC_DDPG"]) || isempty(trajectories["PPO"]) || isempty(trajectories["PPO2"])
    generate_optimal_trajectories()
end