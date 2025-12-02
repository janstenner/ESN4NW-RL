using Random
using IntervalSets
using RL
using FileIO, JLD2



validation_scores = []



# configuration
n_random_variables = 3

t0 = 0.0f0
te = 1.0f0
dt = 0.05f0
min_best_episode = 1

max_time_steps = Int(floor((te - t0) / dt)) + 1
random_state_values = rand(Float32, n_random_variables, max_time_steps)
reward_noise_values = randn(Float32, max_time_steps)


action_dim = 1

function time_to_index(time_value)
    idx = Int(floor((time_value - t0) / dt)) + 1
    return clamp(idx, 1, max_time_steps)
end

function build_state(time_value)
    idx = time_to_index(time_value)
    time_fraction = clamp(Float32(time_value / te), 0.0f0, 1.0f0)
    return vcat(Float32[time_fraction], random_state_values[:, idx])
end

y0 = build_state(t0)
state_dim = length(y0)

sim_space = Space(fill(0..1, (state_dim)))

function featurize(y0 = nothing, t0 = nothing; env = nothing)
    y = isnothing(env) ? y0 : env.y
    return reshape(y, length(y), 1)
end

function prepare_action(action0 = nothing, t0 = nothing; env = nothing)
    action = isnothing(env) ? action0 : env.action
    action = (action .+ 1) .* 0.5
    clamp!(action, 0.0, 1.0)
    return action
end

function reward_function(env)
    return env.reward
end

function do_step(env)
    p_val = env.p isa AbstractArray ? env.p[1] : env.p
    p_val = Float32(p_val)

    idx = time_to_index(env.time)
    noise = reward_noise_values[idx]

    reward = p_val * (sin(2f0 * Float32(pi) * Float32(env.time / env.te)) * 0.2f0 + noise)
    env.reward = [reward]

    new_time = min(env.time + env.dt, env.te)

    new_idx = time_to_index(new_time)
    new_state = Float32[Float32(new_time / env.te); random_state_values[:, new_idx]...]

    if new_time >= env.te
        env.done = true
    end

    env.y = new_state

    return new_state
end


function generate_random_init()
    global random_state_values = rand(Float32, n_random_variables, max_time_steps)
    global reward_noise_values = randn(Float32, max_time_steps)

    y_init = build_state(t0)

    env.y0 = deepcopy(y_init)
    env.y = deepcopy(y_init)
    env.state = env.featurize(; env = env)
    env.done = false
    env.time = t0

    return y_init
end



global set = FileIO.load("./TestEnvironments/Test1/validation_set.jld2","set")
#FileIO.save("./TestEnvironments/Test1/validation_set.jld2","set",set)


function generate_validation_set(;n = 100)

    global set = []

    for i in 1:n
        random_state_values = rand(Float32, n_random_variables, max_time_steps)
        reward_noise_values = randn(Float32, max_time_steps)
        push!(set, (random_state_values, reward_noise_values))
    end

    return set
end


function validate_agent()
    scores = Float32[]
    
    for (rsv, rnv) in set

        global random_state_values = rsv
        global reward_noise_values = rnv

        reset!(env)
        
        y_init = build_state(t0)

        env.y0 = deepcopy(y_init)
        env.y = deepcopy(y_init)
        env.state = env.featurize(; env = env)
        env.done = false
        env.time = t0
        
        total_reward = 0.0f0
        while !env.done
            
            if hasproperty(agent.policy, :actor)
                action = agent.policy.actor.μ(env.state)
            elseif hasproperty(agent.policy, :approximator)
                action = agent.policy.approximator.actor.μ(env.state)
            elseif hasproperty(agent.policy, :behavior_actor)
                action = agent.policy.behavior_actor(env.state)
            end

            env(action)

            total_reward += env.reward[1]
        end
        
        push!(scores, total_reward)
    end
    
    return scores
end





function train(use_random_init = true; visuals = false, num_steps = 10_000, inner_loops = 10, outer_loops = 10, plot_runs = true)
    
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


    global validation_scores
    global agent_save

    

    for j = 1:outer_loops

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

            current_score = mean(validate_agent())
            if !isempty(validation_scores) && current_score > maximum(validation_scores)
                agent_save = deepcopy(agent)
            end
            
            push!(validation_scores, current_score)

            if !isempty(validation_scores)
                println(lineplot(validation_scores, title="Validation scores", xlabel="Episode", ylabel="Score", color=:cyan))

                println("Best validation score: $(maximum(validation_scores))")
            end
            
        end

        if plot_runs
            p1 = render_run(; exploration = true)
        end

    end

    if visuals && false
        rm(dirpath * "/training.mp4", force=true)
        run(`ffmpeg -framerate 16 -i $(dirpath * "/training_frames/a%05d.png") -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le $(dirpath * "/training.mp4")`)
    end

    #save()
end





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


function plot_rewards(smoothing = 30)
    to_plot = Float64[]
    for i in smoothing:length(hook.rewards)
        push!(to_plot, mean(hook.rewards[i+1-smoothing:i]))
    end

    p = plot(to_plot)
    display(p)
end
