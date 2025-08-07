

global set = FileIO.load("./Minimal1/validation_set.jld2","set")
#FileIO.save("./Minimal1/validation_set.jld2","set",set)


global validation_results = FileIO.load("./Minimal1/validation_results.jld2","validation_results")
#FileIO.save("./Minimal1/validation_results.jld2","validation_results",validation_results)


traces = AbstractTrace[]

for (key, value) in validation_results
    trace = box(y=value, name=key, boxpoints="all", quartilemethod="linear")
    push!(traces, trace)
end

plot(traces)

# data = [1,2,3,4,5,6,7,8,9]
# trace1 = box(y=data, boxpoints="all", quartilemethod="linear", name="linear")
# trace2 = box(y=data, boxpoints="all", quartilemethod="inclusive", name="inclusive")
# trace3 = box(y=data, boxpoints="all", quartilemethod="exclusive", name="exclusive")
# plot([trace1, trace2, trace3])




function validate_agent(; optimizer = false)

    global validation_rewards = Float32[]

    for (w, gp) in set

        reset!(env)
        # Perform validation checks on wind and gp
        global wind = [w]
        global grid_price = gp

        y0 = create_state(; generate_day = false)

        env.y0 = deepcopy(y0)
        env.y = deepcopy(y0)
        env.state = env.featurize(; env = env)

        if optimizer
            optimal_actions = optimize_day(5000)
            reward_sum = Float32(sum(evaluate(optimal_actions; collect_rewards = true)))

        else
            reward_sum = 0.0f0
            while !env.done

                if hasproperty(agent.policy, :actor)
                    action = agent.policy.actor.μ(env.state)
                elseif hasproperty(agent.policy, :approximator)
                    action = agent.policy.approximator.actor.μ(env.state)
                end
                
                env(action; reward_shaping = false)

                reward_sum += mean(env.reward)
                
            end
        end

        push!(validation_rewards, reward_sum)
    end

    validation_rewards
end




function generate_validation_set(;n = 200)

    global set = []

    for i in 1:n
        wind = generate_wind()
        gp = generate_grid_price()
        push!(set, (wind, gp))
    end

    return set
end




