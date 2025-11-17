using Zygote:ignore
using JSON, CodecZlib



function train_same_day(n = 100, days_per_trajectory = 10; show_plots = false, inline_trajectory_analysis = false)

    agent.policy.update_step = 0


    for i in 1:n

        @show i

        global multiple_day_trajectory = CircularArrayTrajectory(;
                capacity = 288 * days_per_trajectory,
                state = Float32 => (size(env.state_space)[1], 1),
                action = Float32 => (size(env.action_space)[1], 1),
                reward = Float32 => (1),
                terminal = Bool => (1,),
        )

        for i in 1:days_per_trajectory
            render_run(; new_day = false, exploration = true, return_plot = !show_plots,)

            for j in 1:length(day_trajectory)
                push!(multiple_day_trajectory[:state], day_trajectory[:state][:,:,j])
                push!(multiple_day_trajectory[:action], day_trajectory[:action][:,:,j])
                push!(multiple_day_trajectory[:reward], day_trajectory[:reward][:,j])
                push!(multiple_day_trajectory[:terminal], day_trajectory[:terminal][:,j])
            end
        end


        # if inline_trajectory_analysis
        #     #fig = trajectory_analysis(0; normalize=true, full_y_axis=true)
        #     #display(fig)
        #     fig = trajectory_analysis(0; normalize=false, full_y_axis=true)
        #     display(fig)
        # end



        RL.on_policy_critic_update(agent.policy, multiple_day_trajectory; whole_trajectory = true)


        if inline_trajectory_analysis
            #fig = trajectory_analysis(0; normalize=true, full_y_axis=true)
            #display(fig)
            fig = trajectory_analysis(0; normalize=false, full_y_axis=true)
            display(fig)
        end

        
    end

    println("Training complete")
end




function trajectory_analysis(n = 100; normalize = false, substract_min = false, full_y_axis = false)
    global multiple_day_trajectory

    if n>0
        multiple_day_trajectory = CircularArrayTrajectory(;
                capacity = 288 * n,
                state = Float32 => (size(env.state_space)[1], 1),
                action = Float32 => (size(env.action_space)[1], 1),
                reward = Float32 => (1),
                terminal = Bool => (1,),
        )

        for i in 1:n
            render_run(;
                new_day = false,
                exploration = true,
                return_plot = true, # we dont need to display the plot
                )

            for j in 1:length(day_trajectory)
                push!(multiple_day_trajectory[:state], day_trajectory[:state][:,:,j])
                push!(multiple_day_trajectory[:action], day_trajectory[:action][:,:,j])
                push!(multiple_day_trajectory[:reward], day_trajectory[:reward][:,j])
                push!(multiple_day_trajectory[:terminal], day_trajectory[:terminal][:,j])
            end
        end
    end

    whole_trajectory = true


    xx = collect(dt/60:dt/60:te/60)

    p = agent.policy

    γ, τ, α = p.γ, p.τ, p.α

    r = collect(multiple_day_trajectory[:reward])
    t = collect(multiple_day_trajectory[:terminal])
    s = collect(multiple_day_trajectory[:state])
    a = collect(multiple_day_trajectory[:action])
    next_states = deepcopy(circshift(s, (0,0,-1)))
    next_states[:,:, end] = zeros(Float32, size(s, 1), size(s, 2))  # terminal state

    acc_mu, acc_logp = RL.antithetic_mean_sac2(p, next_states, α)
    
    logp_π′ = acc_logp

    next_values = acc_mu

    if whole_trajectory
        next_values[:,:, end] .*= 0.0f0     # terminal states
    end

    n_envs = size(t, 1)
    next_values = reshape( next_values, n_envs, :)
    
    targets = td_lambda_targets(r, t, next_values, γ; λ = p.λ_targets)


    # calculate real returns for each state
    terminal = t
    states = s
    actions = a
    rewards = r

    global returns = zeros(Float32, length(multiple_day_trajectory))
    for i in length(multiple_day_trajectory):-1:1
        if terminal[i]
            returns[i] = rewards[i]
        else
            returns[i] = rewards[i] + γ * returns[i+1]
        end
    end

    values1 = reshape( agent.policy.qnetwork1( vcat(states, actions) ), 1, :)
    values2 = reshape( agent.policy.qnetwork2( vcat(states, actions) ), 1, :)

    values = min.(values1, values2)


    # Process states and count visits
    states_array = multiple_day_trajectory[:state]
    n_states = size(states_array, 3)  # number of states recorded

    # Create bins for load_left
    if full_y_axis
        min_load = 0.0
    else
        min_load = minimum(states_array[1, 1, :])
    end
    
    global load_bins = collect(LinRange(min_load,1,200))
    n_load_bins = length(load_bins)
    time_bins = 1:288  # Time steps are already discrete 1-288
    # Initialize visitation matrix
    global state_visits = zeros(Int, n_load_bins-1, 288)  # -1 because we're counting intervals between bin edges
    global state_returns = ones(Float32, n_load_bins-1, 288) .* minimum(returns)
    global state_targets = ones(Float32, n_load_bins-1, 288) .* minimum(targets)
    global state_values = ones(Float32, n_load_bins-1, 288) .* minimum(values)

    min_action = minimum(actions[1, :, :])
    max_action = maximum(actions[1, :, :])
    action_bins = collect(LinRange(min_action, max_action, 200))
    n_action_bins = length(action_bins)

    global state_action_visits = zeros(Int, n_action_bins-1, 288)
    global state_action_values = ones(Float32, n_action_bins-1, 288) .* minimum(values)
    global state_action_targets = ones(Float32, n_action_bins-1, 288) .* minimum(targets)

    # For each state, increment the appropriate cell in the visitation matrix
    for i in 1:n_states
        load_left = states_array[1, 1, i]  # state[1] - load_left value
        time_step = round(Int, states_array[end, 1, i] * te / dt + 1)        # time step (1-288)
        
        # Find the appropriate load_left bin
        load_bin = searchsortedfirst(load_bins, load_left) - 1
        load_bin = clamp(load_bin, 1, n_load_bins-1)

        # Increment the visitation count
        if state_visits[load_bin, time_step] == 0
            first = true
        else
            first = false
        end

        state_visits[load_bin, time_step] += 1
        if first
            state_returns[load_bin, time_step] = returns[i]
            state_targets[load_bin, time_step] = targets[i]
            state_values[load_bin, time_step] = values[i]
        else
            state_returns[load_bin, time_step] += returns[i]
            state_targets[load_bin, time_step] += targets[i]
            state_values[load_bin, time_step] += values[i]
        end


        # Find the appropriate action bin
        action_bin = searchsortedfirst(action_bins, actions[i]) - 1
        action_bin = clamp(action_bin, 1, n_action_bins-1)

        # Increment the visitation count
        if state_action_visits[action_bin, time_step] == 0
            first_state_action_visit = true
        else
            first_state_action_visit = false
        end

        state_action_visits[action_bin, time_step] += 1
        if first_state_action_visit
            state_action_values[action_bin, time_step] = values[i]
            state_action_targets[action_bin, time_step] = targets[i]
        else
            state_action_values[action_bin, time_step] += values[i]
            state_action_targets[action_bin, time_step] += targets[i]
        end
    end

    state_returns ./= state_visits .+ (state_visits .== 0) # avoid NaN
    state_targets ./= state_visits .+ (state_visits .== 0)
    state_values ./= state_visits .+ (state_visits .== 0)

    state_action_values ./= state_action_visits .+ (state_action_visits .== 0)
    state_action_targets ./= state_action_visits .+ (state_action_visits .== 0)

    if normalize || substract_min
        # Normalize each time step independently for each matrix
        for t in 1:288
            if !isempty(findall(x -> x > 0,  state_visits[:,t]))
                # For returns
                min_val = minimum(state_returns[state_visits[:, t] .> 0, t])
                max_val = maximum(state_returns[state_visits[:, t] .> 0, t])
                if min_val != max_val
                    state_returns[:, t] = (state_returns[:, t] .- min_val)
                    if normalize
                        state_returns[:, t] ./= (max_val - min_val)
                    end
                else
                    state_returns[:, t] .= 1.0
                end

                # For targets
                min_val = minimum(state_targets[state_visits[:, t] .> 0, t])
                max_val = maximum(state_targets[state_visits[:, t] .> 0, t])
                if min_val != max_val
                    state_targets[:, t] = (state_targets[:, t] .- min_val)
                    if normalize
                        state_targets[:, t] ./= (max_val - min_val)
                    end
                else
                    state_targets[:, t] .= 1.0
                end

                # For values
                min_val = minimum(state_values[state_visits[:, t] .> 0, t])
                max_val = maximum(state_values[state_visits[:, t] .> 0, t])
                if min_val != max_val
                    state_values[:, t] = (state_values[:, t] .- min_val)
                    if normalize
                        state_values[:, t] ./= (max_val - min_val)
                    end
                else
                    state_values[:, t] .= 1.0
                end

                # For critic2_values
                min_val = minimum(state_action_values[state_action_visits[:, t] .> 0, t])
                max_val = maximum(state_action_values[state_action_visits[:, t] .> 0, t])
                if min_val != max_val
                    state_action_values[:, t] = (state_action_values[:, t] .- min_val)
                    if normalize
                        state_action_values[:, t] ./= (max_val - min_val)
                    end
                else
                    state_action_values[:, t] .= 1.0
                end

                # For critic2_targets
                min_val = minimum(state_action_targets[state_action_visits[:, t] .> 0, t])
                max_val = maximum(state_action_targets[state_action_visits[:, t] .> 0, t])
                if min_val != max_val
                    state_action_targets[:, t] = (state_action_targets[:, t] .- min_val)
                    if normalize
                        state_action_targets[:, t] ./= (max_val - min_val)
                    end
                else
                    state_action_targets[:, t] .= 1.0
                end
            end

        end

        state_returns[state_visits .== 0, :] .= -0.2
        state_targets[state_visits .== 0, :] .= -0.2
        state_values[state_visits .== 0, :] .= -0.2

        state_action_values[state_action_visits .== 0, :] .= -0.2
        state_action_targets[state_action_visits .== 0, :] .= -0.2
    end

    # Create color scales
    colorscale1 = [[0.0, "rgb(5, 0, 5)"], [0.01, "rgb(40, 0, 60)"], [0.3, "rgb(160, 0, 200)"], [0.75, "rgb(210, 0, 255)"], [1.0, "rgb(240, 160, 255)"]]
    colorscale2 = [[0.0, "rgb(5, 0, 5)"], [0.1, "rgb(200, 0, 0)"], [0.5, "rgb(210, 210, 0)"], [0.75, "rgb(0, 210, 0)"], [1.0, "rgb(140, 255, 255)"]]

    # Create subplots layout
    layout = Layout(
        grid=attr(rows=2, columns=2, pattern="independent", rowgap=0.01, colgap=0.01),
        title="State Analysis Heatmaps",
        # width=1200,
        # height=1000,
        showlegend=false,
        margin=attr(t=50, pad=0),
        annotations=[
            attr(text="State Visits", x=0.05, y=0.715, xref="paper", yref="paper", showarrow=false, font_size=16),
            attr(text="State Returns", x=0.7, y=0.715, xref="paper", yref="paper", showarrow=false, font_size=16),
            attr(text="State Targets", x=0.05, y=0.325, xref="paper", yref="paper", showarrow=false, font_size=16),
            attr(text="State Values", x=0.7, y=0.325, xref="paper", yref="paper", showarrow=false, font_size=16),
            attr(text="State Action Targets", x=0.05, y=-0.05, xref="paper", yref="paper", showarrow=false, font_size=16),
            attr(text="State Action Values", x=0.7, y=-0.05, xref="paper", yref="paper", showarrow=false, font_size=16),
        ]
    )

    # Create all four heatmaps
    heatmap1 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=state_visits,
        colorscale=colorscale1,
        showscale = false,
        colorbar=attr(title="Visits", x=0.45, y=0.9),
        name="State Visits"
    ))

    heatmap2 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=state_returns,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Returns", x=0.95, y=0.9),
        name="State Returns"
    ))

    heatmap3 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=state_targets,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Targets", x=0.45, y=0.2),
        name="State Targets"
    ))

    heatmap4 = plot(PlotlyJS.heatmap(
        x=1:288, y=load_bins[1:end-1], z=state_values,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Values", x=0.95, y=0.2),
        name="State Values"
    ))

    heatmap5 = plot(PlotlyJS.heatmap(
        x=1:288, y=action_bins[1:end-1], z=state_action_targets,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Critic2 Targets", x=0.95, y=0.2),
        name="Critic2 Targets"
    ))

    heatmap6 = plot(PlotlyJS.heatmap(
        x=1:288, y=action_bins[1:end-1], z=state_action_values,
        colorscale=colorscale2,
        showscale = false,
        colorbar=attr(title="Critic2 Values", x=0.95, y=0.2),
        name="Critic2 Values"
    ))

    # Create and display the combined plot
    fig = [heatmap1 heatmap2; heatmap3 heatmap4; heatmap5 heatmap6]

    relayout!(fig, layout.fields)

    fig
end


function create_training_movie(n_frames = 300, training_dict = nothing; full_y_axis = true, normalize = true)

    rm(dirpath * "/training_frames/", recursive=true, force=true)
    mkdir(dirpath * "/training_frames/")

    for frame in 1:n_frames

        fig = trajectory_analysis(30; normalize = normalize, full_y_axis = full_y_axis)

        if !isnothing(training_dict)
            push!(training_dict["state_visits"], deepcopy(state_visits))
            push!(training_dict["state_returns"], deepcopy(state_returns))
            push!(training_dict["state_targets"], deepcopy(state_targets))
            push!(training_dict["state_values"], deepcopy(state_values))
        end

        PlotlyJS.savefig(fig, dirpath * "/training_frames//a$(lpad(string(frame), 5, '0')).png"; width=1800, height=1600)

        if frame != n_frames
            train_same_day(1,1;)
        end
    end

    rm(dirpath * "/training.mp4", force=true)
    run(`ffmpeg -framerate 16 -i $(dirpath * "/training_frames/a%05d.png") -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le $(dirpath * "/training.mp4")`)
end

function plot_value_target_diff()

    subst = abs.(state_values-state_targets)
    subst[state_visits .== 0, :] .= 0.0

    colorscale2 = [[0.0, "rgb(5, 0, 5)"], [0.1, "rgb(200, 0, 0)"], [0.5, "rgb(210, 210, 0)"], [0.75, "rgb(0, 210, 0)"], [1.0, "rgb(140, 255, 255)"]]

    plot(PlotlyJS.heatmap(
               x=1:288, y=load_bins[1:end-1], z=subst,
                       colorscale=colorscale2,
                               showscale = false,
               colorbar=attr(title="Targets", x=0.45, y=0.2),
                       name="State Values - Targets"
           ))
end

