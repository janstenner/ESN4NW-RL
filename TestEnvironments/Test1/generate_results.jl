using JLD2
using FileIO
using PlotlyJS
using Statistics


# include an agent script to ensure an env gets build
include("./SAC2/Test1_SAC2.jl")

# File path for saving results
results_file = "TestEnvironments/Test1/training_results.jld2"


if isfile(results_file)
    results = FileIO.load(results_file, "results")
    println("Loaded existing results file")
else
    results = Dict{String, Dict{Int, Dict{String, Any}}}()
    println("Created new results dictionary")
    FileIO.save(results_file, "results", results)
end

# List of algorithms to test
algorithms = [
    ("SAC", "TestEnvironments/Test1/SAC/Test1_SAC.jl"),
    ("SAC2", "TestEnvironments/Test1/SAC2/Test1_SAC2.jl"),
    ("PPO", "TestEnvironments/Test1/PPO/Test1_PPO.jl"),
    #("PPO2", "TestEnvironments/Test1/PPO2/Test1_PPO2.jl"),
    ("PPO3", "TestEnvironments/Test1/PPO3/Test1_PPO3.jl"),
    ("DDPG", "TestEnvironments/Test1/DDPG/Test1_DDPG.jl")
]

# Function to ensure nested dictionary structure exists
function ensure_nested_dict!(results, alg_name)
    if !haskey(results, alg_name)
        results[alg_name] = Dict{Int, Dict{String, Any}}()
    end
end





function collect_runs(n = 1; selected_algorithms::Vector{String} = String[])
    # Filter algorithms based on input or use all if none specified
    algs_to_run = if isempty(selected_algorithms)
        algorithms
    else
        filter(a -> a[1] in selected_algorithms, algorithms)
    end
    
    # Run training for each algorithm
    for (alg_name, script_path) in algs_to_run
        println("\n=== Testing $alg_name ===")
        
       
        ensure_nested_dict!(results, alg_name)
        
        for i in 1:n
            println("\nStarting training run $i")
            global seed = i
            include(script_path)
            
            # Algorithm-specific default parameters
            default_params = Dict(
                "SAC2" => Dict(
                    "inner_loops" => 10,
                    "outer_loops" => 25,
                    "num_steps" => 10_000
                ),
                "SAC" => Dict(
                    "inner_loops" => 10,
                    "outer_loops" => 25,
                    "num_steps" => 10_000
                ),
                "PPO" => Dict(
                    "inner_loops" => 10,
                    "outer_loops" => 25,
                    "num_steps" => 10_000
                ),
                "PPO2" => Dict(
                    "inner_loops" => 10,
                    "outer_loops" => 25,
                    "num_steps" => 10_000
                ),
                "PPO3" => Dict(
                    "inner_loops" => 10,
                    "outer_loops" => 25,
                    "num_steps" => 10_000
                ),
                "DDPG" => Dict(
                    "inner_loops" => 10,
                    "outer_loops" => 25,
                    "num_steps" => 10_000
                )
            )

            # Set appropriate training parameters
            train_params = Dict{Symbol,Any}(
                :inner_loops => default_params[alg_name]["inner_loops"],
                :outer_loops => default_params[alg_name]["outer_loops"],
                :num_steps => default_params[alg_name]["num_steps"],
                :plot_runs => false,
            )

            
            # Run training with parameters
            train(;train_params...)
            
            # Store only the policies from the agents
            agent_policy = agent.policy
            agent_save_policy = isnothing(agent_save) ? nothing : agent_save.policy
            
            # Store results
            results[alg_name][seed] = Dict(
                "agent_save_policy" => agent_save_policy,
                "agent_policy" => agent_policy,
                "rewards" => hook.rewards,
                "validation_scores" => validation_scores
            )
            
            FileIO.save(results_file, "results", results)
            println("Saved results for $alg_name with seed $seed")
        end
    end

    println("\nAll training runs completed. Results saved to $results_file")

    # Print structure summary
    println("\nFinal results structure:")
    for alg in keys(results)
        println("Algorithm: $alg")
        
        n_seeds = length(keys(results[alg]))
        println("  └─ $n_seeds seeds")
    end
end






function plot_comparison()
    
    # Ensure results dictionary is loaded
    if !@isdefined(results)
        if isfile(results_file)
            global results = FileIO.load(results_file, "results")
            println("Loaded existing results file")
        else
            global results = Dict{String, Dict{Int, Dict{String, Any}}}()
            println("Created new results dictionary")
        end
    end
    
    # Initialize dictionary to store all validation results
    global best_validation_results = Dict{String, Any}()
    global all_validation_results = Dict{String, Dict{Int, Any}}()

    
    # Go through all results and validate each agent
    for alg_name in keys(results)
        
        # Collect scores for all seeds of this configuration
        config_scores = Dict{Int, Vector{Float32}}()
        config_means = Dict{Int, Float64}()
        config_timelines = Dict{Int, Vector{Float32}}()

        all_validation_results[alg_name] = (Float32[], [])
        
        for seed in keys(results[alg_name])
            # Get the saved policy
            saved_policy = results[alg_name][seed]["agent_save_policy"]
            
            # Skip if policy is nothing
            if isnothing(saved_policy)
                println("Skipping $(alg_name)-seed$(seed) (no policy saved)")
                continue
            end
            
            # Construct new agent with saved policy
            global agent = Agent(saved_policy, Trajectory())
            
            # Run validation
            println("Validating $(alg_name)-seed$(seed)...")
            scores = validate_agent()
            
            # Store scores, mean and timeline
            config_scores[seed] = scores
            config_means[seed] = mean(scores)
            config_timelines[seed] = results[alg_name][seed]["validation_scores"]

            append!(all_validation_results[alg_name][1], scores)
            push!(all_validation_results[alg_name][2], results[alg_name][seed]["validation_scores"])
        end
        
        # Find the best seed based on mean score
        if !isempty(config_means)
            best_seed = argmax(config_means)
            key = "$(alg_name)"
            best_validation_results[key] = (
                config_scores[best_seed],
                config_timelines[best_seed]
            )
        end
    end

    
    # First, calculate the order based on mean scores
    order = sort(collect(keys(best_validation_results)), 
                by=key->mean(best_validation_results[key][1]), 
                rev=true)  # descending order

    order_all = sort(collect(keys(all_validation_results)), 
                by=key->mean(all_validation_results[key][1]), 
                rev=true)  # descending order

    # Create color mapping for consistent colors across plots
    color_map = Dict{String, Vector{Int}}()
    
    
    # Define algorithm colors
    for key in order
        if key != "Optimal" && key != "Untrained"
            parts = split(key, "-")
            alg = parts[1]
            
            # Define semi-muted, aesthetic base colors for algorithms
            base_color = if alg == "SAC"
                [184, 71, 82]    # Richer burgundy
            elseif alg == "SAC2" 
                [239, 83, 239]   # Purple
            elseif alg == "PPO"
                [98, 150, 209]   # Brighter steel blue
            elseif alg == "PPO2"
                [139, 173, 115]  # Livelier sage green
            elseif alg == "PPO3"
                [65, 105, 225]   # Royal blue
            else  # DDPG
                [168, 119, 175]  # Brighter purple
            end
            
            
            color_map[key] = base_color
        end
    end

    # Create first plot (validation scores)
    traces1 = AbstractTrace[]
    
    # Add traces in order
    for key in order
        scores, _ = best_validation_results[key]
        color = color_map[key]
        
        push!(traces1, box(
            y=scores,
            name=key,
            boxpoints="all",
            quartilemethod="linear",
            marker_color="rgb($(color[1]), $(color[2]), $(color[3]))"
        ))
    end
    
    # Create and display the validation scores plot
    layout1 = Layout(
        title="Algorithm Performance Comparison - Best Runs",
        yaxis_title="Validation Score",
        showlegend=true,
        legend=attr(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=attr(b=100)
    )
    
    p1 = plot(traces1, layout1)
    display(p1)



    # Create first and second plot (validation scores)
    traces2 = AbstractTrace[]
    
    # Add traces in order
    for key in order_all
        scores, _ = all_validation_results[key]
        color = color_map[key]
        
        push!(traces2, box(
            y=scores,
            name=key,
            boxpoints="all",
            quartilemethod="linear",
            marker_color="rgb($(color[1]), $(color[2]), $(color[3]))"
        ))
    end
    
    # Create and display the validation scores plot
    layout2 = Layout(
        title="Algorithm Performance Comparison - All Runs",
        yaxis_title="Validation Score",
        showlegend=true,
        legend=attr(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=attr(b=100)
    )
    
    p1 = plot(traces2, layout2)
    display(p1)



    # Create third and fourth plot (validation timelines)
    traces3 = AbstractTrace[]
    
    # Add traces in the same order
    for key in order
        _, timeline = best_validation_results[key]
        
        # Skip if timeline is nothing
        if !isnothing(timeline)
            color = color_map[key]
            
            push!(traces3, scatter(
                y=timeline,
                name=key,
                mode="lines",
                line_color="rgb($(color[1]), $(color[2]), $(color[3]))"
            ))
        end
    end
    
    # Create and display the timeline plot
    layout3 = Layout(
        title="Validation Score Timeline During Training",
        yaxis_title="Validation Score",
        xaxis_title="Training Steps",
        showlegend=true,
        legend=attr(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=attr(b=100)
    )
    
    p2 = plot(traces3, layout3)
    display(p2)


    traces4 = AbstractTrace[]
    
    # Add traces in the same order
    for key in order
        _, timelines = all_validation_results[key]
        
        # Skip if timelines is nothing
        if !isnothing(timelines)
            color = color_map[key]
            
            push!(traces4, scatter(
                y=timelines,
                name=key,
                mode="lines",
                line_color="rgb($(color[1]), $(color[2]), $(color[3]))"
            ))
        end
    end
    
    # Create and display the timeline plot
    layout4 = Layout(
        title="Validation Score Timeline During Training",
        yaxis_title="Validation Score",
        xaxis_title="Training Steps",
        showlegend=true,
        legend=attr(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=attr(b=100)
    )
    
    p2 = plot(traces4, layout4)
    display(p2)
    
    #return p1, p2  # Return both plot objects
end
