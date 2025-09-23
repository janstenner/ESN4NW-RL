using JLD2
using FileIO
using PlotlyJS
using Statistics

# File path for saving results
results_file = "training_results.jld2"

# Initialize or load results dictionary with hierarchical structure
if isfile(results_file)
    results = FileIO.load(results_file, "results")
    println("Loaded existing results file")
else
    # Now we have another level in the hierarchy for reward_shaping
    results = Dict{String, Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}}()
    println("Created new results dictionary")
    FileIO.save(results_file, "results", results)
end

# List of algorithms to test
algorithms = [
    ("SAC", "Minimal1/SAC/Minimal1_SAC.jl"),
    ("PPO", "Minimal1/PPO/Minimal1_PPO.jl"),
    ("PPO2", "Minimal1/PPO2/Minimal1_PPO2.jl"),
    ("DDPG", "Minimal1/DDPG/Minimal1_DDPG.jl")
]

# Function to ensure nested dictionary structure exists
function ensure_nested_dict!(results, alg_name, il_type, reward_shaping)
    if !haskey(results, alg_name)
        results[alg_name] = Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}()
    end
    if !haskey(results[alg_name], il_type)
        results[alg_name][il_type] = Dict{String, Dict{Int, Dict{String, Any}}}()
    end
    if !haskey(results[alg_name][il_type], reward_shaping)
        results[alg_name][il_type][reward_shaping] = Dict{Int, Dict{String, Any}}()
    end
end

trajectories_file = "optimal_trajectories.jld2"
trajectories = FileIO.load(trajectories_file, "trajectories")

function collect_runs(n = 5)
    # Run training for each algorithm
    for (alg_name, script_path) in algorithms
        println("\n=== Testing $alg_name ===")
        
        # Run all combinations of IL and reward_shaping
        for il_type in ["no_IL", "IL"]
            for rs_type in ["with_RS", "no_RS"]
                println("\nRunning $(il_type) with $(rs_type):")
                ensure_nested_dict!(results, alg_name, il_type, rs_type)
                
                for i in 1:n
                    println("\nStarting training run $i")
                    global seed = i
                    include(script_path)
                    
                    # Algorithm-specific default parameters
                    default_params = Dict(
                        "SAC" => Dict(
                            "inner_loops" => 1,
                            "outer_loops" => 100,
                            "optimal_trainings" => 1000,
                            "num_steps" => 10_000
                        ),
                        "PPO" => Dict(
                            "inner_loops" => 6,
                            "outer_loops" => 300,
                            "optimal_trainings" => 1,
                            "num_steps" => 10_000
                        ),
                        "PPO2" => Dict(
                            "inner_loops" => 6,
                            "outer_loops" => 300,
                            "optimal_trainings" => 1,
                            "num_steps" => 10_000
                        ),
                        "DDPG" => Dict(
                            "inner_loops" => 1,
                            "outer_loops" => 100,
                            "optimal_trainings" => 1000,
                            "num_steps" => 10_000
                        )
                    )

                    # Set appropriate training parameters
                    train_params = Dict{Symbol,Any}(
                        :inner_loops => default_params[alg_name]["inner_loops"],
                        :outer_loops => default_params[alg_name]["outer_loops"],
                        :num_steps => default_params[alg_name]["num_steps"],
                        :plot_runs => false,
                        :reward_shaping => (rs_type == "with_RS")
                    )

                    if il_type == "IL"
                        train_params[:optimal_trainings] = default_params[alg_name]["optimal_trainings"]
                        global optimal_trajectory = trajectories[(alg_name == "SAC" || alg_name == "DDPG") ? "SAC_DDPG" : alg_name][rs_type]
                    else
                        train_params[:optimal_trainings] = 0
                    end
                    
                    # Run training with parameters
                    train(;train_params...)
                    
                    # Store only the policies from the agents
                    agent_policy = agent.policy
                    agent_save_policy = isnothing(agent_save) ? nothing : agent_save.policy
                    
                    # Store results
                    results[alg_name][il_type][rs_type][seed] = Dict(
                        "agent_save_policy" => agent_save_policy,
                        "agent_policy" => agent_policy,
                        "rewards" => hook.rewards,
                        "validation_scores" => validation_scores
                    )
                    
                    FileIO.save(results_file, "results", results)
                    println("Saved results for $alg_name ($il_type, $rs_type) with seed $seed")
                end
            end
        end
    end

    println("\nAll training runs completed. Results saved to $results_file")

    # Print structure summary
    println("\nFinal results structure:")
    for alg in keys(results)
        println("Algorithm: $alg")
        for il_type in keys(results[alg])
            n_seeds = length(keys(results[alg][il_type]))
            println("  └─ $il_type: $n_seeds seeds")
        end
    end
end



function clear_all_trajectories!()
    # Ensure results dictionary is loaded
    if !@isdefined(results)
        if isfile(results_file)
            global results = FileIO.load(results_file, "results")
            println("Loaded existing results file")
        else
            println("No results file found")
            return
        end
    end

    # Track number of trajectories cleared
    cleared_count = 0

    # Go through all results
    for alg_name in keys(results)
        # Skip Optimal as it doesn't contain agents
        alg_name == "Optimal" && continue

        for il_type in keys(results[alg_name])
            for rs_type in keys(results[alg_name][il_type])
                for seed in keys(results[alg_name][il_type][rs_type])
                    # Convert stored agent to just its policy
                    if haskey(results[alg_name][il_type][rs_type][seed], "agent")
                        agent = results[alg_name][il_type][rs_type][seed]["agent"]
                        if !isnothing(agent)
                            results[alg_name][il_type][rs_type][seed]["agent_policy"] = agent.policy
                            delete!(results[alg_name][il_type][rs_type][seed], "agent")
                            cleared_count += 1
                        end
                    end

                    # Convert stored agent_save to just its policy
                    if haskey(results[alg_name][il_type][rs_type][seed], "agent_save")
                        agent_save = results[alg_name][il_type][rs_type][seed]["agent_save"]
                        if !isnothing(agent_save)
                            results[alg_name][il_type][rs_type][seed]["agent_save_policy"] = agent_save.policy
                            delete!(results[alg_name][il_type][rs_type][seed], "agent_save")
                            cleared_count += 1
                        end
                    end
                end
            end
        end
    end

    # Save updated results
    FileIO.save(results_file, "results", results; compress=Lz4Filter())
    println("Cleared $cleared_count trajectories and saved results to file")
end


function plot_validation_comparison()
    # Include validation script
    include("Minimal1/Validation_Minimal1.jl")
    
    # Ensure results dictionary is loaded
    if !@isdefined(results)
        if isfile(results_file)
            global results = FileIO.load(results_file, "results")
            println("Loaded existing results file")
        else
            global results = Dict{String, Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}}()
            println("Created new results dictionary")
        end
    end
    
    # Check for cached optimal baseline scores from various sources
    optimal_scores = if haskey(results, "Optimal") && 
                       haskey(results["Optimal"], "baseline") && 
                       haskey(results["Optimal"]["baseline"], "scores") &&
                       haskey(results["Optimal"]["baseline"]["scores"], 1)
        println("Using cached optimal baseline scores from results...")
        results["Optimal"]["baseline"]["scores"][1]["data"]
    elseif @isdefined(validation_results) && haskey(validation_results, "optimizer")
        println("Using optimal baseline scores from validation_results...")
        validation_results["optimizer"]
    else
        println("Computing optimal baseline scores...")
        global optimal_scores = validate_agent(optimizer = true)
        # Cache the optimal scores with proper nested structure
        if !haskey(results, "Optimal")
            results["Optimal"] = Dict{String, Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}}()
        end
        if !haskey(results["Optimal"], "baseline")
            results["Optimal"]["baseline"] = Dict{String, Dict{Int, Dict{String, Any}}}()
        end
        if !haskey(results["Optimal"]["baseline"], "scores")
            results["Optimal"]["baseline"]["scores"] = Dict{Int, Dict{String, Any}}()
        end
        results["Optimal"]["baseline"]["scores"][1] = Dict{String, Any}("data" => optimal_scores)
        # Save updated results to file
        FileIO.save(results_file, "results", results)
        println("Saved optimal baseline scores to results file")
        optimal_scores
    end
    
    # Initialize dictionary to store all validation results
    global best_validation_results = Dict{String, Vector{Float32}}()
    best_validation_results["Optimal"] = optimal_scores
    
    # Add untrained baseline if available
    if @isdefined(validation_results) && haskey(validation_results, "untrained")
        println("Adding untrained baseline scores...")
        best_validation_results["Untrained"] = validation_results["untrained"]
    end
    
    # Go through all results and validate each agent
    for alg_name in keys(results)
        # Skip the Optimal results as they're handled separately
        alg_name == "Optimal" && continue
        
        for il_type in keys(results[alg_name])
            for rs_type in keys(results[alg_name][il_type])
                # Collect scores for all seeds of this configuration
                config_scores = Dict{Int, Vector{Float32}}()
                config_means = Dict{Int, Float64}()
                
                for seed in keys(results[alg_name][il_type][rs_type])
                    # Get the saved policy
                    saved_policy = results[alg_name][il_type][rs_type][seed]["agent_save_policy"]
                    
                    # Skip if policy is nothing
                    if isnothing(saved_policy)
                        println("Skipping $(alg_name)-$(il_type)-$(rs_type)-seed$(seed) (no policy saved)")
                        continue
                    end
                    
                    # Construct new agent with saved policy
                    global agent = Agent(saved_policy, Trajectory())
                    
                    # Run validation
                    println("Validating $(alg_name)-$(il_type)-$(rs_type)-seed$(seed)...")
                    scores = validate_agent()
                    
                    # Store scores and mean
                    config_scores[seed] = scores
                    config_means[seed] = mean(scores)
                end
                
                # Find the best seed based on mean score
                if !isempty(config_means)
                    best_seed = argmax(config_means)
                    key = "$(alg_name)-$(il_type)-$(rs_type)"
                    best_validation_results[key] = config_scores[best_seed]
                end
            end
        end
    end
    
    # Create box plots
    traces = AbstractTrace[]
    
    # Add optimal baseline first
    push!(traces, box(
        y=best_validation_results["Optimal"],
        name="Optimal",
        boxpoints="all",
        quartilemethod="linear",
        marker_color="rgb(76, 175, 80)"  # Muted forest green
    ))
    
    # Add untrained baseline next if available
    if haskey(best_validation_results, "Untrained")
        push!(traces, box(
            y=best_validation_results["Untrained"],
            name="Untrained",
            boxpoints="all",
            quartilemethod="linear",
            marker_color="rgb(239, 83, 80)"  # Reddish color
        ))
    end
    
    # Add all other results
    for (key, value) in sort(collect(best_validation_results))
        if key != "Optimal" && key != "Untrained"
            # Extract algorithm, IL type and RS type for coloring
            parts = split(key, "-")
            alg = parts[1]
            il_type = parts[2]
            rs_type = parts[3]
            
            # Define semi-muted, aesthetic base colors for algorithms
            base_color = if alg == "SAC"
                [184, 71, 82]    # Richer burgundy
            elseif alg == "PPO"
                [98, 150, 209]   # Brighter steel blue
            elseif alg == "PPO2"
                [139, 173, 115]  # Livelier sage green
            else  # DDPG
                [168, 119, 175]  # Brighter purple
            end
            
            # IL affects hue (shifts color)
            if il_type == "IL"
                base_color = base_color .* 1.3 # Slightly lighter
            end
            
            # RS affects saturation
            if rs_type == "with_RS"
                # Decrease saturation by moving towards gray while keeping luminance
                luminance = sum(base_color) / 3
                base_color = round.(Int, base_color .* 0.5 .+ luminance * 0.5)
            end
            
            color = "rgb($(base_color[1]), $(base_color[2]), $(base_color[3]))"
            
            push!(traces, box(
                y=value,
                name=key,
                boxpoints="all",
                quartilemethod="linear",
                marker_color=color
            ))
        end
    end
    
    # Create and display the plot with a more readable layout
    layout = Layout(
        title="Algorithm Performance Comparison",
        yaxis_title="Validation Score",
        #boxmode="group",
        showlegend=true,
        legend=attr(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=attr(b=100)  # Add bottom margin for legend
    )
    
    p = plot(traces, layout)
    display(p)
    
    return p  # Return the plot object in case it's needed
end
