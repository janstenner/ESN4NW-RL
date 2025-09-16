using JLD2
using FileIO

# File path for saving results
results_file = "training_results.jld2"

# Initialize or load results dictionary with hierarchical structure
if isfile(results_file)
    results = FileIO.load(results_file, "results")
    println("Loaded existing results file")
else
    results = Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}()
    println("Created new results dictionary")
    FileIO.save(results_file, "results", results)
end

# List of algorithms to test
algorithms = [
    ("SAC", "SAC/Minimal1_SAC.jl"),
    ("PPO", "PPO/Minimal1_PPO.jl"),
    ("PPO2", "PPO2/Minimal1_PPO2.jl"),
    ("DDPG", "DDPG/Minimal1_DDPG.jl")
]

# Function to ensure nested dictionary structure exists
function ensure_nested_dict!(results, alg_name, il_type)
    if !haskey(results, alg_name)
        results[alg_name] = Dict{String, Dict{Int, Dict{String, Any}}}()
    end
    if !haskey(results[alg_name], il_type)
        results[alg_name][il_type] = Dict{Int, Dict{String, Any}}()
    end
end

function collect_runs(n = 5)
    # Run training for each algorithm
    for (alg_name, script_path) in algorithms
        println("\n=== Testing $alg_name ===")
        
        # First run without IL
        println("\nRunning without Imitation Learning:")
        ensure_nested_dict!(results, alg_name, "no_IL")
        
        for i in 1:n
            println("\nStarting training run $i")
            global seed = i
            include(script_path)
            train()
            
            results[alg_name]["no_IL"][seed] = Dict(
                "agent_save" => agent_save,
                "agent" => agent,
                "rewards" => hook.rewards
            )
            
            FileIO.save(results_file, "results", results)
            println("Saved results for $alg_name (no IL) with seed $seed")
        end
        
        # Then run with IL
        println("\nRunning with Imitation Learning:")
        ensure_nested_dict!(results, alg_name, "IL")
        
        for i in 1:5
            println("\nStarting training run $i")
            global seed = i
            include(script_path)
            train(;optimal_trainings=80)
            
            results[alg_name]["IL"][seed] = Dict(
                "agent_save" => agent_save,
                "agent" => agent,
                "rewards" => hook.rewards
            )
            
            FileIO.save(results_file, "results", results)
            println("Saved results for $alg_name (with IL) with seed $seed")
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



function plot_validation_comparison()
    # Include validation script
    include("Validation_Minimal1.jl")
    
    # Get optimal baseline scores
    println("Computing optimal baseline scores...")
    optimal_scores = validate_agent(optimizer = true)
    
    # Initialize dictionary to store all validation results
    validation_results = Dict{String, Vector{Float32}}()
    validation_results["Optimal"] = optimal_scores
    
    # Go through all results and validate each agent
    for alg_name in keys(results)
        for il_type in keys(results[alg_name])
            for seed in keys(results[alg_name][il_type])
                # Set the global agent to the saved agent
                global agent = results[alg_name][il_type][seed]["agent_save"]
                
                # Skip if agent is nothing (sometimes happens if training wasn't successful)
                if isnothing(agent)
                    println("Skipping $(alg_name)-$(il_type)-seed$(seed) (no agent saved)")
                    continue
                end
                
                # Run validation
                println("Validating $(alg_name)-$(il_type)-seed$(seed)...")
                scores = validate_agent()
                
                # Store results with descriptive key
                key = "$(alg_name)-$(il_type)-s$(seed)"
                validation_results[key] = scores
            end
        end
    end
    
    # Create box plots
    traces = AbstractTrace[]
    
    # Add optimal baseline first
    push!(traces, box(
        y=validation_results["Optimal"],
        name="Optimal",
        boxpoints="all",
        quartilemethod="linear",
        marker_color="rgb(0, 255, 0)"  # Green for optimal
    ))
    
    # Add all other results
    for (key, value) in sort(collect(validation_results))
        if key != "Optimal"
            # Extract algorithm and IL type for coloring
            alg = split(key, "-")[1]
            il_type = split(key, "-")[2]
            
            # Choose color based on algorithm and IL type
            color = if il_type == "IL"
                if alg == "SAC"
                    "rgb(255, 100, 100)"  # Light red
                elseif alg == "PPO"
                    "rgb(100, 100, 255)"  # Light blue
                elseif alg == "PPO2"
                    "rgb(100, 255, 100)"  # Light green
                else  # DDPG
                    "rgb(255, 100, 255)"  # Light purple
                end
            else  # no_IL
                if alg == "SAC"
                    "rgb(200, 0, 0)"  # Dark red
                elseif alg == "PPO"
                    "rgb(0, 0, 200)"  # Dark blue
                elseif alg == "PPO2"
                    "rgb(0, 200, 0)"  # Dark green
                else  # DDPG
                    "rgb(200, 0, 200)"  # Dark purple
                end
            end
            
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
        boxmode="group",
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

