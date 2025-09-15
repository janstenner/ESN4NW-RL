using JLD2
using FileIO

# File path for saving results
results_file = "training_results.jld2"

# Initialize or load results dictionary
if isfile(results_file)
    results = FileIO.load(results_file, "results")
    println("Loaded existing results file")
else
    results = Dict()
    println("Created new results dictionary")
    FileIO.save(results_file, "results", results)
end

# Run training multiple times
for i in 1:5
    println("Starting training run $i")
    
    # Set a new random seed for each run
    global seed = i
    
    # Include the training script
    include("SAC/Minimal1_SAC.jl")
    
    # Run training
    train()
    
    # Collect results using seed as key
    results[seed] = Dict(
        "agent_save" => agent_save,
        "agent" => agent,
        "rewards" => hook.rewards
    )
    
    # Save after each run in case of interruption
    FileIO.save(results_file, "results", results)
    println("Saved results for seed $seed")
end

println("All training runs completed. Results saved to $results_file")