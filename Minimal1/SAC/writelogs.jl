




open("training_data.json", "w") do f
    JSON.print(f, logs)
    println(f)
end



open("run_data.json", "w") do f
    JSON.print(f, run_logs)
    println(f)
end