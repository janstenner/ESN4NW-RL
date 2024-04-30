using LinearAlgebra
using Flux
using FFTW
using PlotlyJS


function reward_function(grid_price, compute_power_used, power_for_free)
    power_for_free_used = min(power_for_free, compute_power_used)
    compute_power_used -= power_for_free
    compute_power_used = max(0.0, compute_power_used)

    reward1 = (50 * compute_power_used)^0.9 * ((grid_price + 0.2)^2) * 0.5 - 0.3 * compute_power_used * 70

    reward2 = - (37 * compute_power_used^1.2) * (1-grid_price*2)

    #factor = clamp(grid_price * 2 - 0.5, 0.0, 1.0)
    factor = sigmoid(grid_price * 9 - 4.0)
    #factor = sigmoid(grid_price * 20 - 13.0)
    factor = 1


    reward_free = (power_for_free_used * 40)^1.2 + (grid_price)^1.2 * power_for_free_used * 10

    reward = - (factor * reward1 + (1 - factor) * reward2) + reward_free

    reward
end


xx1 = collect(LinRange(0.0,1.0,200))
xx2 = collect(LinRange(0.0,0.01,200))


plots = PlotlyJS.SyncPlot[]

colorscale = [[0, "rgb(180, 0, 0)"], [0.4, "rgb(210, 160, 160)"], [0.5, "rgb(224, 224, 224)"], [0.6, "rgb(160, 160, 210)"], [1, "rgb(0, 0, 180)"], ]
layout = Layout(
        plot_bgcolor="#f1f3f7",
        coloraxis = attr(cmin = -0.5, cmid = 0.0, cmax = 0.5, colorscale = colorscale),
    )

for power_for_free in LinRange(0.0, 0.006, 3)
    result = zeros(200,200)

    for i in 1:200
        for j in 1:200
            result[i,j] = reward_function(xx1[i],xx2[j], power_for_free)
        end
    end

    push!(plots, plot(surface(x=xx1, y=xx2, z=result, coloraxis="coloraxis"), layout))
end

fig = [plots[1] plots[2] plots[3]]
relayout!(fig, layout.fields)
display(fig);