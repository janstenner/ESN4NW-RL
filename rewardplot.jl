using PlotlyJS


function reward_function(grid_price, compute_power_used)

    reward1 = sqrt(50 * compute_power_used) * ((grid_price + 0.2)^2) * 0.5

    reward2 = - (37 * compute_power_used^1.2) * (1-grid_price*3)

    factor = clamp(grid_price * 2 - 0.5, 0.0, 1.0)
    reward = - (factor * reward1 + (1 - factor) * reward2)
end


xx1 = collect(LinRange(0.0,1.0,200))
xx2 = collect(LinRange(0.0,0.01,200))


result = zeros(200,200)

for i in 1:200
    for j in 1:200
        result[i,j] = reward_function(xx1[i],xx2[j])
    end
end

plot(surface(x=xx1, y=xx2, z=result))