using LinearAlgebra
using IntervalSets
using StableRNGs
using SparseArrays
using Conda
using FFTW
using PlotlyJS
using FileIO, JLD2
using Flux
using Random
using RL
using DataFrames
using Statistics
using JuMP
using Ipopt
#using Blink

n_turbines = 1



# action vector dim - contains the percentage of maximum power the computer in the turbine will use for the duration of next time step

action_dim = n_turbines

# state vector

# - amount of computation left (starts at 1.0 and goes to 0.0)
# - wind stituation at every turbine (gradient of power output, current power output and curtailment enegry)
# - current gradient and price of energy from the grid
# - current time

state_dim = 4 + 3*n_turbines



# env parameters

seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false

te = 1440.0
dt = 5.0
t0 = 0.0
min_best_episode = 1

sim_space = Space(fill(0..1, (state_dim)))

function generate_wind()
    wind_constant_day = rand()
    deviation = 1/5

    result = sign(randn()) * sin.(collect(LinRange(rand()*3+1, 4+rand()*4, Int(te/dt)+1)))

    for i in 1:4
        result += sign(randn()) * sin.(collect(LinRange(rand()+4, 5+rand()*i*4, Int(te/dt)+1)))
    end

    result .-= minimum(result)
    result ./= maximum(result)
    result .*= deviation

    day_wind = sign(randn()) * sin.(collect(LinRange(wind_constant_day*2*pi, 2+wind_constant_day*2*pi, Int(te/dt)+1)))
    day_wind .+= 1.0
    day_wind ./= 4
    day_wind .+= 0.25


    result .+= day_wind

    clamp!(result, -1.0, 1.0)

    result
end

function generate_grid_price()

    factor = 1.0;
    factor = 0.6;

    gp = (-sin.(collect(LinRange(rand()*1.5*factor, 2+rand()*2.5*factor, Int(te/dt)+1))) .+(1+(rand()*factor)))

    clamp!(gp, -1, 1)

    return gp
end

y0 = [1.0]

wind = [generate_wind() for i in 1:n_turbines]

# layout = Layout(
#                 plot_bgcolor="#f1f3f7",
#                 yaxis=attr(range=[0,1]),
#             )

# to_plot = [scatter(y=wind[i]) for i in 1:1]
# plot(Vector{AbstractTrace}(to_plot), layout)

grid_price = generate_grid_price()
plot(scatter(y=grid_price), Layout(yaxis=attr(range=[0,1])))

for i in 1:n_turbines
    push!(y0, wind[i][2] - wind[i][1])
    push!(y0, wind[i][2])
    push!(y0, max(0.0, wind[i][2] - 0.4))
end

push!(y0, grid_price[2] - grid_price[1])
push!(y0, grid_price[2])

push!(y0, 0.0)

y0 = Float32.(y0)








function softplus_shifted(x)
    factor = 700
    log( 1 + exp(factor * (x - 0.006)) ) / factor
end




function do_step(env)
    global wind_only
    
    y = [ env.y[1] ]
    step = env.steps + 2

    compute_power = 0.0
    for i in 1:n_turbines
        compute_power += env.p[i]*0.01
    end

    # subtracting the computed load
    compute_power_used = min(y[1], compute_power)
    y[1] -= compute_power
    y[1] = max(y[1], 0.0)

    if y[1] == 0.0
        env.done = true
    end

    #normalizing
    compute_power_used *= 100/n_turbines

    # reward calculation
    
    power_for_free = 0.0
    for i in 1:n_turbines

        # curtailment energy onlny when wind is above 0.4
        temp_free_power = (wind[i][step-1] - 0.4)
        temp_free_power = max(0.0, temp_free_power)

        power_for_free += temp_free_power
    end
    power_for_free_used = min(power_for_free, compute_power_used)
    compute_power_used -= power_for_free
    # compute_power_used = max(0.0, compute_power_used)
    compute_power_used = softplus_shifted(compute_power_used)


    #normalizing
    compute_power_used *= (n_turbines * 0.01)
    

    reward1 = compute_power_used * grid_price[step-1]

    reward = - reward1

    if (env.time + env.dt) >= env.te 
        reward -= y[1] * 2
    else
        #reward shaping
        #reward = (-1) * abs((reward * 45))^2.2

        #delta_action punish
        # reward -= 0.002 * mean(abs.(env.delta_action))
        #clamp!(env.reward, -1.0, 0.0)
    end


    #env.reward = [ -(reward^2)]
    env.reward = [reward]
    

    
    for i in 1:n_turbines
        push!(y, wind[i][step] - wind[i][step-1])
        push!(y, wind[i][step])
        push!(y, max(0.0, wind[i][step] - 0.4))
    end

    push!(y, grid_price[step] - grid_price[step-1])
    push!(y, grid_price[step])

    push!(y, env.time / env.te)

    

    y = Float32.(y)

    return y
end
