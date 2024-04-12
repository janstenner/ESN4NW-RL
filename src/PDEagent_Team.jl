using ReinforcementLearning
using Flux
using StableRNGs
using IntervalSets
using Setfield: @set
using Zygote: ignore
using CUDA

include(pwd() * "/src/custom_nna.jl")

export create_agent, DDPGTeamPolicy


function create_NNA(;na, ns, use_gpu, is_actor, init, copyfrom = nothing, nna_scale, drop_middle_layer, learning_rate = 0.001, fun = relu, num_actuators = 0, critic_window = 1)
    nna_size_actor = Int(floor(10 * nna_scale))
    nna_size_critic = Int(floor(20 * nna_scale))

    if is_actor
        if drop_middle_layer
            n = Chain(
                Dense(ns, nna_size_actor, fun; init = init),
                Dense(nna_size_actor, na, tanh; init = init),
            )
        else
            n = Chain(
                Dense(ns, nna_size_actor, fun; init = init),
                Dense(nna_size_actor, nna_size_actor, fun; init = init),
                Dense(nna_size_actor, na, tanh; init = init),
            )
        end
    else
        window_half_size = Int(floor(critic_window/2))
        if drop_middle_layer
            n = Chain(
                Dense(num_actuators + (ns + na) * (1 + 2*window_half_size), nna_size_critic, fun; init = init),
                Dense(nna_size_critic, 1; init = init),
            )
        else
            n = Chain(
                Dense(num_actuators + (ns + na) * (1 + 2*window_half_size), nna_size_critic, fun; init = init),
                Dense(nna_size_critic, nna_size_critic, fun; init = init),
                Dense(nna_size_critic, 1; init = init),
            )
        end
    end

    nna = CustomNeuralNetworkApproximator(
        model = use_gpu ? n |> gpu : n,
        optimizer = Flux.ADAM(learning_rate),
    )

    if copyfrom !== nothing
        copyto!(nna, copyfrom) 
    end

    nna
end


function create_agent(;num_actuators, critic_window, critic_window_indexes, action_space, state_space, use_gpu,
                    rng, y, p, batch_size,
                    start_steps, start_policy, update_after, update_freq, update_loops = 1, reset_stage = POST_EPISODE_STAGE, act_limit,
                    act_noise, noise_hold = 1,
                    nna_scale = 1, nna_scale_critic = nothing, drop_middle_layer = false, drop_middle_layer_critic = nothing, memory_size = 0, trajectory_length = 1000, learning_rate = 0.001, learning_rate_critic = nothing, fun = relu, fun_critic = nothing)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    isnothing(drop_middle_layer_critic) &&  (drop_middle_layer_critic = drop_middle_layer)
    isnothing(fun_critic)               &&  (fun_critic = fun)
    isnothing(learning_rate_critic)     &&  (learning_rate_critic = learning_rate)
    
    init = Flux.glorot_uniform(rng)

    actors = []
    
    for i in 1:num_actuators
        temp_dict = Dict()

        temp_dict["behavior"] = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, learning_rate = learning_rate, fun = fun)

        temp_dict["target"] = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, learning_rate = learning_rate, fun = fun)

        copyto!(temp_dict["behavior"], temp_dict["target"])  # force sync

        push!(actors, temp_dict)
    end

    behavior_critic = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, learning_rate = learning_rate_critic, fun = fun_critic, num_actuators = num_actuators, critic_window = critic_window)

    target_critic = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale_critic, drop_middle_layer = drop_middle_layer_critic, learning_rate = learning_rate_critic, fun = fun_critic, num_actuators = num_actuators, critic_window = critic_window)

    copyto!(behavior_critic, target_critic)  # force sync

    
    reward_size = (1, num_actuators)

    actions = zeros(size(action_space))
    last_noise = randn(rng, size(actions[1:end-memory_size,:])) .* act_noise
    
    Agent(
        policy = DDPGTeamPolicy(
            action_space = action_space,
            state_space = state_space,

            critic_window = critic_window,
            critic_window_indexes = critic_window_indexes,

            rng = rng,

            actors = actors,
            target_critic = target_critic,
            behavior_critic = behavior_critic,

            use_gpu = use_gpu,

            y = y,
            p = p,
            batch_size = batch_size,
            start_steps = start_steps,
            start_policy = start_policy,
            update_after = update_after,
            update_freq = update_freq,
            update_loops = update_loops,
            reset_stage = reset_stage,
            act_limit = act_limit,
            act_noise = act_noise,
            noise_hold = noise_hold,
            last_noise = last_noise,
            memory_size = memory_size,
        ),
        trajectory = 
            CircularArrayTrajectory(; capacity = trajectory_length, state = Float32 => size(state_space), action = Float32 => size(action_space), reward = Float32 => reward_size, terminal = Bool => ()),
    )
end

Base.@kwdef mutable struct DDPGTeamPolicy{
    BC<:CustomNeuralNetworkApproximator,
    TC<:CustomNeuralNetworkApproximator,
    P,
    R,
} <: AbstractPolicy

    action_space::Space
    state_space::Space

    critic_window
    critic_window_indexes

    rng::R

    actors

    behavior_critic::BC
    target_critic::TC

    use_gpu::Bool

    y
    p
    batch_size
    start_steps
    start_policy::P
    update_after
    update_freq
    update_loops
    reset_stage
    act_limit
    act_noise
    noise_hold
    last_noise
    memory_size

    update_step::Int = 0
    actor_loss::Float32 = 0.0f0
    critic_loss::Float32 = 0.0f0
end

# Flux.functor(x::DDPGTeamPolicy) = (
#     ba = x.behavior_actor,
#     bc = x.behavior_critic,
#     ta = x.target_actor,
#     tc = x.target_critic,
# ),
# y -> begin
#     x = @set x.behavior_actor = y.ba
#     x = @set x.behavior_critic = y.bc
#     x = @set x.target_actor = y.ta
#     x = @set x.target_critic = y.tc
#     x
# end


function (policy::DDPGTeamPolicy)(env; learning = true)
    if learning
        policy.update_step += 1
    end

    if policy.update_step <= policy.start_steps
        policy.start_policy(env)
    else
        D = device(policy.actors[1]["behavior"])
        s = DynamicStyle(env) == SEQUENTIAL ? state(env) : state(env, player)

        s = send_to_device(D, s)
        
        
        actions = zeros(size(policy.action_space))

        for i in eachindex(actions)
            actions[i] = policy.actors[i]["behavior"](s[:,i])[1]
        end
        

        actions = actions |> send_to_host

        if learning
            if policy.update_step % policy.noise_hold == 0
                policy.last_noise = randn(policy.rng, size(actions[1:end-policy.memory_size,:])) .* policy.act_noise
            end
            actions[1:end-policy.memory_size,:] += policy.last_noise
            actions = clamp.(actions, -policy.act_limit, policy.act_limit)
        else
            actions = clamp.(actions, -policy.act_limit, policy.act_limit)
        end

        actions
    end
end



function (policy::DDPGTeamPolicy)(stage::AbstractStage, env::AbstractEnv)
    nothing
end

function RLBase.update!(
    policy::DDPGTeamPolicy,
    traj::Trajectory,
    ::AbstractEnv,
    stage::PostEpisodeStage,
)
    if stage == policy.reset_stage
        policy.update_step = 0
    end
end

function RLBase.update!(
    policy::DDPGTeamPolicy,
    traj::Trajectory,
    ::AbstractEnv,
    stage::PostExperimentStage,
)
    if stage == policy.reset_stage
        policy.update_step = 0
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    p::DDPGTeamPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::DDPGTeamPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)

    push!(trajectory[:state], s)
    push!(trajectory[:action], action)
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::DDPGTeamPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = policy isa NamedPolicy ? reward(env, nameof(policy)) : reward(env)

    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::DDPGTeamPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    
end

#custom sample function
function pde_sample(rng::AbstractRNG, t::AbstractTrajectory, s::BatchSampler)
    inds = rand(rng, 1:length(t), s.batch_size)
    pde_fetch!(s, t, inds)
    inds, s.cache
end

function pde_fetch!(s::BatchSampler, t::Trajectory, inds::Vector{Int})

    batch = NamedTuple{SARTS}((
        (consecutive_view(t[x], inds) for x in SART)...,
        consecutive_view(t[:state], inds .+ 1),
    ))

    
    if isnothing(s.cache)
        s.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(s.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
end

function RLBase.update!(
    policy::DDPGTeamPolicy,
    traj::Trajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    length(traj) > policy.update_after || return
    policy.update_step % policy.update_freq == 0 || return
    #println("UPDATE!")
    for i = 1:policy.update_loops
        inds, batch = pde_sample(policy.rng, traj, BatchSampler{SARTS}(policy.batch_size))
        update!(policy, batch)
    end
end

function RLBase.update!(policy::DDPGTeamPolicy, batch::NamedTuple{SARTS})
    
    s, a, r, t, snext = batch

    # s has dimensions state_size x num_actuators x batch_size
    # a and r have dimensions 1 x num_actuators x batch_size
    # t has dimensions batch_size

    C = policy.behavior_critic
    Cₜ = policy.target_critic

    y = policy.y
    p = policy.p

    s, a, r, t, snext = send_to_device(device(policy.behavior_critic), (s, a, r, t, snext))

    num_actuators = size(s)[2]

    for i in 1:num_actuators

        # one-hot position vector in dimensions num_actuators x batch_size
        one_hot = zeros(num_actuators, size(s)[3])
        one_hot[i,:] .= 1.0
        one_hot = send_to_device(device(policy.behavior_critic), one_hot)

        temp_indexes = policy.critic_window_indexes(i)

        c_input = one_hot

        for ind in temp_indexes
            anext = policy.actors[ind]["target"](snext[:,ind,:])
            c_input = vcat(c_input, snext[:,ind,:])
            c_input = vcat(c_input, anext)
        end
        
        
        qₜ = Cₜ(c_input) |> vec
        qnext = r[:,i,:] .+ y .* (1 .- t) .* qₜ
        #a = Flux.unsqueeze(a, ndims(a)+1)

        gs1 = gradient(Flux.params(C)) do
            c_input = one_hot

            for ind in temp_indexes
                c_input = vcat(c_input, s[:,ind,:])
                c_input = vcat(c_input, a[:,ind,:])
            end

            q = C(c_input) |> vec
            loss = mean((qnext .- q) .^ 2)
            ignore() do
                policy.critic_loss = loss
            end
            loss
        end

        update!(C, gs1)

        gs2 = gradient(Flux.params(policy.actors[i]["behavior"])) do
            c_input = one_hot

            for ind in temp_indexes
                c_input = vcat(c_input, s[:,ind,:])
                if ind == i
                    c_input = vcat(c_input, policy.actors[i]["behavior"](s[:,ind,:]))
                else
                    c_input = vcat(c_input, a[:,ind,:])
                end
            end

            loss = -mean(C(c_input))
            ignore() do
                policy.actor_loss = loss
            end
            loss
        end
    
        update!(policy.actors[i]["behavior"], gs2)
    end

    # polyak averaging
    for (dest, src) in zip(Flux.params(Cₜ), Flux.params(C))
        dest .= p .* dest .+ (1 - p) .* src
    end

    for i in 1:num_actuators
        for (dest, src) in zip(Flux.params(policy.actors[i]["target"]), Flux.params(policy.actors[i]["behavior"]))
            dest .= p .* dest .+ (1 - p) .* src
        end
    end

    # memory leak fix
    GC.gc()
    CUDA.reclaim()
end

struct ZeroPolicy <: AbstractPolicy
    action_space
end

(p::ZeroPolicy)(env) = zeros(size(p.action_space))
