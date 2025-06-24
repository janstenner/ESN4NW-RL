using Zygote:ignore

p = agent.policy

t = agent.trajectory



rng = p.rng
AC = p.approximator
γ = p.γ
λ = p.λ
n_epochs = p.n_epochs
n_microbatches = p.n_microbatches
clip_range = p.clip_range
w₁ = p.actor_loss_weight
w₂ = p.critic_loss_weight
w₃ = p.entropy_loss_weight
w₄ = p.critic_regularization_loss_weight
w₅ = p.logσ_regularization_loss_weight
D = RL.device(AC)
to_device(x) = send_to_device(D, x)

n_envs, n_rollout = size(t[:terminal])
@assert n_envs * n_rollout % n_microbatches == 0 "size mismatch"
microbatch_size = n_envs * n_rollout ÷ n_microbatches

n = length(t)
states = to_device(t[:state])


states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))

values = reshape(send_to_host(AC.critic(flatten_batch(states))), n_envs, :)
next_values = reshape(flatten_batch(t[:next_values]), n_envs, :)

advantages = generalized_advantage_estimation(
    t[:reward],
    values,
    next_values,
    γ,
    λ;
    dims=2,
    terminal=t[:terminal]
)
returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
advantages = to_device(advantages)

if p.normalize_advantage
    advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
end

actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)
explore_mod = to_device(t[:explore_mod])

stop_update = false

actor_losses = Float32[]
critic_losses = Float32[]
entropy_losses = Float32[]
logσ_regularization_losses = Float32[]
critic_regularization_losses = Float32[]

excitements = Float32[]
fears = Float32[]

rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))

i = 1

inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

# s = to_device(select_last_dim(states_flatten_on_host, inds))
# !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
a = to_device(collect(select_last_dim(actions_flatten, inds)))
exp_m = to_device(collect(select_last_dim(explore_mod, inds)))

if eltype(a) === Int
    a = CartesianIndex.(a, 1:length(a))
end

r = vec(returns)[inds]
log_p = vec(action_log_probs)[inds]
adv = vec(advantages)[inds]

clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-8 to avoid inf





# s_neg = sample_negatives_far(s)


actor = AC.actor
critic = AC.critic


v′ = critic(s) |> vec

μ, logσ = actor(s)


log_p′ₐ = vec(sum(normlogpdf(μ, exp.(logσ) .* exp_m, a), dims=1))


entropy_loss =
    mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2

ratio = exp.(log_p′ₐ .- log_p)

ignore() do
    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

    if approx_kl_div > p.target_kl
        println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
        stop_update = true
    end
end

excitement = ratio .* adv
fear = (abs.((ratio .- 1)) + ones(size(ratio))).^2 .* p.fear_factor

# if !(isempty(s_neg))
#     v_neg = critic(s_neg)
#     critic_regularization = mean((v_neg .- p.critic_target) .^ 2)
# else
#     critic_regularization = 0.0f0
# end

# if AC.actor.logσ_is_network
#     logσ_neg = AC.actor.logσ(s_neg)
#     logσ_regularization  = mean((logσ_neg .- log(AC.actor.max_σ)) .^ 2)	
# else
#     logσ_regularization  = 0.0f0
# end

surr1 = ratio .* adv
surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

actor_loss = -mean(min.(surr1, surr2) .* exp_m[:])

actor_loss = -mean(excitement - fear)
critic_loss = mean((r .- v′) .^ 2)
loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss #+ w₄ * critic_regularization #+ w₅ * logσ_regularization


ignore() do
    # println("---------------------")
    # println(critic_loss)
    # println(critic_regularization)
    # println("---------------------")
    push!(actor_losses, w₁ * actor_loss)
    push!(critic_losses, w₂ * critic_loss)
    push!(entropy_losses, -w₃ * entropy_loss)
    # push!(critic_regularization_losses, w₄ * critic_regularization)
    # push!(logσ_regularization_losses, w₅ * logσ_regularization)

    push!(excitements, mean(excitement))
    push!(fears, mean(fear))

    # polyak update critic target
    p.critic_target = p.critic_target * 0.9 + (maximum(r) - 0.1) * 0.1
end

loss


if !stop_update
    Flux.update!(AC.actor_state_tree, AC.actor.μ, g_actor.μ)
    Flux.update!(AC.sigma_state_tree, AC.actor.logσ, g_actor.logσ)
    Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
else
    break
end


