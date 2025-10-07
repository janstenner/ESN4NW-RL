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
clip_range_vf = p.clip_range_vf
w₁ = p.actor_loss_weight
w₂ = p.critic_loss_weight
w₃ = p.entropy_loss_weight
D = RL.device(AC)
to_device(x) = send_to_device(D, x)

n_envs, n_rollout = size(t[:terminal])

microbatch_size = Int(floor(n_envs * n_rollout ÷ n_microbatches))
actorbatch_size = p.actorbatch_size

n = length(t)
states = to_device(t[:state])
#next_states = to_device(t[:next_state])
actions = to_device(t[:action])

states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))
#next_states_flatten_on_host = flatten_batch(select_last_dim(t[:next_state], 1:n))

values = reshape(send_to_host(AC.critic(flatten_batch(states))), n_envs, :)

#values = prepare_values(values, t[:terminal])

#mus = AC.actor.μ(states_flatten_on_host)
#offsets = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), mus) )) , n_envs, :)

# advantages = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), flatten_batch(actions)) )) , n_envs, :) - offsets

critic2_input = p.critic2_takes_action ? vcat(flatten_batch(states), flatten_batch(actions)) : flatten_batch(states)

critic2_values = reshape(send_to_host( AC.critic2( critic2_input ) ) , n_envs, :)

rewards = collect(to_device(t[:reward]))
terminal = collect(to_device(t[:terminal]))

gae_deltas = rewards .+ critic2_values .* (1 .- terminal) .- values

#@show size(gae_deltas)
#error("abb")


advantages, returns = generalized_advantage_estimation(
    gae_deltas,
    zeros(Float32, size(gae_deltas)),
    zeros(Float32, size(gae_deltas)),
    γ,
    λ;
    dims=2,
    terminal=t[:terminal]
)

# returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
advantages = to_device(advantages)

# if p.normalize_advantage
#     advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
# end

positive_advantage_indices = findall(>(0), vec(advantages))


actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)
explore_mod = to_device(t[:explore_mod])

stop_update = false

actor_losses = Float32[]
critic_losses = Float32[]
critic2_losses = Float32[]
entropy_losses = Float32[]





if isnothing(AC.actor_state_tree) || isnothing(AC.sigma_state_tree) || isnothing(AC.critic_state_tree) || isnothing(AC.critic2_state_tree)
    println("________________________________________________________________________")
    println("Reset Optimizers")
    println("________________________________________________________________________")
    AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor.μ)
    AC.sigma_state_tree = Flux.setup(AC.optimizer_sigma, AC.actor.logσ)
    AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
    AC.critic2_state_tree = Flux.setup(AC.optimizer_critic2, AC.critic2)
end



next_states = to_device(flatten_batch(t[:next_state]))


v_ref = AC.critic_frozen( flatten_batch(states) )[:] 

q_ref = AC.critic2_frozen( critic2_input )[:] 


next_values = reshape( AC.critic( next_states ), n_envs, :)

dones = terminal

Gs = [nstep_targets(rewards, dones, next_values, γ; n=k) for k in 1:3]
Gs2 = [nstep_targets_ppo3(rewards, dones, next_values, γ; n=k) for k in 1:3]

targets = lambda_truncated_targets(rewards, terminal, next_values, γ)[:]
#targets = nstep_targets(rewards, terminal, next_values, γ; n=1)[:]

#targets_critic2 = lambda_truncated_targets_ppo3(rewards, terminal, next_values, γ; n=1)[:]
#targets_critic2 = nstep_targets_ppo3(rewards, terminal, next_values, γ; n=1)[:]

#critic2_input = p.critic2_takes_action ? vcat(next_states, AC.actor.μ(next_states)) : next_states
#next_critic2_values = reshape( AC.critic2( critic2_input ), n_envs, :)
#targets_critic2 = lambda_truncated_targets_ppo3(rewards, terminal, next_values, γ)[:]
targets_critic2 = RL.special_targets_ppo3(rewards, terminal, critic2_values, next_values, γ)[:]


plot([
    scatter(y=targets, name="targets", mode="lines"),
    scatter(y=targets_critic2, name="targets_critic2", mode="lines"),
    scatter(y=targets_critic2 + rewards[:], name="targets_critic2 + rewards", mode="lines"),
    scatter(y=targets_critic2 .- targets, name="targets substracted", mode="lines"),
    scatter(y=Float32.(dones)[:], name="dones", mode="lines"),
    scatter(y=Gs[1][:], name="Gs[1]", mode="lines"),
    scatter(y=Gs[2][:], name="Gs[2]", mode="lines"),
    scatter(y=Gs[3][:], name="Gs[3]", mode="lines"),
    scatter(y=rewards[:], name="rewards", mode="lines"),
])



collector = BatchQuantileCollector()


rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))