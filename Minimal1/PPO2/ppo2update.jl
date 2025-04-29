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

actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)

stop_update = false

actor_losses = Float32[]
critic_losses = Float32[]
entropy_losses = Float32[]
logσ_regularization_losses = Float32[]
critic_regularization_losses = Float32[]

excitements = Float32[]
fears = Float32[]

for epoch in 1:n_epochs

    rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
    for i in 1:n_microbatches

        inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

        # s = to_device(select_last_dim(states_flatten_on_host, inds))
        # !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
        s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
        a = to_device(collect(select_last_dim(actions_flatten, inds)))

        if eltype(a) === Int
            a = CartesianIndex.(a, 1:length(a))
        end

        r = vec(returns)[inds]
        log_p = vec(action_log_probs)[inds]
        adv = vec(advantages)[inds]

        clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-8 to avoid inf

        if p.normalize_advantage
            adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
        end

        if isnothing(AC.actor_state_tree)
            AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor.μ)
        end

        if isnothing(AC.sigma_state_tree)
            AC.sigma_state_tree = Flux.setup(AC.optimizer_sigma, AC.actor.logσ)
        end

        if isnothing(AC.critic_state_tree)
            AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
        end

        s_neg = sample_negatives_far(s)

        global g_actor

        g_actor, g_critic = Flux.gradient(AC.actor, AC.critic) do actor, critic
            v′ = critic(s) |> vec
            if actor isa GaussianNetwork
                μ, logσ = actor(s)
                
                if ndims(a) == 2
                    log_p′ₐ = vec(sum(normlogpdf(μ, exp.(logσ), a), dims=1))
                else
                    log_p′ₐ = normlogpdf(μ, exp.(logσ), a)
                end
                entropy_loss =
                    mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2
            else
                # actor is assumed to return discrete logits
                logit′ = actor(s)

                p′ = softmax(logit′)
                log_p′ = logsoftmax(logit′)
                log_p′ₐ = log_p′[a]
                entropy_loss = -sum(p′ .* log_p′) * 1 // size(p′, 2)
            end
            ratio = exp.(log_p′ₐ .- log_p)

            ignore() do
                approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

                if approx_kl_div > p.target_kl
                    println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                    stop_update = true
                end
            end

            excitement = ratio .* adv
            fear = abs.((ratio .- 1)) .* p.fear_factor

            if !(isempty(s_neg))
                v_neg = critic(s_neg)
                critic_regularization = mean((v_neg .- p.critic_target) .^ 2)
            else
                critic_regularization = 0.0f0
            end

            if AC.actor.logσ_is_network
                logσ_neg = AC.actor.logσ(s_neg)
                logσ_regularization  = mean((logσ_neg .- log(AC.actor.max_σ)) .^ 2)	
            else
                logσ_regularization  = 0.0f0
            end

            actor_loss = -mean((excitement - fear) .^2)
            critic_loss = mean((r .- v′) .^ 2)
            loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss + w₄ * critic_regularization + w₅ * logσ_regularization


            ignore() do
                # println("---------------------")
                # println(critic_loss)
                # println(critic_regularization)
                # println("---------------------")
                push!(actor_losses, w₁ * actor_loss)
                push!(critic_losses, w₂ * critic_loss)
                push!(entropy_losses, -w₃ * entropy_loss)
                push!(critic_regularization_losses, w₄ * critic_regularization)
                push!(logσ_regularization_losses, w₅ * logσ_regularization)

                push!(excitements, mean(excitement))
                push!(fears, mean(fear))

                # polyak update critic target
                p.critic_target = p.critic_target * 0.9 + minimum(r) * 0.1
            end

            loss
        end
        
        if !stop_update
            Flux.update!(AC.actor_state_tree, AC.actor.μ, g_actor.μ)
            Flux.update!(AC.sigma_state_tree, AC.actor.logσ, g_actor.logσ)
            Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
        else
            break
        end

    end

    if stop_update
        break
    end
end


if p.adaptive_weights
    # everything here is just magnitude (abs), not real mean

    mean_actor_loss = mean(abs.(actor_losses))
    mean_critic_loss = mean(abs.(critic_losses))
    mean_entropy_loss = mean(abs.(entropy_losses))
    mean_logσ_regularization_loss = mean(abs.(logσ_regularization_losses))
    mean_critic_regularization_loss = mean(abs.(critic_regularization_losses))
    
    println("---")
    println("mean actor loss: $(mean_actor_loss)")
    println("mean critic loss: $(mean_critic_loss)")
    println("mean entropy loss: $(mean_entropy_loss)")
    println("mean logσ regularization loss: $(mean_logσ_regularization_loss)")
    println("mean critic regularization loss: $(mean_critic_regularization_loss)")

    mean_excitement = mean(abs.(excitements))
    mean_fear = mean(abs.(fears))
    
    println("mean excitement: $(mean_excitement)")
    println("mean fear: $(mean_fear)")


    actor_factor = clamp(1.0/mean_actor_loss, 0.9, 1.1)
    critic_factor = clamp(0.5/mean_critic_loss, 0.9, 1.1)
    entropy_factor = clamp(0.01/mean_entropy_loss, 0.9, 1.1)
    logσ_regularization_factor = clamp(0.1/mean_logσ_regularization_loss, 0.9, 1.1)
    critic_regularization_factor = clamp(0.03/mean_critic_regularization_loss, 0.9, 1.1)

    fear_factor_factor = clamp(((mean_excitement * 0.5) / (mean_fear)), 0.1, 1.01)

    println("changing actor weight from $(w₁) to $(w₁*actor_factor)")
    println("changing critic weight from $(w₂) to $(w₂*critic_factor)")
    println("changing entropy weight from $(w₃) to $(w₃*entropy_factor)")
    println("changing logσ regularization weight from $(w₅) to $(w₅*logσ_regularization_factor)")
    println("changing critic regularization weight from $(w₄) to $(w₄*critic_regularization_factor)")
    println("changing fear factor from $(p.fear_factor) to $(p.fear_factor*fear_factor_factor)")

    println("current critic_target is $(p.critic_target)")

    p.actor_loss_weight = w₁ * actor_factor
    p.critic_loss_weight = w₂ * critic_factor
    p.entropy_loss_weight = w₃ * entropy_factor
    p.logσ_regularization_loss_weight = w₅ * logσ_regularization_factor
    p.critic_regularization_loss_weight = w₄ * critic_regularization_factor

    p.fear_factor = p.fear_factor * fear_factor_factor
end

println("---")
