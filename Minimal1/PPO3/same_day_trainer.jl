using Zygote:ignore


function train_same_day(n = 100; n_microbatches = 5, update_actor = true, update_critic = true)

    agent.policy.update_step = 0

    render_run(; new_day = false, exploration = true)

    for i in 1:n

        agent.policy.update_step += agent.policy.update_freq

        update_complete!(; n_microbatches = n_microbatches, update_actor = update_actor, update_critic = update_critic)

        render_run(; new_day = false, exploration = true)
    end
end



function update_complete!(; n_microbatches = 5, update_actor = true, update_critic = true)
    p = agent.policy
    t = day_trajectory



    rng = p.rng
    AC = p.approximator
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    #n_microbatches = p.n_microbatches
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

    #gae_deltas = rewards .+ critic2_values .* (1 .- terminal) .- values
    gae_deltas = critic2_values

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
    #targets = lambda_truncated_targets(rewards, terminal, next_values, γ)[:]
    targets = nstep_targets(rewards, terminal, next_values, γ)[:]
    nstep_targets

    #targets for critic2 now below


    collector = BatchQuantileCollector()
    

    

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        #rand_inds_actor = shuffle!(rng, Vector(1:n_envs*n_rollout-actorbatch_size))

        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            #inds_actor = collect(rand_inds_actor[i]:rand_inds_actor[i]+actorbatch_size-1)
            inds_actor = inds[1:clamp(actorbatch_size, 1, length(inds))]

            #inds = positive_advantage_indices

            # s = to_device(select_last_dim(states_flatten_on_host, inds))
            # !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            s_actor = to_device(collect(select_last_dim(states_flatten_on_host, inds_actor)))
            a = to_device(collect(select_last_dim(actions_flatten, inds_actor)))
            exp_m = to_device(collect(select_last_dim(explore_mod, inds_actor)))

            if eltype(a) === Int
                a = CartesianIndex.(a, 1:length(a))
            end

            #r = vec(returns)[inds]
            log_p = vec(action_log_probs)[inds_actor]
            adv = vec(advantages)[inds_actor]

            tar = vec(targets)[inds]

            old_v = vec(values)[inds]

            

            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-8 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end

            # s_neg = sample_negatives_far(s)

            g_actor, g_critic = Flux.gradient(AC.actor, AC.critic) do actor, critic
                v′ = critic(s) |> vec

                # nv′ = AC.critic(ns) |> vec
                # nv = critic2(vcat(s,a)) |> vec

                μ, logσ = actor(s_actor)

                σ = (exp.(logσ) .* exp_m) #.+ (exp_m .* 0.25)

                if ndims(a) == 2
                    log_p′ₐ = vec(sum(normlogpdf(μ, σ, a), dims=1))
                else
                    log_p′ₐ = normlogpdf(μ, σ, a)
                end

                #clamp!(log_p′ₐ, log(1e-8), Inf)

                entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2
                
                ratio = exp.(log_p′ₐ .- log_p)

                ignore() do
                    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

                    if approx_kl_div > p.target_kl
                        println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                        stop_update = true
                    end
                end

                
                fear = (ratio .- 1).^2 .* p.fear_factor


                if p.new_loss
                    actor_loss_values = ((ratio .* adv) - fear)  .* exp_m[:]
                    actor_loss = -mean(actor_loss_values)
                else
                    surr1 = ratio .* adv
                    surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))
                end


                if isnothing(clip_range_vf) || clip_range_vf == 0.0
                    values_pred = v′
                else
                    # clipped value function loss, from OpenAI SpinningUp implementation
                    Δ = v′ .- old_v
                    values_pred = old_v .+ clamp.(Δ, -clip_range_vf, clip_range_vf)
                end



                
                bellman = mean(((tar .- values_pred) .^ 2))
                fr_term = mean((values_pred .- v_ref[inds]) .^ 2)
                critic_loss = bellman + 0.4 * fr_term # .* exp_m[:])


                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss 


                ignore() do
                    push!(actor_losses, w₁ * actor_loss)
                    push!(critic_losses, w₂ * critic_loss)
                    push!(entropy_losses, -w₃ * entropy_loss)

                    RL.update!(collector, ratio, adv; p=0.9, within_batch_weighted=true)
                end

                loss
            end
            
            if !stop_update
                if (p.update_step / p.update_freq) % p.actor_update_freq == 0
                    if update_actor
                        Flux.update!(AC.actor_state_tree, AC.actor.μ, g_actor.μ)
                        Flux.update!(AC.sigma_state_tree, AC.actor.logσ, g_actor.logσ)
                    end
                end
                if update_critic
                    Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
                end
            else
                break
            end

            #update!(AC.critic.layers[end], tar) 

        end
    end

    #critic2_input = p.critic2_takes_action ? vcat(next_states, AC.actor.μ(next_states)) : next_states
    #next_critic2_values = reshape( AC.critic2( critic2_input ), n_envs, :)
    #targets_critic2 = lambda_truncated_targets_ppo3(rewards, terminal, next_values, γ)[:]
    values = reshape( AC.critic( states ), n_envs, :)
    next_values = reshape( AC.critic( next_states ), n_envs, :)
    targets_critic2 = special_targets_ppo3(rewards, terminal, values, next_values, γ)[:]

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))

        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            a = to_device(collect(select_last_dim(actions_flatten, inds)))

            # nv′ = vec(next_values)[inds]
            # rew = vec(rewards)[inds]
            # ter = vec(terminal)[inds]

            #tar = rew + γ * nv′ .* (1 .- ter)
            tar = vec(targets_critic2)[inds]

            old_v2 = vec(critic2_values)[inds]

            critic2_input = p.critic2_takes_action ? vcat(s, a) : s

            g_critic2 = Flux.gradient(AC.critic2) do critic2
                v2′ = critic2(critic2_input) |> vec

                if isnothing(clip_range_vf) || clip_range_vf == 0.0
                    values_pred2 = v2′
                else
                    # clipped value function loss, from OpenAI SpinningUp implementation
                    Δ = v2′ .- old_v2
                    values_pred2 = old_v2 .+ clamp.(Δ, -clip_range_vf, clip_range_vf)
                end


                bellman = mean(((tar .- values_pred2) .^ 2))
                fr_term = mean((values_pred2 .- q_ref[inds]) .^ 2)
                critic2_loss = bellman + 0.4 * fr_term # .* exp_m[:]


                ignore() do
                    push!(critic2_losses, w₂ * critic2_loss)
                end

                loss = w₂ * critic2_loss

                loss
            end

            if update_critic
                Flux.update!(AC.critic2_state_tree, AC.critic2, g_critic2[1])
            end

            #update!(AC.critic2.layers[end], tar) 

        end
    end


    #println(p.update_step / p.update_freq)

    if (p.update_step / p.update_freq) % p.critic_frozen_update_freq == 0
        if update_critic
            println("CRITIC FROZEN UPDATE")
            AC.critic_frozen = deepcopy(AC.critic)
            AC.critic2_frozen = deepcopy(AC.critic2)
        end
    end


    # everything here is just magnitude (abs), not real mean

    mean_actor_loss = mean(abs.(actor_losses))
    mean_critic_loss = mean(abs.(critic_losses))
    mean_critic2_loss = mean(abs.(critic2_losses))
    mean_entropy_loss = mean(abs.(entropy_losses))
    # mean_logσ_regularization_loss = mean(abs.(logσ_regularization_losses))
    # mean_critic_regularization_loss = mean(abs.(critic_regularization_losses))
    
    println("---")
    println("mean actor loss: $(mean_actor_loss)")
    println("mean critic loss: $(mean_critic_loss)")
    println("mean critic2 loss: $(mean_critic2_loss)")
    println("mean entropy loss: $(mean_entropy_loss)")
    # println("mean logσ regularization loss: $(mean_logσ_regularization_loss)")
    # println("mean critic regularization loss: $(mean_critic_regularization_loss)")

    q = RL.finalize(collector; p_over_epochs=0.9, weighted=true)

    # q.q_eps   : 0.9-Quantil der (pro Batch) 0.9-Quantile von |r-1|
    # q.q_adv   : 0.9-Quantil der (pro Batch) 0.9-Quantile von |A|
    # q.wq_eps  : A-gewichtetes 0.9-Quantil über die Batch-Quantile von |r-1|

    
    println("0.9-Quantil excitement: $(q.q_adv)")
    println("weighted 0.9-Quantil |r-1|: $(q.wq_eps)")

    if update_actor
        if p.adaptive_weights && (p.update_step / p.update_freq) % 4 == 0

            old_fear_factor = deepcopy(p.fear_factor)

            A_ref = q.q_adv                          # robustes |A|-Quantil
            eps_meas = q.wq_eps
            eps_target = 0.1                         # Zielwert für |r-1|

            # Guardrails / Fallbacks
            if !isfinite(eps_meas) || eps_meas <= 0
                eps_meas = eps_target                # Startfall oder Degenerat
            end
            if !isfinite(A_ref) || A_ref <= 0
                A_ref = 1.0                          # konservativer Fallback
            end

            # Baseline λ* aus Größenordnungs-Match nahe r≈1
            λ_star = 0.35 * (A_ref / eps_target)

            λ_prev = p.fear_factor
            gamma = 1.0
            beta = 0.9

            lambda_min = 1e-3
            lambda_max = 1e2

            # Regler-Update
            factor = (eps_meas / eps_target)^gamma
            λ_raw = (1 - beta) * λ_prev + beta * λ_star * factor

            # Clamping & Sanity
            λ_next = clamp(λ_raw, lambda_min, lambda_max)

            # polyak update fear_factor
            p.fear_factor = λ_next


            # println("changing actor weight from $(w₁) to $(w₁*actor_factor)")
            # println("changing critic weight from $(w₂) to $(w₂*critic_factor)")
            # println("changing entropy weight from $(w₃) to $(w₃*entropy_factor)")
            # println("changing logσ regularization weight from $(w₅) to $(w₅*logσ_regularization_factor)")
            # println("changing critic regularization weight from $(w₄) to $(w₄*critic_regularization_factor)")
            println("changing fear factor from $(old_fear_factor) to $(λ_next)")

            # println("current critic_target is $(p.critic_target)")

            # p.actor_loss_weight = w₁ * actor_factor
            # p.critic_loss_weight = w₂ * critic_factor
            # p.entropy_loss_weight = w₃ * entropy_factor
            # p.logσ_regularization_loss_weight = w₅ * logσ_regularization_factor
            # p.critic_regularization_loss_weight = w₄ * critic_regularization_factor

            
        end
    end

    println("---")
end


function trajectory_analysis(n = 100)
    
    global multiple_day_trajectory = CircularArrayTrajectory(;
                capacity = 288 * n,
                state = Float32 => (size(env.state_space)[1], 1),
                action = Float32 => (size(env.action_space)[1], 1),
                action_log_prob = Float32 => (1),
                reward = Float32 => (1),
                explore_mod = Float32 => (1),
                terminal = Bool => (1,),
                next_state = Float32 => (size(env.state_space)[1], 1),
        )

    for i in 1:n
        render_run(;
            new_day = false,
            exploration = true,
            return_plot = true, # we dont need to display the plot
            )

        for j in 1:length(day_trajectory)
            push!(multiple_day_trajectory[:state], day_trajectory[:state][:,:,j])
            push!(multiple_day_trajectory[:action], day_trajectory[:action][:,:,j])
            push!(multiple_day_trajectory[:action_log_prob], day_trajectory[:action_log_prob][:,j])
            push!(multiple_day_trajectory[:reward], day_trajectory[:reward][:,j])
            push!(multiple_day_trajectory[:explore_mod], day_trajectory[:explore_mod][:,j])
            push!(multiple_day_trajectory[:terminal], day_trajectory[:terminal][:,j])
            push!(multiple_day_trajectory[:next_state], day_trajectory[:next_state][:,:,j])
        end
    end

    xx = collect(dt/60:dt/60:te/60)

    γ = agent.policy.γ
    rewards = collect(multiple_day_trajectory[:reward])
    terminal = collect(multiple_day_trajectory[:terminal])
    next_states = flatten_batch(multiple_day_trajectory[:next_state])
    next_values = reshape( agent.policy.approximator.critic( next_states ), 1, :)

    # calculate real returns for each state
    returns = zeros(Float32, length(multiple_day_trajectory))
    for i in length(multiple_day_trajectory):-1:1
        if terminal[i]
            returns[i] = rewards[i]
        else
            returns[i] = rewards[i] + γ * returns[i+1]
        end
    end

    # calculate targets for critic
    targets = lambda_truncated_targets(rewards, terminal, next_values, γ)[:]



    # Process states and count visits
    states_array = multiple_day_trajectory[:state]
    n_states = size(states_array, 3)  # number of states recorded

    # Create bins for load_left
    min_load = minimum(states_array[1, 1, :])
    load_bins = collect(LinRange(min_load,1,200))
    n_load_bins = length(load_bins)
    time_bins = 1:288  # Time steps are already discrete 1-288

    # Initialize visitation matrix
    state_visits = zeros(Int, n_load_bins-1, 288)  # -1 because we're counting intervals between bin edges

    # For each state, increment the appropriate cell in the visitation matrix
    for i in 1:n_states
        load_left = states_array[1, 1, i]  # state[1] - load_left value
        time_step = (i-1) % 288 + 1        # time step (1-288)
        
        # Find the appropriate load_left bin
        load_bin = searchsortedfirst(load_bins, load_left) - 1
        load_bin = clamp(load_bin, 1, n_load_bins-1)
        
        # Increment the visitation count
        state_visits[load_bin, time_step] += 1
    end

    # Create heatmap of state visits
    colorscale2 = [[0.0, "rgb(5, 0, 5)"], [0.01, "rgb(40, 0, 60)"], [0.3, "rgb(160, 0, 200)"], [0.75, "rgb(210, 0, 255)"], [1.0, "rgb(240, 160, 255)"]]
    layout = Layout(
        title = "State Visitation Heatmap",
        xaxis_title = "Time Step",
        yaxis_title = "Load Left",
        coloraxis_colorscale = colorscale2
    )

    heatmap_plot = plot(PlotlyJS.heatmap(
        x = 1:288,
        y = load_bins[1:end-1],
        z = state_visits,
        coloraxis = "coloraxis"
    ), layout)

    display(heatmap_plot)
end