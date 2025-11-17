traj = multiple_day_trajectory;
p = agent.policy;
device = RL.device

n_samples = length(traj)
s = send_to_device(device(p.qnetwork1), traj[:state])
a = send_to_device(device(p.qnetwork1), traj[:action])
r = send_to_device(device(p.qnetwork1), traj[:reward])
t = send_to_device(device(p.qnetwork1), traj[:terminal])
next_states = deepcopy(circshift(s, (0,0,-1)))


γ, τ, α = p.γ, p.τ, p.α


K = 16

μ, logσ = p.actor(p.device_rng, next_states)

acc_mu  = zeros(Float32, 1, size(μ,2), size(μ,3)) 
acc_logp = zeros(Float32, 1, size(μ,2), size(μ,3)) 

for k in 1:K
    a_plus, logp_π_plus, a_minus, logp_π_minus = p.actor(p.device_rng, next_states; is_sampling=true, is_return_log_prob=true, is_antithetic = true)

    acc_logp .+= logp_π_plus .+ logp_π_minus


    y_plus1  = send_to_host(p.target_qnetwork1(vcat(next_states, a_plus)))
    y_minus1 = send_to_host(p.target_qnetwork1(vcat(next_states, a_minus)))

    y_plus2  = send_to_host(p.target_qnetwork2(vcat(next_states, a_plus)))
    y_minus2 = send_to_host(p.target_qnetwork2(vcat(next_states, a_minus)))


    acc_mu .+= min.(y_plus1, y_plus2) .- α .* logp_π_plus
    acc_mu .+= min.(y_minus1, y_minus2) .- α .* logp_π_minus
end


acc_mu ./= 2*K
acc_logp ./= 2*K

logp_π′ = acc_logp

next_values = acc_mu
next_values[:,:, end] .*= 0.0f0     # terminal states

n_envs = size(t, 1)
next_values = reshape( next_values, n_envs, :)

targets = td_lambda_targets(r, t, next_values, γ; λ = p.λ_targets)

q_input = vcat(s, a)

q1 = dropdims(p.qnetwork1(q_input), dims=1)
q2 = dropdims(p.qnetwork2(q_input), dims=1)

p1 = scatter(y=targets[:], name = "targets")
p2 = scatter(y=min.(q1, q2)[:], name = "min q1 q2")

plot([p1, p2])