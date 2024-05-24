

Base.@kwdef struct CustomNeuralNetworkApproximator{M,O}
    model::M
    optimizer::O = nothing
end

# some model may accept multiple inputs
(app::CustomNeuralNetworkApproximator)(args...; kwargs...) = app.model(args...; kwargs...)

@forward CustomNeuralNetworkApproximator.model Flux.testmode!,
Flux.trainmode!,
Flux.params,
device


update!(app::CustomNeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, Flux.params(app), gs)

copyto!(dest::CustomNeuralNetworkApproximator, src::CustomNeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, Flux.params(src))