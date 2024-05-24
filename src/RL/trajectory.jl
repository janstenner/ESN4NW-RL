Base.haskey(t::AbstractTrajectory, s::Symbol) = s in keys(t)
Base.isempty(t::AbstractTrajectory) = all(k -> isempty(t[k]), keys(t))

function Base.empty!(t::AbstractTrajectory)
    for k in keys(t)
        empty!(t[k])
    end
end

function Base.push!(t::AbstractTrajectory; kwargs...)
    for (k, v) in kwargs
        push!(t[k], v)
    end
end

function Base.pop!(t::AbstractTrajectory)
    for k in keys(t)
        pop!(t[k])
    end
end

function Base.show(io::IO, t::AbstractTrajectory)
    println(io, "Trajectory of $(length(keys(t))) traces:")
    for k in keys(t)
        show(io, k)
        println(io, " $(summary(t[k]))")
    end
end

#####
# Common Keys
#####

const SART = (:state, :action, :reward, :terminal)
const SARTS = (:state, :action, :reward, :terminal, :next_state)
const SARTSA = (:state, :action, :reward, :terminal, :next_state, :next_action)
const SLART = (:state, :legal_actions_mask, :action, :reward, :terminal)
const SLARTSL = (
    :state,
    :legal_actions_mask,
    :action,
    :reward,
    :terminal,
    :next_state,
    :next_legal_actions_mask,
)
const SLARTSLA = (
    :state,
    :legal_actions_mask,
    :action,
    :reward,
    :terminal,
    :next_state,
    :next_legal_actions_mask,
    :next_action,
)




#####
# Trajectory
#####

"""
    Trajectory(;[trace_name=trace_container]...)

A simple wrapper of `NamedTuple`.
Define our own type here to avoid type piracy with `NamedTuple`
"""
struct Trajectory{T} <: AbstractTrajectory
    traces::T
end

Trajectory(; kwargs...) = Trajectory(values(kwargs))

@forward Trajectory.traces Base.getindex, Base.keys

Base.merge(a::Trajectory, b::Trajectory) = Trajectory(merge(a.traces, b.traces))
Base.merge(a::Trajectory, b::NamedTuple) = Trajectory(merge(a.traces, b))
Base.merge(a::NamedTuple, b::Trajectory) = Trajectory(merge(a, b.traces))

#####

"""
    CircularArrayTrajectory(; capacity::Int, kw::Pair{<:DataType, <:Tuple{Vararg{Int}}}...)

A specialized [`Trajectory`](@ref) which uses
[`CircularArrayBuffer`](https://github.com/JuliaReinforcementLearning/CircularArrayBuffers.jl#usage)
as the underlying storage. `kw` specifies the name, the element type and the
size of each trace. `capacity` is used to define the maximum length of the
underlying buffer.

See also [`CircularArraySARTTrajectory`](@ref),
[`CircularArraySLARTTrajectory`](@ref), [`CircularArrayPSARTTrajectory`](@ref).
"""
function CircularArrayTrajectory(; capacity, kwargs...)
    Trajectory(map(values(kwargs)) do x
        CircularArrayBuffer{eltype(first(x))}(last(x)..., capacity)
    end)
end

"""
    CircularVectorTrajectory(;capacity, kw::DataType)

Similar to [`CircularArrayTrajectory`](@ref), except that the underlying storage is
[`CircularVectorBuffer`](https://github.com/JuliaReinforcementLearning/CircularArrayBuffers.jl#usage).

!!! note
    Note the different type of the `kw` between `CircularVectorTrajectory` and `CircularArrayTrajectory`. With 
    [`CircularVectorBuffer`](https://github.com/JuliaReinforcementLearning/CircularArrayBuffers.jl#usage)
    as the underlying storage, we don't need the size info.

See also [`CircularVectorSARTTrajectory`](@ref), [`CircularVectorSARTSATrajectory`](@ref).
"""
function CircularVectorTrajectory(; capacity, kwargs...)
    Trajectory(map(values(kwargs)) do x
        CircularVectorBuffer{x}(capacity)
    end)
end

#####

const CircularArraySARTTrajectory = Trajectory{
    <:NamedTuple{
        SART,
        <:Tuple{
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
        },
    },
}

"""
    CircularArraySARTTrajectory(;capacity::Int, kw...)

A specialized [`CircularArrayTrajectory`](@ref) with traces of [`SART`](@ref).
Note that the capacity of the `:state` and `:action` trace is one step longer
than the capacity of the `:reward` and `:terminal` trace, so that we can reuse
the same trace to represent the next state and next action in a typical
transition in reinforcement learning.

# Keyword arguments

- `capacity::Int`, the maximum number of transitions.
- `state::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Int => ()`,
- `action::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Int => ()`,
- `reward::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Float32 => ()`,
- `terminal::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Bool => ()`,

# Example

```julia-repl
julia> t = CircularArraySARTTrajectory(;
           capacity = 3,
           state = Vector{Int} => (4,),
           action = Int => (),
           reward = Float32 => (),
           terminal = Bool => (),
       )
Trajectory of 4 traces:
:state 4×0 CircularArrayBuffers.CircularArrayBuffer{Int64, 2}
:action 0-element CircularArrayBuffers.CircularVectorBuffer{Int64}
:reward 0-element CircularArrayBuffers.CircularVectorBuffer{Float32}
:terminal 0-element CircularArrayBuffers.CircularVectorBuffer{Bool}


julia> for i in 1:4
           push!(t;state=ones(Int, 4) .* i, action = i, reward=i/2, terminal=iseven(i))
       end

julia> push!(t;state=ones(Int,4) .* 5, action = 5)

julia> t[:state]
4×4 CircularArrayBuffers.CircularArrayBuffer{Int64, 2}:
 2  3  4  5
 2  3  4  5
 2  3  4  5
 2  3  4  5

julia> t[:action]
4-element CircularArrayBuffers.CircularVectorBuffer{Int64}:
 2
 3
 4
 5

julia> t[:reward]
3-element CircularArrayBuffers.CircularVectorBuffer{Float32}:
 1.0
 1.5
 2.0

julia> t[:terminal]
3-element CircularArrayBuffers.CircularVectorBuffer{Bool}:
 1
 0
 1
```
"""
CircularArraySARTTrajectory(;
    capacity::Int,
    state = Int => (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
) = merge(
    CircularArrayTrajectory(; capacity = capacity + 1, state = state, action = action),
    CircularArrayTrajectory(; capacity = capacity, reward = reward, terminal = terminal),
)

const CircularArraySLARTTrajectory = Trajectory{
    <:NamedTuple{
        SLART,
        <:Tuple{
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
        },
    },
}

"Similar to [`CircularArraySARTTrajectory`](@ref) with an extra `legal_actions_mask` trace."
CircularArraySLARTTrajectory(;
    capacity::Int,
    state = Int => (),
    legal_actions_mask,
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
) = merge(
    CircularArrayTrajectory(;
        capacity = capacity + 1,
        state = state,
        legal_actions_mask = legal_actions_mask,
        action = action,
    ),
    CircularArrayTrajectory(; capacity = capacity, reward = reward, terminal = terminal),
)

#####

const CircularVectorSARTTrajectory = Trajectory{
    <:NamedTuple{
        SART,
        <:Tuple{
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
        },
    },
}

"""
    CircularVectorSARTTrajectory(;capacity, kw::DataType...)

A specialized [`CircularVectorTrajectory`](@ref) with traces of [`SART`](@ref).
Note that the capacity of traces `:state` and `:action` are one step longer than
the traces of `:reward` and `:terminal`, so that we can reuse the same
underlying storage to represent the next state and next action in a typical
transition in reinforcement learning.

# Keyword arguments

- `capacity::Int`
- `state` = `Int`,
- `action` = `Int`,
- `reward` = `Float32`,
- `terminal` = `Bool`,

# Example

```julia-repl
julia> t = CircularVectorSARTTrajectory(;
           capacity = 3,
           state = Vector{Int},
           action = Int,
           reward = Float32,
           terminal = Bool,
       )
Trajectory of 4 traces:
:state 0-element CircularArrayBuffers.CircularVectorBuffer{Vector{Int64}}
:action 0-element CircularArrayBuffers.CircularVectorBuffer{Int64}
:reward 0-element CircularArrayBuffers.CircularVectorBuffer{Float32}
:terminal 0-element CircularArrayBuffers.CircularVectorBuffer{Bool}


julia> for i in 1:4
           push!(t;state=ones(Int, 4) .* i, action = i, reward=i/2, terminal=iseven(i))
       end

julia> push!(t;state=ones(Int,4) .* 5, action = 5)

julia> t[:state]
4-element CircularArrayBuffers.CircularVectorBuffer{Vector{Int64}}:
 [2, 2, 2, 2]
 [3, 3, 3, 3]
 [4, 4, 4, 4]
 [5, 5, 5, 5]

julia> t[:action]
4-element CircularArrayBuffers.CircularVectorBuffer{Int64}:
 2
 3
 4
 5

julia> t[:reward]
3-element CircularArrayBuffers.CircularVectorBuffer{Float32}:
 1.0
 1.5
 2.0

julia> t[:terminal]
3-element CircularArrayBuffers.CircularVectorBuffer{Bool}:
 1
 0
 1
```
"""
CircularVectorSARTTrajectory(;
    capacity::Int,
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
) = merge(
    CircularVectorTrajectory(; capacity = capacity + 1, state = state, action = action),
    CircularVectorTrajectory(; capacity = capacity, reward = reward, terminal = terminal),
)

#####

const CircularVectorSARTSATrajectory = Trajectory{
    <:NamedTuple{
        SARTSA,
        <:Tuple{
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
            <:CircularVectorBuffer,
        },
    },
}

"Similar to [`CircularVectorSARTTrajectory`](@ref) with another two traces of `(:next_state, :next_action)`"
CircularVectorSARTSATrajectory(;
    capacity::Int,
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
    next_state = state,
    next_action = action,
) = CircularVectorTrajectory(;
    capacity = capacity,
    state = state,
    action = action,
    reward = reward,
    terminal = terminal,
    next_state = next_state,
    next_action = next_action,
)

#####

"""
    ElasticArrayTrajectory(;[trace_name::Pair{<:DataType, <:Tuple{Vararg{Int}}}]...)

A specialized [`Trajectory`](@ref) which uses [`ElasticArray`](https://github.com/JuliaArrays/ElasticArrays.jl) as the underlying
storage. See also [`ElasticSARTTrajectory`](@ref).
"""
function ElasticArrayTrajectory(; kwargs...)
    Trajectory(map(values(kwargs)) do x
        ElasticArray{eltype(first(x))}(undef, last(x)..., 0)
    end)
end

const ElasticSARTTrajectory = Trajectory{
    <:NamedTuple{SART,<:Tuple{<:ElasticArray,<:ElasticArray,<:ElasticArray,<:ElasticArray}},
}

"""
    ElasticSARTTrajectory(;kw...)

A specialized [`ElasticArrayTrajectory`](@ref) with traces of [`SART`](@ref).

# Keyword arguments

- `state::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Int => ()`, by default it
  means the state is a scalar of `Int`.
- `action::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Int => ()`,
- `reward::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Float32 => ()`,
- `terminal::Pair{<:DataType, <:Tuple{Vararg{Int}}}` = `Bool => ()`,

# Example

```julia-repl
julia> t = ElasticSARTTrajectory(;
           state = Vector{Int} => (4,),
           action = Int => (),
           reward = Float32 => (),
           terminal = Bool => (),
       )
Trajectory of 4 traces:
:state 4×0 ElasticArrays.ElasticMatrix{Int64, Vector{Int64}}
:action 0-element ElasticArrays.ElasticVector{Int64, Vector{Int64}}
:reward 0-element ElasticArrays.ElasticVector{Float32, Vector{Float32}}
:terminal 0-element ElasticArrays.ElasticVector{Bool, Vector{Bool}}


julia> for i in 1:4
           push!(t;state=ones(Int, 4) .* i, action = i, reward=i/2, terminal=iseven(i))
       end

julia> push!(t;state=ones(Int,4) .* 5, action = 5)

julia> t
Trajectory of 4 traces:
:state 4×5 ElasticArrays.ElasticMatrix{Int64, Vector{Int64}}
:action 5-element ElasticArrays.ElasticVector{Int64, Vector{Int64}}
:reward 4-element ElasticArrays.ElasticVector{Float32, Vector{Float32}}
:terminal 4-element ElasticArrays.ElasticVector{Bool, Vector{Bool}}

julia> t[:state]
4×5 ElasticArrays.ElasticMatrix{Int64, Vector{Int64}}:
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5
 1  2  3  4  5

julia> t[:action]
5-element ElasticArrays.ElasticVector{Int64, Vector{Int64}}:
 1
 2
 3
 4
 5

julia> t[:reward]
4-element ElasticArrays.ElasticVector{Float32, Vector{Float32}}:
 0.5
 1.0
 1.5
 2.0

julia> t[:terminal]
4-element ElasticArrays.ElasticVector{Bool, Vector{Bool}}:
 0
 1
 0
 1

julia> empty!(t)

julia> t
Trajectory of 4 traces:
:state 4×0 ElasticArrays.ElasticMatrix{Int64, Vector{Int64}}
:action 0-element ElasticArrays.ElasticVector{Int64, Vector{Int64}}
:reward 0-element ElasticArrays.ElasticVector{Float32, Vector{Float32}}
:terminal 0-element ElasticArrays.ElasticVector{Bool, Vector{Bool}}
```

"""
function ElasticSARTTrajectory(;
    state = Int => (),
    action = Int => (),
    reward = Float32 => (),
    terminal = Bool => (),
)
    ElasticArrayTrajectory(;
        state = state,
        action = action,
        reward = reward,
        terminal = terminal,
    )
end

#####
# VectorTrajectory
#####

"""
    VectorTrajectory(;[trace_name::DataType]...)

A [`Trajectory`](@ref) with each trace using a `Vector` as the storage.
"""
function VectorTrajectory(; kwargs...)
    Trajectory(map(values(kwargs)) do x
        Vector{x}()
    end)
end

const VectorSARTTrajectory =
    Trajectory{<:NamedTuple{SART,<:Tuple{<:Vector,<:Vector,<:Vector,<:Vector}}}

"""
    VectorSARTTrajectory(;kw...)

A specialized [`VectorTrajectory`] with traces of [`SART`](@ref).

# Keyword arguments

- `state::DataType = Int`
- `action::DataType = Int`
- `reward::DataType = Float32`
- `terminal::DataType = Bool`
"""
function VectorSARTTrajectory(;
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
)
    VectorTrajectory(; state = state, action = action, reward = reward, terminal = terminal)
end

const VectorSATrajectory =
    Trajectory{<:NamedTuple{(:state, :action),<:Tuple{<:Vector,<:Vector}}}

"""
    VectorSATrajectory(;kw...)

A specialized [`VectorTrajectory`] with traces of `(:state, :action)`.

# Keyword arguments

- `state::DataType = Int`
- `action::DataType = Int`
"""
function VectorSATrajectory(; state = Int, action = Int)
    VectorTrajectory(; state = state, action = action)
end
#####

Base.@kwdef struct PrioritizedTrajectory{T,P} <: AbstractTrajectory
    traj::T
    priority::P
end

Base.keys(t::PrioritizedTrajectory) = (:priority, keys(t.traj)...)

Base.length(t::PrioritizedTrajectory) = length(t.priority)

Base.getindex(t::PrioritizedTrajectory, s::Symbol) =
    if s == :priority
        t.priority
    else
        getindex(t.traj, s)
    end


#####
# Common
#####

function Base.length(
    t::Union{
        CircularArraySARTTrajectory,
        CircularArraySLARTTrajectory,
        CircularVectorSARTSATrajectory,
        ElasticSARTTrajectory,
    },
)
    x = t[:terminal]
    size(x, ndims(x))
end

Base.length(t::VectorSARTTrajectory) = length(t[:terminal])
Base.length(t::VectorSATrajectory) = length(t[:action])




abstract type AbstractInserter end

Base.@kwdef struct NStepInserter <: AbstractInserter
    n::Int = 1
end

function Base.push!(
    t::CircularVectorSARTSATrajectory,
    𝕥::CircularArraySARTTrajectory,
    inserter::NStepInserter,
)
    N = length(𝕥)
    n = inserter.n
    for i in 1:(N-n+1)
        for k in SART
            push!(t[k], select_last_dim(𝕥[k], i))
        end
        push!(t[:next_state], select_last_dim(𝕥[:state], i + n))
        push!(t[:next_action], select_last_dim(𝕥[:action], i + n))
    end
end

#####
# Samplers
#####

abstract type AbstractSampler{traces} end

"""
    sample([rng=Random.GLOBAL_RNG], trajectory, sampler, [traces=Val(keys(trajectory))])

!!! note
    Here we return a copy instead of a view:
    1. Each sample is independent of the original `trajectory` so that `trajectory` can be updated async.
    2. [Copy is not always so bad](https://docs.julialang.org/en/v1/manual/performance-tips/#Copying-data-is-not-always-bad).
"""
function sample(t::AbstractTrajectory, sampler::AbstractSampler)
    sample(Random.GLOBAL_RNG, t, sampler)
end

#####
## BatchSampler
#####

mutable struct BatchSampler{traces} <: AbstractSampler{traces}
    batch_size::Int
    cache::Any
    rng::Any
end

BatchSampler(batch_size::Int; cache = nothing, rng = Random.GLOBAL_RNG) =
    BatchSampler{SARTSA}(batch_size, cache, rng)
BatchSampler{T}(batch_size::Int; cache = nothing, rng = Random.GLOBAL_RNG) where {T} =
    BatchSampler{T}(batch_size, cache, rng)

(s::BatchSampler)(t::AbstractTrajectory) = sample(s.rng, t, s)

function sample(rng::AbstractRNG, t::AbstractTrajectory, s::BatchSampler)
    inds = rand(rng, 1:length(t)-1, s.batch_size)
    fetch!(s, t, inds)
    inds, s.cache
end

function fetch!(s::BatchSampler, t::AbstractTrajectory, inds::Vector{Int})
    batch = NamedTuple{keys(t)}(view(t[x], inds) for x in keys(t))
    if isnothing(s.cache)
        s.cache = map(Flux.batch, batch)
    else
        map(s.cache, batch) do dest, src
            batch!(dest, src)
        end
    end
end

function fetch!(s::BatchSampler{traces}, t::Union{CircularArraySARTTrajectory, CircularArraySLARTTrajectory}, inds::Vector{Int}) where {traces}
    if traces == SARTS
        batch = NamedTuple{SARTS}((
            (consecutive_view(t[x], inds) for x in SART)...,
            consecutive_view(t[:state], inds .+ 1),
        ))
    elseif traces == SLARTSL
        batch = NamedTuple{SLARTSL}((
            (consecutive_view(t[x], inds) for x in SLART)...,
            consecutive_view(t[:state], inds .+ 1),
            consecutive_view(t[:legal_actions_mask], inds .+ 1),
        ))
    else
        @error "unsupported traces $traces"
    end
    
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



# custom sample function

function pde_sample(rng::AbstractRNG, t::AbstractTrajectory, s::BatchSampler, number_actuators::Int)
    inds = rand(rng, 1:length(t)-number_actuators, s.batch_size)
    pde_fetch!(s, t, inds, number_actuators)
    inds, s.cache
end

function pde_fetch!(s::BatchSampler, t::Trajectory, inds::Vector{Int}, number_actuators::Int)

    batch = NamedTuple{SARTS}((
        (consecutive_view(t[x], inds) for x in SART)...,
        consecutive_view(t[:state], inds .+ number_actuators),
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