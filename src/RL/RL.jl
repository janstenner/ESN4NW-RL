# TODO: make module
# module RL


# imports

using LinearAlgebra
using IntervalSets
using DataFrames
using Statistics
using UnicodePlots:lineplot
using Flux
using CUDA
using Adapt
using Random
using CircularArrayBuffers
using ElasticArrays
using MacroTools: @forward
using ProgressMeter
using StableRNGs
using Setfield: @set
using Zygote




# abstract types

abstract type AbstractEnv end
abstract type AbstractPolicy end
abstract type AbstractHook end
abstract type AbstractStage end
abstract type AbstractTrajectory end




# includes

include("./stages.jl")
include("./device.jl")
include("./base.jl")
include("./env.jl")
include("./hook.jl")
include("./trajectory.jl")
include("./agent.jl")
include("./stop_conditions.jl")
include("./run.jl")
include("./custom_nna.jl")
include("./agent_ddpg.jl")
include("./agent_ddpg team.jl")



# code to export all, taken from https://discourse.julialang.org/t/exportall/4970/18
# for n in names(@__MODULE__; all=true)
#     if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
#         @eval export $n
#     end
# end

# end
# module