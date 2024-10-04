module SchwarzOpt

using Printf
using DataStructures
using LinearAlgebra
using Metis
import Base.@kwdef

using MathOptInterface, Plasmo, JuMP

include("utils.jl")

include("optimizer.jl")

include("interface.jl")

end
