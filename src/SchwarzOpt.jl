module SchwarzOpt

using Printf
using DataStructures
using LinearAlgebra
import Base.@kwdef

using MathOptInterface, Plasmo, JuMP

include("utils.jl")

include("optimizer.jl")

include("moi.jl")

end
