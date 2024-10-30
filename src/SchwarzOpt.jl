module SchwarzOpt

using Printf
using DataStructures
using LinearAlgebra
using Metis
import Base.@kwdef

using MathOptInterface, Plasmo, JuMP

include("utils.jl")

include("algorithm.jl")

include("plasmo_interface.jl")

end
