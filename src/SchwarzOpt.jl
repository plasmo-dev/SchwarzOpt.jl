module SchwarzOpt

using Printf
using DataStructures
using LinearAlgebra
import Base.@kwdef
# import LinearAlgebra: BLAS

using MathOptInterface, Plasmo, JuMP

include("utils.jl")

include("optimizer.jl")

include("moi.jl")


end
