module SchwarzOpt

using Printf
using DataStructures
using LinearAlgebra
import LinearAlgebra: BLAS

using MathOptInterface, Plasmo, JuMP

export schwarz_solve

include("utils.jl")

include("optimizer.jl")

include("schwarz_solve.jl")

end
