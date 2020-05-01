module MGSchwarzSolver

using Printf
using DataStructures
using LinearAlgebra
using MathOptInterface
using Plasmo
using Ipopt


export schwarz_solve

include("schwarz_solver.jl")

end # module
