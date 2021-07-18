include((@__DIR__)*"/../examples/optimal_control.jl")

@test termination_status(graph) == MOI.LOCALLY_SOLVED
