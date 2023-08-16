include((@__DIR__) * "/../examples/optimal_control.jl")

@test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
