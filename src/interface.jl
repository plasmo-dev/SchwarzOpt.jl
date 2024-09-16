# MOI.optimize! does not work for Plasmo Algorithm optimizers
function MOI.optimize!(optimizer::Optimizer)
    # return error(
    #     "SchwarzOpt does not work directly through MathOptInterface. ",
    # )
    run_algorithm!(optimizer)
end


#MOI functions to make the Schwarz optimizer work directly with Plasmo.jl
MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue) = optimizer.objective_value
MOI.get(optimizer::Optimizer, attr::MOI.TerminationStatus) = optimizer.status
MOI.get(optimizer::Optimizer, attr::MOI.SolveTimeSec) = optimizer.solve_time


# Plasmo.value(optimizer::Optimizer, nvref::NodeVariableRef)

# Plasmo.dual()

