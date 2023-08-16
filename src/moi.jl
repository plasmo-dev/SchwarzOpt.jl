#MOI functions to make the Schwarz optimizer work directly with Plasmo.jl
MOI.get(optimizer::Optimizer,attr::MOI.ObjectiveValue) = optimizer.objective_value
MOI.get(optimizer::Optimizer,attr::MOI.TerminationStatus) = optimizer.status
MOI.get(optimizer::Optimizer,attr::MOI.SolveTimeSec) = optimizer.solve_time
MOI.optimize!(optimizer::Optimizer) = error("SchwarzOpt does not yet work directly through MathOptInterface.  The solver currently only supports models created using Plasmo.jl.")