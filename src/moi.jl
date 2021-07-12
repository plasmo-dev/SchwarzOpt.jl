#MOI functions to make the Schwarz optimizer work directly with Plasmo.jl

#NOTE: Optinode solutions will point to the Optimizer solution
MOI.get(optimizer::Optimizer,attr::MOI.ObjectiveValue) = optimizer.objective_value
MOI.get(optimizer::Optimizer,attr::MOI.TerminationStatus) = optimizer.status
MOI.get(optimizer::Optimizer,attr::MOI.SolveTime) = optimizer.solve_time
Plasmo.supported_structures(optimizer::Optimizer) = [Plasmo.RECURSIVE_GRAPH_STRUCTURE]

#This function points Plasmo.jl to the correct optimize function
MOI.get(optimizer::Optimizer,::Plasmo.OptiGraphOptimizeHook) = optimizer.plasmo_optimizer_hook

#JuMP will hit this if a user decides to try to use SchwarzOpt with it
MOI.optimize!(optimizer::Optimizer) = error("SchwarzOpt does not yet work directly through MathOptInterface.  The solver currently only supports models created using Plasmo.jl.")

#These are needed to get attach_optimizer and AttributeFromOptimizer to work correctly.
function MOI.is_empty(optimizer::Optimizer) = optimizer.graph=nothing
#TODO
MOI.empty!(::Optimizer) = nothing
MOI.copy_to(dest::Optimizer,src::MOI.ModelLike;kwargs...) = MOIU.default_copy_to(dest,src;kwargs...)
MOIU.supports_default_copy_to(model::Optimizer, copy_names::Bool) = true
