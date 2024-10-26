# JuMP.jl/Plasmo.jl methods

function Plasmo.value(optimizer::Optimizer, nvref::NodeVariableRef)
    node = get_node(nvref)
    subproblem_graph = _get_subproblem_graph(node)
    return Plasmo.value(subproblem_graph, nvref)
end

function Plasmo.dual(optimizer::Optimizer, cref::Plasmo.EdgeConstraintRef)
    edge = get_edge(cref)
    subproblem_graph = _get_subproblem_graph(edge)
    return Plasmo.dual(subproblem_graph, cref)
end

function Plasmo.objective_value(optimizer::Optimizer)
    return optimizer.objective_value
end

function Plasmo.termination_status(optimizer::Optimizer)
    return optimizer.termination_status
end

function Plasmo.solve_time(optimizer::Optimizer)
    return optimizer.solve_time
end

# TODO: Possible MOI Methods
# MOI functions to make the Schwarz optimizer work directly as a backend. 
# We use Plasmo.jl methods in the algorithm, so we are abusing MOI in a sense by doing this. 

# function MOI.optimize!(optimizer::Optimizer)
#     run_algorithm!(optimizer)
# end

# MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue) = optimizer.objective_value

# MOI.get(optimizer::Optimizer, attr::MOI.TerminationStatus) = optimizer.status

# MOI.get(optimizer::Optimizer, attr::MOI.SolveTimeSec) = optimizer.solve_time
