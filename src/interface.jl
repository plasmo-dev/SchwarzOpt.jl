# Supported methods for querying algorithm results

function Plasmo.value(algorithm::Algorithm, nvref::NodeVariableRef)
    node = get_node(nvref)
    subproblem_graph = _get_subproblem(algorithm, node)
    return Plasmo.value(subproblem_graph, nvref)
end

function Plasmo.dual(algorithm::Algorithm, cref::Plasmo.EdgeConstraintRef)
    edge = get_edge(cref)
    subproblem_graph = _get_subproblem(algorithm, edge)
    return Plasmo.dual(subproblem_graph, cref)
end

function Plasmo.objective_value(algorithm::Algorithm)
    return algorithm.objective_value
end

function Plasmo.termination_status(algorithm::Algorithm)
    return algorithm.status
end

function Plasmo.solve_time(algorithm::Algorithm)
    return algorithm.solve_time
end
