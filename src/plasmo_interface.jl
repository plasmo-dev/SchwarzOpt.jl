# printing

function Base.string(algorithm::Algorithm)
    return @sprintf(
        """
        SchwarzAlgorithm
        %30s %9s
        %30s %9s
        %30s %9s
        %30s %9s
        """,
        "Number of subproblems:",
        length(algorithm.expanded_subgraphs),
        "Number of variables:",
        Plasmo.num_variables(algorithm.graph),
        "Number of constraints:",
        Plasmo.num_constraints(algorithm.graph),
        "Number of linking constraints:",
        Plasmo.num_local_link_constraints(algorithm.graph)
    )
end
Base.print(io::IO, algorithm::Algorithm) = Base.print(io, Base.string(algorithm))
Base.show(io::IO, algorithm::Algorithm) = Base.print(io, algorithm)

# JuMP.jl/Plasmo.jl methods for algorithm

function Plasmo.value(algorithm::Algorithm, nvref::NodeVariableRef)
    node = get_node(nvref)
    subproblem_graph = _get_subproblem_graph(node)
    return Plasmo.value(subproblem_graph, nvref)
end

function Plasmo.dual(algorithm::Algorithm, cref::Plasmo.EdgeConstraintRef)
    edge = get_edge(cref)
    subproblem_graph = _get_subproblem_graph(edge)
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
