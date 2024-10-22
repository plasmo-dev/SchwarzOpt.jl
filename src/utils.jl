"""
    _find_boundary_edges(
        projection::Plasmo.HyperGraphProjection, 
        subgraphs::Vector{OptiGraph}
    )

    Find the boundary edges given a hypergraph projection and vector of subgraphs.
"""
function _find_boundary_edges(
    projection::Plasmo.HyperGraphProjection, 
    subgraphs::Vector{OptiGraph}
)
    boundary_edge_list = Vector{Vector{OptiEdge}}()
    for subgraph in subgraphs
        subgraph_nodes = all_nodes(subgraph)
        boundary_edges = Plasmo.incident_edges(projection, subgraph_nodes)
        push!(boundary_edge_list, boundary_edges)
    end
    return boundary_edge_list
end

"""
    _get_boundary_constraints(subgraph_boundary_edges::Vector{Vector{OptiEdge}})

    Collect the constraints associated with subgraph boundary edges. Return a vector
    of constraints for each subgraph.
"""
function _get_boundary_constraints(subgraph_boundary_edges::Vector{Vector{OptiEdge}})
    subgraph_boundary_constraints = Vector{Vector{ConstraintRef}}()
    for edge_vector in subgraph_boundary_edges
        edge_constraints = Vector{ConstraintRef}()
        for edge in edge_vector
            append!(edge_constraints, all_constraints(edge))
        end
        push!(subgraph_boundary_constraints, edge_constraints)
    end
    return subgraph_boundary_constraints
end

function _is_objective_separable(::Number)
    return true
end

function _is_objective_separable(::Plasmo.NodeVariableRef)
    return true
end

function _is_objective_separable(::Plasmo.GenericAffExpr{<:Number,NodeVariableRef})
    return true
end

function _is_objective_separable(
    func::Plasmo.GenericQuadExpr{<:Number,NodeVariableRef},
)
    # check each term; make sure they are all on the same subproblem
    for term in Plasmo.quad_terms(func)
        # term = (coefficient, variable_1, variable_2)
        node1 = get_node(term[2])
        node2 = get_node(term[3])
        
        # if any term is split across nodes, the objective is not separable
        if node1 != node2
            return false
        end
    end
    return true
end

function _is_objective_separable( 
    func::Plasmo.GenericNonlinearExpr{NodeVariableRef}
)
    # check for a constant multiplier
    if func.head == :*
        if !(func.args[1] isa Number)
            return false
        end
    end

    # if not additive, check if term is separable
    if func.head != :+ && func.head != :-
        vars = Plasmo._extract_variables(func)
        nodes = get_node.(vars)
        if length(unique(nodes)) > 1
            return false
        end
    end

    # check each argument
    for arg in func.args
        if !(_is_objective_separable(arg))
            return false
        end
    end
    return true
end

# NOTE: objective must be separable
function set_node_objectives_from_graph(graph::OptiGraph)
    obj = objective_function(graph)
    if !(_is_objective_separable(obj))
        error("Cannot set node objectives from graph. It is not separable across nodes.")
    end
    sense = objective_sense(graph)
    _set_node_objectives_from_graph(obj, sense)
    return nothing
end

function _set_node_objectives_from_graph(
    func::Plasmo.NodeVariableRef, 
    sense::MOI.OptimizationSense
)
    node = get_node(func)
    set_objective_function(node, func)
    set_objective_sense(node, sense)
    return nothing
end

function _set_node_objectives_from_graph(
    func::Plasmo.GenericAffExpr{<:Number,NodeVariableRef}, 
    sense::MOI.OptimizationSense
)
    # collect terms for each node

    
    node = get_node(func)
    set_objective_function(node, func)
    set_objective_sense(node, sense)
    return nothing
end

# TODO Utilites

# check that overlap is at least 1
function _check_overlap()
end

# check whether partitions are contiguous
function _check_contiguous_partitions()
end

# modify non_contiguous partitions to make them contiguous
function _fix_non_contiguous_partitions()
end

function _check_hierarchical()
end

