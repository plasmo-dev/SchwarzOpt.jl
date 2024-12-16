"""
    _find_boundary_edges(
        projection::Plasmo.HyperGraphProjection, 
        subgraphs::Vector{OptiGraph}
    )

    Find the boundary edges given a hypergraph projection and vector of subgraphs.
"""
function _find_boundary_edges(graph::OptiGraph, subgraphs::Vector{OptiGraph})
    projection = hyper_projection(graph)
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
function _extract_constraints(subgraph_boundary_edges::Vector{Vector{OptiEdge}})
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

# TODO: utilities for helpful checks

function _is_hierarchical(graph::OptiGraph)
    return !isempty(graph.subgraphs) && !isempty(graph.optinodes)
end

# check that overlap is at least 1
function _check_overlap() end

# check whether partitions are contiguous
function _check_contiguous_partitions() end

# modify non_contiguous partitions to make them contiguous
function _fix_non_contiguous_partitions() end
