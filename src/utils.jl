"""
    _find_boundaries(projection::Plasmo.HyperGraphProjection, subgraphs::Vector{OptiGraph})

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
    _get_edge_constraints(subgraph_boundary_edges::Vector{Vector{OptiEdge}})

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

# function _expand_subgraphs(graph::OptiGraph, overlap::Int64)
#     subproblem_graphs = []
#     boundary_linkedges_list = []
#     hypergraph, hyper_map = gethypergraph(graph)

#     #NOTE: This could all be calculated simultaneously
#     for subgraph in getsubgraphs(graph)
#         println("Performing neighborhood expansion...")
#         subnodes = all_nodes(subgraph)
#         hypernodes = [hyper_map[node] for node in subnodes]

#         #Return new hypernodes and hyperedges covered through the expansion.
#         overlap_hnodes = Plasmo.neighborhood(hypergraph, hypernodes, overlap)
#         overlap_hedges = Plasmo.induced_edges(hypergraph, overlap_hnodes)
#         boundary_hedges = Plasmo.incident_edges(hypergraph, overlap_hnodes)

#         overlap_nodes = [hyper_map[node] for node in overlap_hnodes]
#         overlap_edges = [hyper_map[edge] for edge in overlap_hedges]
#         boundary_edges = [hyper_map[edge] for edge in boundary_hedges]

#         #Setup subproblem graphs
#         subproblem_graph = OptiGraph()
#         subproblem_graph.modelnodes = overlap_nodes
#         subproblem_graph.linkedges = overlap_edges

#         push!(subproblem_graphs, subproblem_graph)
#         push!(boundary_linkedges_list, boundary_edges)
#     end

#     return subproblem_graphs, boundary_linkedges_list
# end