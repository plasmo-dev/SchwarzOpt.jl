# collect linkconstraints on boundary edges
function _gather_links(subgraphs, subgraph_boundary_edges)
    subgraph_links = []
    for (i, edge_set) in enumerate(subgraph_boundary_edges)
        links = LinkConstraintRef[]
        for edge in edge_set
            linkrefs = edge.linkrefs
            for linkref in linkrefs
                push!(links, linkref)
            end
        end
        push!(subgraph_links, links)
    end
    return subgraph_links
end

# find boundary edges of expanded subgraphs
function _find_boundaries(optigraph::OptiGraph, subgraphs::Vector{OptiGraph})
    boundary_linkedges_list = []
    for subgraph in subgraphs
        subnodes = all_nodes(subgraph)
        boundary_edges = Plasmo.incident_edges(optigraph, subnodes)
        push!(boundary_linkedges_list, boundary_edges)
    end
    return boundary_linkedges_list
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