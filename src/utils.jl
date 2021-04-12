#TODO: Update internal expand
function _expand_subgraphs(graph::OptiGraph,overlap::Int64)
    subproblem_graphs = []
    boundary_linkedges_list = []
    hypergraph,hyper_map = gethypergraph(graph)

    #NOTE: This could all be calculated simultaneously
    for subgraph in getsubgraphs(mg)

        println("Performing neighborhood expansion...")
        subnodes = all_nodes(subgraph)
        hypernodes = [hyper_map[node] for node in subnodes]

        #Return new hypernodes and hyperedges covered through the expansion.
        overlap_hnodes = Plasmo.neighborhood(hypergraph,hypernodes,overlap)
        overlap_hedges = Plasmo.induced_edges(hypergraph,overlap_hnodes)
        boundary_hedges = Plasmo.incident_edges(hypergraph,overlap_hnodes)

        overlap_nodes = [hyper_map[node] for node in overlap_hnodes]
        overlap_edges = [hyper_map[edge] for edge in overlap_hedges]
        boundary_edges = [hyper_map[edge] for edge in boundary_hedges]

        #Setup subproblem graphs
        subproblem_graph = OptiGraph()
        subproblem_graph.modelnodes = overlap_nodes
        subproblem_graph.linkedges = overlap_edges

        push!(subproblem_graphs,subproblem_graph)
        push!(boundary_linkedges_list,boundary_edges)
    end

    return subproblem_graphs,boundary_linkedges_list
end

#Links can be formulated as either constraints or penalties
#BUG: Nodes are not showing up in correct subgraphs
function _assign_links(subgraphs,subgraph_boundary_edges,input_primal_links,input_dual_links)
    subgraph_primal_links = []
    subgraph_dual_links = []

    for (i,edge_set) in enumerate(subgraph_boundary_edges)
        primal_links = LinkConstraintRef[]
        dual_links = LinkConstraintRef[]

        for edge in edge_set
            linkrefs = edge.linkrefs
            for linkref in linkrefs
                if linkref in input_primal_links
                    push!(primal_links,linkref)
                elseif linkref in input_dual_links
                    push!(dual_links,linkref)
                else
                    #use attached node to assign link
                    target_node = constraint_object(linkref).attached_node
                    #target_node = edge.nodes[end]
                    if !(target_node in all_nodes(subgraphs[i]))
                        push!(dual_links,linkref)  #send primal info to target node, receive dual info
                    else
                        push!(primal_links,linkref)   #receive primal info at target node, send back dual info
                    end
                end
            end
        end
        push!(subgraph_primal_links,primal_links)
        push!(subgraph_dual_links,dual_links)
    end
    return subgraph_primal_links,subgraph_dual_links
end

#Find boundary edges of expanded subgraphs
#BUG: the boundary nodes don't show up in the subgraph
function _find_boundaries(optigraph::OptiGraph,subgraphs::Vector{OptiGraph})
    boundary_linkedges_list = []
    for subgraph in subgraphs
        subnodes = all_nodes(subgraph)
        #overlap_nodes = Plasmo.neighborhood(optigraph,subnodes,overlap)
        boundary_edges = Plasmo.incident_edges(optigraph,subnodes)
        push!(boundary_linkedges_list,boundary_edges)
    end
    return boundary_linkedges_list
end
