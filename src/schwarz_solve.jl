# function schwarz_solve(optigraph::OptiGraph,subgraphs::Vector{OptiGraph};
#     sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"tol" => 1e-8,"print_level" => 0),
#     max_iterations = 100,
#     tolerance = 1e-3,
#     primal_links = LinkConstraintRef[],
#     dual_links = LinkConstraintRef[])
#
#      #new optigraph references used for subproblems
#     println("Initializing optimizer...")
#     #Create new optigraphs to use as subproblems
#     subproblems = [OptiGraph(all_nodes(subgraph),all_edges(subgraph)) for subgraph in subgraphs]
#     optimizer = _initalize_optimizer!(optigraph,subproblems,sub_optimizer,primal_links,dual_links)
#
#     #Get original linkconstraints that connect subgraphs
#     original_linkcons = getlinkconstraints(optigraph)
#     graph_obj = objective_function(optigraph)
#     optigraph.obj_dict[:err_save] = []
#     optigraph.obj_dict[:objective_iters] = []
#
#     #START THE ALGORITHM
#     @printf "Running Schwarz algorithm with: %2i threads\n" Threads.nthreads()
#     println()
#     optimize!(optimizer)
#     #_clean_changes() #TODO: Remove changes we made to optinodes and optigraphs
#
#     return nothing
# end
#
# function schwarz_solve(optigraph::OptiGraph,overlap::Int64;
#     sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"tol" => 1e-8,"print_level" => 0),
#     max_iterations = 100,
#     tolerance = 1e-6,
#     primal_links = [],
#     dual_links = [])
#
#     has_subgraphs(optigraph) || error("OptiGraph $optigraph does not contains any subgraph structure.
#     Consider creating partitions using a graph partitioner. See Documentation for details on how to do this.")
#
#     subgraphs = getsubgraphs(optigraph)
#
#     println("Expanding subgraph domains...")
#     expanded_subs = expand.(Ref(optigraph),subgraphs,Ref(overlap))
#
#     status = schwarz_solve(optigraph,expanded_subs;sub_optimizer = sub_optimizer,max_iterations = max_iterations,tolerance = tolerance,
#     primal_links = primal_links,dual_links = dual_links)
#
#     return status
# end
