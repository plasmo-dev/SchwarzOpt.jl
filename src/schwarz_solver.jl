
function schwarz_solve(optigraph::OptiGraph,subgraphs::Vector{OptiGraph};
    sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"tol" => 1e-8,"print_level" => 0),
    max_iterations = 100,
    tolerance = 1e-3,
    primal_links = LinkConstraintRef[],
    dual_links = LinkConstraintRef[])

     #new optigraph references used for subproblems
    println("Initializing subproblems...")
    subproblems = [OptiGraph(all_nodes(subgraph),all_edges(subgraph)) for subgraph in subgraphs]
    schwarz_data,schwarz_sol = _initalize_schwarz!(optigraph,subproblems,sub_optimizer,primal_links,dual_links)
    #schwarz_sol = SchwarzSolution(subgraphs)

    #Get original linkconstraints that connect subgraphs
    original_linkcons = getlinkconstraints(optigraph)
    graph_obj = objective_function(optigraph)

    #START THE ALGORITHM
    println("Start algorithm...")
    err_pr = Inf
    err_du = Inf
    #err_save = []
    optigraph.obj_dict[:err_save] = []
    optigraph.obj_dict[:objective_iters] = []

    # Initialize primal and dual information for each subproblem
    iteration = 0

    @printf "Running Schwarz algorithm with: %2i threads\n" Threads.nthreads()
    println()
    while err_pr > tolerance || err_du > tolerance
        iteration += 1
        if iteration > max_iterations
            return MOI.ITERATION_LIMIT
        end

        #Do iteration for each subproblem
        # Threads.@threads for (subproblem,submap) in subproblems  #each subproblem is a OptiNode
        Threads.@threads for subgraph in subgraphs
        #for (subproblem,submap) in subproblems  #each subproblem is a OptiNode
            #Get primal and dual information for this subproblem
            x_in_inds = schwarz_data.x_in_indices[subgraph]
            l_in_inds = schwarz_data.l_in_indices[subgraph]

            x_in = schwarz_sol.x_vals[x_in_inds]
            l_in = schwarz_sol.l_vals[l_in_inds]

            xk,lk = do_iteration(subproblem,x_in,l_in)  #return primal and dual information we need to communicate to other subproblems

            #Update primal and dual information for other subproblems.  Use restriction to make sure we grab the right values
            schwarz_sol.x_out_vals[subgraph] = xk    #Update primal info we need to communicate to other subproblems
            schwarz_sol.l_out_vals[subgraph] = lk    #Update dual info we need to communicate to other subproblems
        end

        #UPDATE x_vals and l_vals for the subproblems to send information to
        for (subgraph,vals) in x_out_vals
            x_out_inds = schwarz_data.x_out_indices[subgraph]
            schwarz_sol.x_vals[x_out_inds] .= vals
        end
        for (subgraph,vals) in l_out_vals
            l_out_inds = schwarz_data.l_out_indices[subgraph]
            schwarz_sol.l_vals[l_out_inds] .= vals
        end

        #Update graph solution only for nodes in original subgraphs.  Used to calculate residual
        _update_graph_solution!(optigraph,subproblem_subgraph_map,node_subgraph_map)

        #Check primal and dual feasibility.  Evaluate original partitioned link constraints (i.e. do the restriction) based on direction
        # prf = [nodevalue(linkcon.func) - linkcon.set.value for linkcon in original_linkcons]
        prf = [value(linkcon.func) - linkcon.set.value for linkcon in original_linkcons]
        duf = []
        for linkcon in getlinkconstraints(optigraph)
            #linkcon = optigraph.linkconstraints[linkref.idx]
            nodes = getnodes(linkcon)
            lambdas = []
            for node in nodes
                # subgraph = node_subgraph_map[node]
                # subproblem = subproblem_subgraph_map[subgraph]
                #subproblem,sub_map = subproblem_subgraph_map[subgraph]
                l_val = dual(linkcon)
                push!(lambdas,l_val)
            end
            #TODO: Correctly calculate dual resiudal.  This only works for simple overlaps
            dual_res = lambdas[1] - lambdas[2]
            push!(duf,dual_res)
        end


        for (subproblem,submap) in subproblems
            JuMP.set_start_value.(all_variables(subproblem),value.(all_variables(subproblem)))
        end

        err_pr = norm(prf[:],Inf)
        err_du = norm(duf[:],Inf)
        # obj = nodevalue(graph_obj)
        obj = value(graph_obj)

        if iteration % 20 == 0 || iteration == 1
            @printf "%4s | %8s | %8s | %8s" "Iter" "Obj" "Prf" "Duf\n"
        end
        @printf("%4i | %7.2e | %7.2e | %7.2e\n",iteration,obj,err_pr,err_du)
        push!(optigraph.obj_dict[:err_save],[err_pr err_du])
        push!(optigraph.obj_dict[:objective_iters],obj)
    end

    #_clean_changes() #TODO: Remove changes we made to optinodes and optigraphs


    #TODO: Return MOI Status Code
    return MOI.OPTIMAL
end

function schwarz_solve(optigraph::OptiGraph,overlap::Int64;
    sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"tol" => 1e-8,"print_level" => 0),
    max_iterations = 100,
    tolerance = 1e-6,
    primal_links = [],
    dual_links = [])

    has_subgraphs(optigraph) || error("OptiGraph $optigraph does not contains any subgraph structure.
    Consider creating partitions using a graph partitioner. See Documentation for details on how to do this.")

    subgraphs = getsubgraphs(optigraph)

    println("Expanding subgraph domains...")
    expanded_subs = expand.(Ref(optigraph),subgraphs,Ref(overlap))

    status = schwarz_solve(optigraph,expanded_subs;sub_optimizer = sub_optimizer,max_iterations = max_iterations,tolerance = tolerance,
    primal_links = primal_links,dual_links = dual_links)

    return status
end


function do_iteration(node::OptiNode,x_in::Vector{Float64},l_in::Vector{Float64})

    update_values!(node,x_in,l_in)  #update x_in and l_in


    optimize!(node) #optimize a subproblem

    #Update start point for next iteration
    term_status = termination_status(node)
    !(term_status in [MOI.TerminationStatusCode(4),MOI.TerminationStatusCode(1),MOI.TerminationStatusCode(10)]) && @warn("Suboptimal solution detected for problem $node with status $term_status")
    has_values(getmodel(node)) || error("Could not obtain values for problem $node with status $term_status")
    #TODO: Also check subproblem status.

    #return values we need to communicate to
    x_out = node.ext[:x_out]    #primal variables to communicate along in edges
    l_out = node.ext[:l_out]    #dual variables to communicate along out edges

    #get variable and dual values
    xk = value.(x_out)
    lk = dual.(l_out)

    return xk, lk
end

function update_values!(node::OptiNode,x_in::Vector{Float64},l_in::Vector{Float64})
    #set primal values
    for (i,var) in enumerate(node.ext[:x_in]) #make sure x_in and node.ext[:x_in] match up
        fix(var,x_in[i])
    end
    obj_original = node.ext[:original_objective]
    obj = obj_original

    funcs = GenericAffExpr{Float64,VariableRef}[]
    for (i,con) in enumerate(node.ext[:l_in])
        func = l_in[i]*node.ext[:lmap][con]
        push!(funcs,func)
    end
    set_objective_function(node,obj_original - sum(funcs)) #sum(l_in[i]*funcs[i] for i = 1:length(funcs)))
    return nothing
end


# #Initialize subproblem vectors
# x_in_indices = Dict{OptiNode,Vector{Int64}}()   #map subproblem to its x_in_indices
# l_in_indices = Dict{OptiNode,Vector{Int64}}()
# x_out_indices = Dict{OptiNode,Vector{Int64}}()
# l_out_indices = Dict{OptiNode,Vector{Int64}}()
# for sub in subproblems
#     x_in_indices[sub] = Int64[]
#     x_out_indices[sub] = Int64[]
#     l_in_indices[sub] = Int64[]
#     l_out_indices[sub] = Int64[]
# end
# #Subproblem outputs
# x_out_vals = Dict{OptiNode,Vector{Float64}}()
# l_out_vals = Dict{OptiNode,Vector{Float64}}()

# for edge in optigraph.linkedges
#     #linkrefs = edge.linkconstraints
#     linkcons = getlinkconstraints(edge)
#
#     #for linkref in linkrefs
#     for linkcon in linkcons
#         #linkcon = optigraph.linkconstraints[linkref.idx]
#         nodes = getnodes(linkcon)
#         lambdas = []
#         for node in nodes
#             subgraph = node_subgraph_map[node]
#             subproblem,sub_map = subproblem_subgraph_map[subgraph]
#             l_val = dual(sub_map.linkconstraintmap[linkcon])
#             push!(lambdas,l_val)
#         end
#         #TODO: Correctly calculate dual resiudal.  This only works for simple overlaps
#         dual_res = lambdas[1] - lambdas[2]
#         push!(duf,dual_res)
#     end
#
# end

#TODO: Check that subgraphs cover the entire optigraph
# x_vals = Vector{Float64}()  #Primal values for communication
# l_vals = Vector{Float64}()  #Dual values for communication
#ext_var_index_map = Dict{VariableRef,Int64}()  #map boundary variables to indices

# #TODO Build up Schwarz data
# function SchwarzData(subgraphs::Vector{OptiGraph})
#     x_in_indices = Dict{OptiGraph,Vector{Int64}}()   #map subproblem to its x_in_indices
#     l_in_indices = Dict{OptiGraph,Vector{Int64}}()
#     x_out_indices = Dict{OptiGraph,Vector{Int64}}()
#     l_out_indices = Dict{OptiGraph,Vector{Int64}}()
#     return SchwarzData(x_in_indices,l_in_indices,x_out_indices,l_out_indices)
# end
#graph_obj = sum(objective_function(node) for node in all_nodes(optigraph))

#Map nodes to their orignal subgraphs
# original_subgraphs = getsubgraphs(optigraph)
# node_subgraph_map = Dict()
# for sub in original_subgraphs
#     for node in all_nodes(sub)
#         node_subgraph_map[node] = sub
#     end
# end
#
# #FiND BOUNDARIES AND ASSIGN LINKS
# subgraph_boundary_edges = _find_boundaries(optigraph,subgraphs)
# primal_links,dual_links = _assign_links(subgraphs,subgraph_boundary_edges,primal_links,dual_links)

#Map subproblems to their original (non-expanded) subgraphs
#TODO: Try just optimizing without aggregating
# subproblems = aggregate.(subgraphs)
# subproblems = subgraphs #expanded subgraphs

# subproblem_subgraph_map = Dict()
# for i = 1:length(subproblems)
#     subproblem = subproblems[i]
#     subproblem_subgraph_map[original_subgraphs[i]] = subproblem
# end

#INITIALIZE SUBPROBLEM DATA


#_setup_subproblems!(subproblems,x_out_vals,l_out_vals,sub_optimizer)

#_modify_subproblems!(optigraph,subproblems,schwarz_data)



#TODO: Condense inputs into data structure
# _modify_subproblems!(optigraph,subproblems,x_vals,x_in_indices,x_out_indices,l_vals,l_in_indices,l_out_indices,node_subgraph_map,subproblem_subgraph_map,
# primal_links,dual_links,ext_var_index_map)
