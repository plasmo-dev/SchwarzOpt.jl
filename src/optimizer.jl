@kwdef mutable struct Timers
    start_time::Float64 = 0.
    initialize_time::Float64 = 0.
    eval_objective_time::Float64 = 0.
    eval_primal_feasibility_time::Float64 = 0.
    eval_dual_feasibility_time::Float64 = 0.
    solve_subproblem_time::Float64 = 0.
    total_time::Float64 = 0.
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    graph::OptiGraph                              #Subgraphs should not have any overlap and should cover all the nodes
    subproblem_graphs::Vector{OptiGraph}          #These are expanded subgraphs (i.e. subgraphs with overlap)

    sub_optimizer::Any
    tolerance::Float64
    max_iterations::Int64
    timers::Timers

    x_in_indices::Dict{OptiGraph,Vector{Int64}}   #primals into subproblem
    l_in_indices::Dict{OptiGraph,Vector{Int64}}   #duals into subproblem
    x_out_indices::Dict{OptiGraph,Vector{Int64}}  #primals out of subproblem
    l_out_indices::Dict{OptiGraph,Vector{Int64}}  #duals out of subproblem

    node_subgraph_map::Dict                       #map optinodes to "owning" subgraph
    expanded_subgraph_map::Dict
    primal_links::Vector                          #link constraints that do primal updates
    dual_links::Vector                            #link constraints that do dual updates
    incident_variable_map::Dict                   #map variable indices in optigraphs to ghost (copy) variables

    x_out_vals::Dict{OptiGraph,Vector{Float64}}   #primal values out
    l_out_vals::Dict{OptiGraph,Vector{Float64}}   #dual values out
    x_vals::Vector{Float64}                       #current primal values that get communicated
    l_vals::Vector{Float64}                       #current dual values that get communicated

    status::MOI.TerminationStatusCode
    err_pr::Float64
    err_du::Float64
    objective_value::Float64

    iteration::Int64
    primal_error_iters::Vector{Float64}
    dual_error_iters::Vector{Float64}
    objective_iters::Vector{Float64}
    solve_time::Float64

    plasmo_optimizer_hook::Function
    loaded::Bool

    function Optimizer()
        optimizer = new()

        optimizer.subproblem_graphs = Vector{OptiGraph}()
        optimizer.x_in_indices = Dict{OptiGraph,Vector{Int64}}()
        optimizer.l_in_indices = Dict{OptiGraph,Vector{Int64}}()
        optimizer.x_out_indices = Dict{OptiGraph,Vector{Int64}}()
        optimizer.l_out_indices = Dict{OptiGraph,Vector{Int64}}()

        optimizer.node_subgraph_map = Dict()
        optimizer.expanded_subgraph_map = Dict()
        optimizer.primal_links = LinkConstraintRef[]
        optimizer.dual_links = LinkConstraintRef[]
        optimizer.incident_variable_map = Dict()

        optimizer.x_vals = Vector{Float64}()  #Primal values for communication
        optimizer.l_vals = Vector{Float64}()  #Dual values for communication
        optimizer.x_out_vals = Dict{OptiGraph,Vector{Float64}}()
        optimizer.l_out_vals = Dict{OptiGraph,Vector{Float64}}()

        optimizer.err_pr = Inf
        optimizer.err_du = Inf
        optimizer.objective_value = 0
        optimizer.status = MOI.OPTIMIZE_NOT_CALLED
        optimizer.iteration = 0
        optimizer.primal_error_iters = Float64[]
        optimizer.dual_error_iters = Float64[]
        optimizer.objective_iters = Float64[]

        optimizer.plasmo_optimizer_hook = SchwarzOpt.optimize!
        optimizer.loaded = false

        return optimizer
    end
end

function Optimizer(graph::OptiGraph,subgraphs::Vector{OptiGraph};
    sub_optimizer = nothing,
    primal_links::Vector,
    dual_links::Vector,
    tolerance = 1e-6,
    max_iterations = 100)

    optimizer = Optimizer()
    optimizer.graph = graph
    optimizer.sub_optimizer = sub_optimizer
    optimizer.tolerance = tolerance
    optimizer.max_iterations = max_iterations

    graph,
    subgraphs,
    sub_optimizer,
    primal_links,
    dual_links,
    tolerance,
    max_iterations)
    return optimizer
end

function _check_valid_subgraphs(graph::Optigraph,subgraphs::Vector{OptiGraph})
    @assert union(all_nodes.(subgraphs)) == all_nodes(graph)

    #TODO: check for overlap of at least 1.

    #TODO: check for non-contiguous partitions and raise warning
    return true
end

function _initialize_optimizer!(optimizer::Optimizer)
    if optimizer.sub_optimizer == nothing
        error("No optimizer set for the subproblems.  Please provide an optimizer constructor to use to solve subproblem optigraphs")
    end

    optimizer.timers.initialize_time = @elapsed begin
        graph = optimizer.graph
        overlap_subgraphs = optimizer.subproblem_graphs
        primal_links = optimizer.primal_links
        dual_links = optimizer.dual_links

        _check_valid_subgraphs(graph,overlap_subgraphs)

        for subgraph in overlap_subgraphs
            sub = Plasmo.optigraph_reference(subgraph)
            push!(optimizer.subproblem_graphs,sub)
            optimizer.x_out_vals[sub] = Float64[]
            optimizer.l_out_vals[sub] = Float64[]
            optimizer.x_in_indices[sub] = Int64[]
            optimizer.l_in_indices[sub] = Int64[]
            optimizer.x_out_indices[sub] = Int64[]
            optimizer.l_out_indices[sub] = Int64[]
        end
        n_subproblems = length(optimizer.subproblem_graphs)

        #MAP OPTINODES TO ORIGNAL SUBGRAPHS
        original_subgraphs = getsubgraphs(graph)
        for sub in original_subgraphs
            for node in all_nodes(sub)
                optimizer.node_subgraph_map[node] = sub
            end
        end

        #Map subgraphs to their expanded versions
        for i = 1:length(optimizer.subproblem_graphs)
            original_subgraph = original_subgraphs[i]
            expanded_subgraph = optimizer.subproblem_graphs[i]
            @assert intersect(all_nodes(original_subgraph),all_nodes(expanded_subgraph)) == all_nodes(original_subgraph)
            optimizer.expanded_subgraph_map[original_subgraph] = expanded_subgraph #TODO: make sure these match up
        end

        #FIND SUBGRAPH BOUNDARIES AND ASSIGN LINKS AS EITHER PRIMAL OR DUAL
        subgraph_boundary_edges = _find_boundaries(graph,optimizer.subproblem_graphs)
        primal_links,dual_links = _assign_links(optimizer.subproblem_graphs,subgraph_boundary_edges,primal_links,dual_links)
        optimizer.primal_links = primal_links
        optimizer.dual_links = dual_links
        @assert length(primal_links) == n_subproblems
        @assert length(dual_links) == n_subproblems
        ########################################################
        #INITIALIZE SUBPROBLEM DATA
        ########################################################
        for subgraph in optimizer.subproblem_graphs
            #Primal data
            subgraph.ext[:x_in_map] = Dict{Int64,VariableRef}()                              #copy variables into subproblem
            subgraph.ext[:x_out_map] = Dict{Int64,VariableRef}()                             #variables out of subproblem
            subgraph.ext[:incident_variable_map] = Dict{VariableRef,VariableRef}()           #map incident variables to local variables

            #Dual data
            subgraph.ext[:l_in_map] = Dict{Int64,GenericAffExpr{Float64,VariableRef}}()                      #duals for penalty
            subgraph.ext[:l_out_map] = Dict{Int64,LinkConstraintRef}()                     #duals from linkconstraint

            #Original objective function
            JuMP.set_objective(subgraph,MOI.MIN_SENSE,sum(objective_function(node) for node in all_nodes(subgraph)))
            obj = objective_function(subgraph)
            subgraph.ext[:original_objective] = obj
        end

        #IDEA TODO:pre-allocate vectors based on dual and primal links
        #inspect to figure out dual links, incident variables, and source and target variables
        #######################################################
        #INITIALIZE SUBPROBLEMS
        #######################################################
        for i = 1:length(optimizer.subproblem_graphs)
            subproblem_graph = optimizer.subproblem_graphs[i]
            subproblem_dual_links = dual_links[i]
            subproblem_primal_links = primal_links[i]

            #DUAL LINKS
            #check in/out for each link
            for link_reference in subproblem_dual_links
                #INPUTS to subproblem
                dual_start = 0
                push!(optimizer.l_vals,dual_start)
                idx = length(optimizer.l_vals)
                push!(optimizer.l_in_indices[subproblem_graph],idx)                                #add index to subproblem l inputs
                _add_subproblem_dual_penalty!(subproblem_graph,link_reference,idx)   #add penalty to subproblem objective

                #push!(subproblem_graph.ext[:l_in],link_reference)
                #subproblem_graph.ext[:l_in][idx] = link_reference

                #OUTPUTS from subproblem
                link = constraint_object(link_reference)
                vars = collect(keys(link.func.terms))                                     #variables in linkconsstraint
                incident_variables = setdiff(vars,all_variables(subproblem_graph))
                incident_node = Plasmo.attached_node(link)                                         #this is the owning node for the link
                @assert !(incident_node in all_nodes(subproblem_graph))
                original_subgraph = optimizer.node_subgraph_map[incident_node]                        #get the restricted subgraph
                target_subgraph = optimizer.expanded_subgraph_map[original_subgraph]                     #the subproblem that "owns" this link_constraint
                push!(optimizer.l_out_indices[target_subgraph],idx)                            #add index to target subproblem outputs
                target_subgraph.ext[:l_out_map][idx] = link_reference
            end

            #PRIMAL LINKS
            #check in/out for each link
            for link_reference in subproblem_primal_links
                link = constraint_object(link_reference)
                vars = collect(keys(link.func.terms))

                local_variables = intersect(vars,all_variables(subproblem_graph))
                for var in local_variables
                    subproblem_graph.ext[:incident_variable_map][var] = var
                end

                incident_variables = setdiff(vars,all_variables(subproblem_graph))
                for incident_variable in incident_variables
                    if !(incident_variable in keys(optimizer.incident_variable_map))                  #if incident variable hasn't been counted yet, create a new one
                        JuMP.start_value(incident_variable) == nothing ? start = 0 : start = JuMP.start_value(incident_variable)
                        push!(optimizer.x_vals,start)                                                 #increment x_vals

                        idx = length(optimizer.x_vals)                                                #get index
                        optimizer.incident_variable_map[incident_variable] = idx                      #map index for external variable

                        incident_node = getnode(incident_variable)                                    #get the node for this external variable
                        original_subgraph = optimizer.node_subgraph_map[incident_node]                #get the restricted subgraph
                        source_subgraph = optimizer.expanded_subgraph_map[original_subgraph]                    #get the subproblem that owns this external variable

                        #OUTPUTS
                        push!(optimizer.x_out_indices[source_subgraph],idx)                           #add index to source subproblem outputs
                        #push!(source_subgraph.ext[:x_out],incident_variable)                          #map incident variable to source problem primal outputs
                        source_subgraph.ext[:x_out_map][idx] = incident_variable
                    else
                        idx = optimizer.incident_variable_map[incident_variable]
                    end
                    #If this subproblem needs to make a local copy of the incident variable
                    if !(incident_variable in keys(subproblem_graph.ext[:incident_variable_map]))
                        _add_subproblem_variable!(subproblem_graph,incident_variable,idx)
                        push!(optimizer.x_in_indices[subproblem_graph],idx)
                    end
                end
                _add_subproblem_constraint!(subproblem_graph,link_reference)                                    #Add link constraint to the subproblem. This problem "owns" the constraint
            end
        end
        optimizer.loaded = true
    end
end

function _add_subproblem_variable!(subproblem_graph::OptiGraph,incident_variable::VariableRef,idx::Int64)
    JuMP.start_value(incident_variable) == nothing ? start = 0 : start = JuMP.start_value(incident_variable)
    copy_node = @optinode(subproblem_graph)
    copy_variable = @variable(copy_node,start = start)
    JuMP.set_name(copy_variable,name(incident_variable)*"_copy")
    if JuMP.has_lower_bound(incident_variable)
        JuMP.set_lower_bound(copy_variable,lower_bound(incident_variable))
    end
    if JuMP.has_upper_bound(incident_variable)
        JuMP.set_upper_bound(copy_variable,upper_bound(incident_variable))
    end

    #JuMP.fix(copy_variable,start)
    subproblem_graph.ext[:incident_variable_map][incident_variable] = copy_variable
    subproblem_graph.ext[:x_in_map][idx] = copy_variable
    return nothing
end

#Add link constraint to optigraph connecting to copy variable
function _add_subproblem_constraint!(subproblem_graph::OptiGraph,link_reference::LinkConstraintRef)
    varmap = subproblem_graph.ext[:incident_variable_map]
    link = constraint_object(link_reference)
    copy_link = Plasmo._copy_constraint(link,varmap)
    copy_linkref = JuMP.add_constraint(subproblem_graph,copy_link) #this is a linkconstraint
    #push!(graph.ext[:incident_constraints], copy_linkref) #this isn't used anywhere?
    return nothing
end

#Add penalty to subproblem objective
function _add_subproblem_dual_penalty!(subproblem_graph::OptiGraph,link_reference::LinkConstraintRef,idx::Int64)
    link = constraint_object(link_reference)
    link_variables = collect(keys(link.func.terms))
    local_link_variables = intersect(link_variables, all_variables(subproblem_graph))
    penalty = sum(local_link_variables)
    set_objective_function(subproblem_graph,objective_function(subproblem_graph) - penalty)
    subproblem_graph.ext[:l_in_map][idx] = penalty
    return nothing
end

function _do_iteration(subproblem_graph::OptiGraph)
    Plasmo.optimize!(subproblem_graph)

    term_status = termination_status(subproblem_graph)
    !(term_status in [MOI.TerminationStatusCode(4),MOI.TerminationStatusCode(1),MOI.TerminationStatusCode(10)]) && @warn("Suboptimal solution detected for subproblem with status $term_status")
    #has_values(subproblem_graph) || error("Could not obtain values for problem $subproblem_graph with status $term_status")

    x_out = subproblem_graph.ext[:x_out_map]           #primal variables to communicate
    l_out = subproblem_graph.ext[:l_out_map]           #dual variables to communicate

    xk = Dict(key => value(subproblem_graph,val) for (key,val) in x_out)
    lk = Dict(key => dual(subproblem_graph,val) for (key,val) in l_out)
    return xk, lk
end

function _update_subproblem!(subproblem_graph::OptiGraph,x_in_vals::Vector{Float64},l_in_vals::Vector{Float64},x_in_inds::Vector{Int64},l_in_inds::Vector{Int64})
    #Fix primal inputs
    @assert length(x_in_vals) == length(x_in_inds)
    @assert length(l_in_vals) == length(l_in_inds)
    for (i,idx) in enumerate(x_in_inds)
        variable = subproblem_graph.ext[:x_in_map][idx]
        JuMP.fix(variable,x_in_vals[i];force = true) #fix variable for this subproblem.  variable should be the copy made for this subgraph
    end
    for (i,idx) in enumerate(l_in_inds)
        penalty = subproblem_graph.ext[:l_in_map][idx]
        #TODO: just update objective function coefficient in the backend.
        JuMP.set_objective_function(subproblem_graph,subproblem_graph.ext[:original_objective] - l_in_vals[i]*penalty)
    end
    return nothing
end

#TODO: more efficient calculations.  This is bottlenecking
function _calculate_objective_value(optimizer)
    obj_val = 0
    for node in all_nodes(optimizer.graph)
        graph = optimizer.node_subgraph_map[node]
        subproblem_graph = optimizer.expanded_subgraph_map[graph]
        obj_val += objective_value(subproblem_graph,node)
    end
    return obj_val
end

#TODO: figure out these mappings ahead of time.  This is bottlenecking
function _calculate_primal_feasibility(optimizer)
    linkrefs = getlinkconstraints(optimizer.graph)
    prf = []
    for linkref in linkrefs
        val = 0
        linkcon = constraint_object(linkref)
        terms = linkcon.func.terms
        for (term,coeff) in terms
            node = getnode(term)
            graph = optimizer.node_subgraph_map[node]
            subproblem_graph = optimizer.expanded_subgraph_map[graph]
            val += coeff*value(subproblem_graph,term)
        end
        push!(prf,val - linkcon.set.value)
    end
    return prf
end

#NOTE: Need at least overlap of one
function _calculate_dual_feasibility(optimizer)
    linkrefs = getlinkconstraints(optimizer.graph)
    duf = []
    for linkref in linkrefs
        lambdas = []
        linkcon = constraint_object(linkref)
        graphs = []
        for node in getnodes(linkcon)
            graph = optimizer.node_subgraph_map[node]
            subproblem_graph = optimizer.expanded_subgraph_map[graph]
            push!(graphs,subproblem_graph)
        end
        graphs = unique(graphs)
        for subproblem_graph in graphs
            l_val = dual(subproblem_graph,linkref)   #check each subproblem's dual value for this linkconstraint
            push!(lambdas,l_val)
        end
        #@assert length(lambdas) == 2
        dual_res = maximum(diff(lambdas))#lambdas[1] - lambdas[2]
        push!(duf,dual_res)
    end
    return duf
end

function optimize!(optimizer::Optimizer)
    if !optimizer.loaded
        println("Initializing SchwarzOpt...")
        _initialize_optimizer!(optimizer)
    end

    println("###########################################################")
    println("Optimizing with SchwarzOpt v0.1.0 using $(Threads.nthreads()) threads")
    println("###########################################################")
    println()
    println("Number of variables: $(num_all_variables(optimizer.graph))")
    println("Number of constraints: $(num_all_constraints(optimizer.graph) + num_all_linkconstraints(optimizer.graph))")
    println("Number of subproblems: $(length(optimizer.subproblem_graphs))")
    println("Overlap: ")
    println("Subproblem sizes: $(num_all_variables.(optimizer.subproblem_graphs))")
    println()

    #TODO: use the timers
    optimizer.timers = Timers()
    optimizer.timers.start_time = time()

    for subgraph in optimizer.subproblem_graphs
        JuMP.set_optimizer(subgraph,optimizer.sub_optimizer)
    end

    optimizer.err_pr = Inf
    optimizer.err_du = Inf
    optimizer.iteration = 0
    while optimizer.err_pr > optimizer.tolerance || optimizer.err_du > optimizer.tolerance
        optimizer.iteration += 1
        if optimizer.iteration > optimizer.max_iterations
            optimizer.status = MOI.ITERATION_LIMIT
            break
        end

        #Do iteration for each subproblem
        optimizer.timers.update_subproblem_time += @elapsed begin
            if optimizer.iteration > 1 #don't fix variables in first iteration
                for subproblem_graph in optimizer.subproblem_graphs
                    x_in_inds = optimizer.x_in_indices[subproblem_graph]
                    l_in_inds = optimizer.l_in_indices[subproblem_graph]
                    x_in_vals = optimizer.x_vals[x_in_inds]
                    l_in_vals = optimizer.l_vals[l_in_inds]
                    _update_subproblem!(subproblem_graph,x_in_vals,l_in_vals,x_in_inds,l_in_inds)
                end
            end
        end

        optimizer.timers.update_subproblem_time += @elapsed begin
            Threads.@threads for subproblem_graph in optimizer.subproblem_graphs
                #Returns primal and dual information we need to communicate to other subproblems
                xk,lk = _do_iteration(subproblem_graph)

                #Updates primal and dual information for other subproblems.
                for (idx,val) in xk
                    optimizer.x_vals[idx] = val
                end
                for (idx,val) in lk
                    optimizer.l_vals[idx] = val
                end
            end
        end

        #Evaluate residuals
        optimizer.timers.eval_primal_feasibility_time += @elapsed prf = _calculate_primal_feasibility(optimizer)
        optimizer.timers.eval_dual_feasibility_time += @elapsed duf = _calculate_dual_feasibility(optimizer)
        optimizer.err_pr = norm(prf[:],Inf)
        optimizer.err_du = norm(duf[:],Inf)


        #NOTE: This is the wrong way to calculate this
        #TODO: Calculate objective value correctly

        optimizer.timers.eval_objective_time += @elapsed optimizer.objective_value = _calculate_objective_value(optimizer)
        #optimizer.objective_value = value(objective_function(optimizer.graph))
        #optimizer.objective_value = 0

        push!(optimizer.primal_error_iters,optimizer.err_pr)
        push!(optimizer.dual_error_iters,optimizer.err_du)
        push!(optimizer.objective_iters,optimizer.objective_value)

        #Print iteration
        if optimizer.iteration % 20 == 0 || optimizer.iteration == 1
            @printf "%4s | %8s | %8s | %8s" "Iter" "Obj" "Prf" "Duf\n"
        end
        @printf("%4i | %7.2e | %7.2e | %7.2e\n",optimizer.iteration,optimizer.objective_value,optimizer.err_pr,optimizer.err_du)

        optimizer.timers.update_subproblem_time += @elapsed begin
            for subproblem in optimizer.subproblem_graphs
                JuMP.set_start_value.(Ref(subproblem),all_variables(subproblem),value.(Ref(subproblem),all_variables(subproblem)))
            end
        end
    end

    #Point variables to restricted solutions
    for (node,subgraph) in optimizer.node_subgraph_map
        subproblem_graph = optimizer.expanded_subgraph_map[subgraph]
        backend(node).last_solution_id = subproblem_graph.id
    end

    optimizer.timers.total_time =  time() - optimizer.timers.start_time
    optimizer.status = termination_status(optimizer.subproblem_graphs[1])

    println()
    println("Number of Iterations: ",length(optimizer.objective_iters))
    println("Solution Time: ",optimizer.solve_time)
    println("EXIT: SchwarzOpt Finished")
end

#call with SchwarzOpt.optimize!(graph)
"""
    SchwarzOpt.optimize!(graph::OptiGraph;kwargs...)

Optimize an optigraph with overlapping schwarz decomposition.
"""
function optimize!(graph::OptiGraph;
    subgraphs = Plasmo.OptiGraph[],
    overlap = 1,
    primal_links = [],
    dual_links = [])

    #check subgraphs, or apply overlap
    if len(subgraphs) > 0
        optimizer = Optimizer(graph,subgraphs;primal_links = primal_links, dual_links = dual_links)
    elseif Plasmo.has_subgraphs(graph)
        subgraphs = expand(graph,subgraphs,overlap)
        optimizer = Optimizer(graph,subgraphs;primal_links = primal_links, dual_links = dual_links)
    else
        error("Invalid optigraph")
    end

    #set the optimizer for querying attributes on the optigraph
    optigraph.optimizer = optimizer
    optimize!(optimizer)
end
