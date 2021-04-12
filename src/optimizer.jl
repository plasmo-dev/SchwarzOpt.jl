mutable struct SchwarzOptimizer #<: Plasmo.OptiGraphOptimizer
    graph::OptiGraph
    subproblems::Vector{OptiGraph}

    x_in_indices::Dict{OptiGraph,Vector{Int64}}   #primals into subproblem
    l_in_indices::Dict{OptiGraph,Vector{Int64}}   #duals into subproblem
    x_out_indices::Dict{OptiGraph,Vector{Int64}}  #primals out of subproblem
    l_out_indices::Dict{OptiGraph,Vector{Int64}}  #duals out of subproblem

    node_subgraph_map::Dict                       #map optinodes to owning subgraph
    primal_links::Vector                          #link constraints that do primal updates
    dual_links::Vector                            #link constraints that do dual updates
    ghost_var_index_map::Dict                     #map variables indices to ghost variables

    x_out_vals::Dict{OptiGraph,Vector{Float64}}   #primal values out
    l_out_vals::Dict{OptiGraph,Vector{Float64}}   #dual values out
    x_vals::Vector{Float64}                       #current primal values
    l_vals::Vector{Float64}                       #current dual values

    status::MOI.TerminationStatusCode
    err_pr::Float64
    err_du::Float64
    objective_value::Float64

    primal_error_iters::Vector{Float64}
    dual_error_iters::Vector{Float64}
    objective_iters::Vector{Float64}

    function SchwarzOptimizer()
        optimizer = new()

        optimizer.x_in_indices = Dict{OptiGraph,Vector{Int64}}()
        optimizer.l_in_indices = Dict{OptiGraph,Vector{Int64}}()
        optimizer.x_out_indices = Dict{OptiGraph,Vector{Int64}}()
        optimizer.l_out_indices = Dict{OptiGraph,Vector{Int64}}()

        optimizer.node_subgraph_map = Dict()
        optimizer.primal_links = LinkConstraintRef[]
        optimizer.dual_links = LinkConstraintRef[]
        optimizer.ghost_var_index_map = Dict()

        optimizer.x_vals = Vector{Float64}()  #Primal values for communication
        optimizer.l_vals = Vector{Float64}()  #Dual values for communication
        optimizer.x_out_vals = Dict{OptiGraph,Vector{Float64}}()
        optimizer.l_out_vals = Dict{OptiGraph,Vector{Float64}}()

        optimizer.err_pr = Inf
        optimizer.err_du = Inf
        optimizer.objective_value = 0
        optimizer.status = MOI.OPTIMIZE_NOT_CALLED
        optimizer.primal_error_iters = Float64[]
        optimizer.dual_error_iters = Float64[]
        optimizer.objective_iters = Float64[]

        return optimizer
    end
end

function SchwarzOptimizer(graph::OptiGraph,subgraphs::Vector{OptiGraph})
    optimizer = SchwarzOptimizer()
    optimizer.graph = graph
    optimizer.subproblems = subgraphs
    for sub in subgraphs
        optimizer.x_out_vals[sub] = Float64[]
        optimizer.l_out_vals[sub] = Float64[]
    end
    return optimizer
end

function _initialize_optimizer!(optigraph::OptiGraph,
    subgraphs::Vector{OptiGraph},
    sub_optimizer::Any,
    primal_links::Vector,
    dual_links::Vector)

    #MAP OPTINODES TO ORIGNAL SUBGRAPHS
    node_subgraph_map = Dict()
    original_subgraphs = getsubgraphs(optigraph)
    for sub in original_subgraphs
        for node in all_nodes(sub)
            node_subgraph_map[node] = sub
        end
    end

    #Map subproblems to subgraphs
    expanded_subgraph_map = Dict()
    for i = 1:length(subgraphs)
        expanded_subgraph = subgraphs[i]
        expanded_subgraph_map[original_subgraphs[i]] = expanded_subgraph
    end

    #FIND SUBGRAPH BOUNDARIES AND ASSIGN LINKS AS EITHER PRIMAL OR DUAL
    subgraph_boundary_edges = _find_boundaries(optigraph,subgraphs)
    primal_links,dual_links = _assign_links(subgraphs,subgraph_boundary_edges,primal_links,dual_links)

    #INITIALIZE SUBPROBLEM DATA
    for subgraph in subgraphs
        #Primal data
        subgraph.ext[:x_in] = VariableRef[]                              #variables into subproblem
        subgraph.ext[:x_out] = VariableRef[]                             #variables out of subproblem
        subgraph.ext[:ghost_varmap] = Dict{VariableRef,VariableRef}()    #map external variables to local variables
        subgraph.ext[:ghost_constraints] = ConstraintRef[]               #link constraints added to this subproblem

        #Dual data
        subgraph.ext[:l_in] = LinkConstraint[]                     #duals into subproblem
        subgraph.ext[:l_out] = ConstraintRef[]                     #duals out of subproblem
        subgraph.ext[:dual_map] = Dict{LinkConstraint,GenericAffExpr{Float64,VariableRef}}()  #map linkconstraints to penalty terms

        #Original objective function
        obj = objective_function(subgraph)
        subgraph.ext[:original_objective] = obj

        #setup optimizer
        JuMP.set_optimizer(subgraph,sub_optimizer)
    end

    #INITIALIZE
    optimizer = SchwarzOptimizer(graph,subgraphs)
    for i = 1:length(subgraphs)
        subgraph = subgraphs[i]

        #DUAL LINKS
        for linkref in dual_links[i]
            link = constraint_object(linkref)
            edge = owner_model(linkref)

            #Initialize dual values
            if !(haskey(edge.dual_values,link))
                edge.dual_values[link] = 0.0
            end

            #INPUTS to subproblem
            if edge.dual_values[link] != nothing
                push!(optimizer.l_vals,edge.dual_values[link])
            else
                push!(optimizer.l_vals,0.0)  #initial dual value
            end

            idx = length(optimizer.l_vals)
            push!(optimizer.l_in_indices[subgraph],idx)                                         #add index to subproblem l inputs

            #OUTPUTS from subproblem
            vars = collect(keys(link.func.terms))                                     #variables in linkconsstraint
            external_vars = [var for var in vars if !(var in all_variables(subgraph))]
            external_node = getnode(external_vars[end])    #TODO: use attached_node                           #get the target node for this link

            original_subgraph = node_subgraph_map[external_node]                        #get the restricted subgraph
            target_subgraph = expanded_subgraph_map[original_subgraph]                  #the subproblem that "owns" this link_constraint

            push!(optimizer.l_out_indices[target_subgraph],idx)                               #add index to target subproblem outputs
            push!(target_subgraph.ext[:l_out],link)   #map linkconstraint to target subproblem dual outputs
            _add_subproblem_dual_penalty!(subgraph,link,optimizer.l_vals[idx])  #Add penalty to subproblem objective
        end

        #PRIMAL LINKS
        for linkref in primal_links[i]
            link = constraint_object(linkref)
            vars = collect(keys(link.func.terms))                                       #variables in linkconsstraint
            external_vars = [var for var in vars if !(var in all_variables(subgraph))]  #variables not part of this subgraph

            for ext_var in external_vars
                if !(ext_var in keys(optimizer.ghost_var_index_map))     #if external variable hasn't been counted yet
                    JuMP.start_value(ext_var) == nothing ? start = 1 : start = JuMP.start_value(ext_var)
                    push!(optimizer.x_vals,start)                                                 #increment x_vals
                    idx = length(optimizer.x_vals)                                                #get index
                    optimizer.ghost_var_index_map[ext_var] = idx                                    #map index for external variable

                    external_node = getnode(ext_var)                                    #get the node for this external variable
                    original_subgraph = optimizer.node_subgraph_map[external_node]                  #get the restricted subgraph
                    source_subgraph = expanded_subgraph_map[original_subgraph]    #get the subproblem that owns this external variable

                    #OUTPUTS
                    push!(optimizer.x_out_indices[source_subgraph],idx)                           #add index to source subproblem outputs
                    push!(source_subgraph.ext[:x_out],ext_var)     #map external variable to source problem primal outputs
                else
                    idx = optimizer.ghost_var_index_map[ext_var]
                end


                if !(ext_var in keys(subgraph.ext[:ghost_varmap])) #If this subproblem needs to make a local copy of the external variable
                    _add_subproblem_variable!(subgraph,ext_var)  #create local variable on subgraph subproblem
                    #INPUTS
                    push!(optimizer.x_in_indices[subgraph],idx)
                end
            end
            _add_subproblem_constraint!(subgraph,linkref)  #subgraph.ext[:varmap]             #Add link constraint to the subproblem
        end
    end

end

function _add_subproblem_variable!(subgraph::OptiGraph,external_variable::VariableRef)
    ghost_node = @optinode(subgraph)
    ghost_variable = @variable(ghost_node)
    JuMP.set_name(ghost_var,name(external_variable)*"ghost")
    JuMP.start_value(external_variable) == nothing ? start = 1 : start = JuMP.start_value(external_variable)
    JuMP.fix(ghost_variable,start) #external_var is fixed each iteration

    subgraph.ext[:ghost_varmap][external_variable] = ghost_variable
    push!(subproblem.ext[:x_in],ghost_variable)

    return nothing
end

#Add link constraint to optigraph connecting to ghost variable
function _add_subproblem_constraint!(subgraph::OptiGraph,linkref::LinkConstraintRef)
    varmap = subgraph.ext[:ghost_varmap]
    link = constraint_object(linkref)
    ghost_link = Plasmo._copy_constraint(link,varmap)
    ghost_linkref = JuMP.add_constraint(subgraph,ghost_link) #this is a linkconstraint
    push!(graph.ext[:ghost_constraints], ghost_linkref)

    return nothing
end

#Set optigraph objective
function _add_subproblem_dual_penalty!(subgraph::OptiGraph,linkref::LinkConstraintRef)
    push!(graph.ext[:l_in],con)

    vars = collect(keys(con.func.terms))
    local_vars = [var for var in vars if var in all_variables(subgraph)]
    link = constraint_object(linkref)

    #Create new function with local variables
    con_func = link.func
    terms = con_func.terms
    new_terms = OrderedDict([(var_ref,coeff) for (var_ref,coeff) in terms if var_ref in local_vars])
    new_func = JuMP.GenericAffExpr{Float64,JuMP.VariableRef}()
    new_func.terms = new_terms
    new_func.constant = con_func.constant

    subgraph.ext[:dual_map][linkref] = new_func

    return nothing
end

function do_iteration(subgraph::OptiGraph,x_in::Vector{Float64},l_in::Vector{Float64})
    update_subproblem!(subgraph,x_in,l_in)  #update x_in and l_in
    optimize!(subgraph) #optimize subproblem.  Should catch any underlying updates

    term_status = termination_status(graph)
    !(term_status in [MOI.TerminationStatusCode(4),MOI.TerminationStatusCode(1),MOI.TerminationStatusCode(10)]) && @warn("Suboptimal solution detected for problem $node with status $term_status")
    has_values(graph) || error("Could not obtain values for problem $node with status $term_status")

    x_out = graph.ext[:x_out]           #primal variables to communicate to input edges
    l_out = graph.ext[:l_out]           #dual variables to communicate to output edges
    xk = value.(Ref(subgraph),x_out)    #grab primals for subproblem
    lk = dual.(Ref(subgraph),l_out)     #grab duals for subproblem
    return xk, lk
end

function update_subproblem!(subgraph::OptiGraph,x_in::Vector{Float64},l_in::Vector{Float64})
    for (i,var) in enumerate(graph.ext[:x_in]) #make sure x_in and node.ext[:x_in] match up
        JuMP.fix(var,x_in[i])
    end
    obj_original = graph.ext[:original_objective]
    obj = obj_original

    #TODO: just update coefficients in constraint
    funcs = GenericAffExpr{Float64,VariableRef}[]
    for (i,con) in enumerate(graph.ext[:l_in])
        func = l_in[i]*graph.ext[:lmap][con]
        push!(funcs,func)
    end
    #Add dual penalties to objective function
    set_objective_function(subgraph,obj_original - sum(funcs)) #sum(l_in[i]*funcs[i] for i = 1:length(funcs)))
    return nothing
end

function optimize!(optimizer::SchwarzOptimizer)
    original_linkcons = getlinkconstraints(optimizer.graph)
    iteration = 0
    while optimizer.err_pr > tolerance || optimizer.err_du > tolerance
        iteration += 1
        if iteration > max_iterations
            optimizer.status = MOI.ITERATION_LIMIT
            break
        end
        #Do iteration for each subproblem
        Threads.@threads for subproblem in optimizer.subproblems
            x_in_inds = optimizer.x_in_indices[subproblem]
            l_in_inds = optimizer.l_in_indices[subproblem]

            x_in = optimizer.x_vals[x_in_inds]
            l_in = optimizer.l_vals[l_in_inds]

            xk,lk = do_iteration(subproblem,x_in,l_in)  #return primal and dual information we need to communicate to other subproblems

            #Update primal and dual information for other subproblems.  Use restriction to make sure we grab the right values
            optimizer.x_out_vals[subproblem] = xk    #Update primal info we need to communicate to other subproblems
            optimizer.l_out_vals[subproblem] = lk    #Update dual info we need to communicate to other subproblems
        end

        #UPDATE x_vals and l_vals for the subproblems to send information to
        for (subproblem,values) in optimizer.x_out_vals
            x_out_inds = optimizer.x_out_indices[subproblem]
            optimizer.x_vals[x_out_inds] .= values
        end
        for (subproblem,values) in optimizer.l_out_vals
            l_out_inds = optimizer.l_out_indices[subproblem]
            optimizer.l_vals[l_out_inds] .= values
        end

        #Evaluate residuals
        prf = [value(linkcon.func) - linkcon.set.value for linkcon in original_linkcons]
        duf = []
        for linkcon in original_linkcons
            lambdas = []
            for node in getnodes(linkcon)
                l_val = dual(linkcon)
                push!(lambdas,l_val)
            end
            #TODO: Correctly calculate dual resiudal.  This only works for simple overlaps. Is this causing issues with larger problems?
            @assert length(lambdas) == 2
            dual_res = lambdas[1] - lambdas[2]
            push!(duf,dual_res)
        end

        #Update start values
        for subproblem in optimizer.subproblems
            JuMP.set_start_value.(all_variables(subproblem),value.(all_variables(subproblem)))
        end

        optimizer.err_pr = norm(prf[:],Inf)
        optimizer.err_du = norm(duf[:],Inf)
        optimizer.obj = value(graph_obj)
        push!(optimizer.primal_error_iters,err_pr)
        push!(optimizer.dual_error_iters,err_du)
        push!(optimizer.objective_iters,obj)

        #Print iteration
        if iteration % 20 == 0 || iteration == 1
            @printf "%4s | %8s | %8s | %8s" "Iter" "Obj" "Prf" "Duf\n"
        end
        @printf("%4i | %7.2e | %7.2e | %7.2e\n",iteration,optimizer.obj,optimizer.err_pr,optimizer.err_du)

    end
end
