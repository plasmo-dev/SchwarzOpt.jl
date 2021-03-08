mutable struct SchwarzData
    x_in_indices::Dict{OptiGraph,Vector{Int64}}
    l_in_indices::Dict{OptiGraph,Vector{Int64}}
    x_out_indices::Dict{OptiGraph,Vector{Int64}}
    l_out_indices::Dict{OptiGraph,Vector{Int64}}

    node_subgraph_map::Dict
    subproblem_subgraph_map::Dict
    primal_links::Vector
    dual_links::Vector
    ext_var_index_map::Dict
end

mutable struct SchwarzSolution
    x_out_vals::Dict{OptiGraph,Vector{Float64}}
    l_out_vals::Dict{OptiGraph,Vector{Float64}}
    x_vals::Vector{Float64}
    l_vals::Vector{Float64}
end
function SchwarzSolution(subgraphs::Vector{OptiGraph})
    x_vals = Vector{Float64}()  #Primal values for communication
    l_vals = Vector{Float64}()  #Dual values for communication
    x_out_vals = Dict{OptiGraph,Vector{Float64}}()
    l_out_vals = Dict{OptiGraph,Vector{Float64}}()
    for sub in subgraphs
        x_out_vals[sub] = Float64[]
        l_out_vals[sub] = Float64[]
    end
    return SchwarzSolution(x_vals,l_vals,x_out_vals,l_out_vals)
end

function _initialize_schwarz!(optigraph::OptiGraph,subgraphs::Vector{OptiGraph},primal_links::Vector,dual_links::Vector)
    #MAP OPTINODES TO ORIGNAL SUBGRAPHS
    node_subgraph_map = Dict()
    original_subgraphs = getsubgraphs(optigraph)
    for sub in original_subgraphs
        for node in all_nodes(sub)
            node_subgraph_map[node] = sub
        end
    end
    expanded_subgraph_map = Dict()
    for i = 1:length(subgraphs)
        expanded_subgraph = subgraphs[i]
        expanded_subgraph_map[original_subgraphs[i]] = expanded_subgraph
    end

    #FIND BOUNDARIES AND ASSIGN LINKS
    subgraph_boundary_edges = _find_boundaries(optigraph,subgraphs)
    primal_links,dual_links = _assign_links(subgraphs,subgraph_boundary_edges,primal_links,dual_links)

    #_setup_subproblems!(subproblems,x_out_vals,l_out_vals,sub_optimizer)
    #INITIALIZE SUBPROBLEM DATA
    for subgraph in subgraphs
        #Primal data
        subgraph.ext[:x_in] = VariableRef[]                        #variables into subproblem
        subgraph.ext[:x_out] = VariableRef[]                       #variables out of subproblem
        subgraph.ext[:varmap] = Dict{VariableRef,VariableRef}()    #map subgraph variables to created subproblem variables
        subgraph.ext[:added_constraints] = ConstraintRef[]         #link constraints added to this subproblem

        #Dual data
        subgraph.ext[:l_in] = LinkConstraint[]                     #duals into subproblem
        subgraph.ext[:l_out] = ConstraintRef[]                     #duals out of subproblem
        subgraph.ext[:lmap] = Dict{LinkConstraint,GenericAffExpr{Float64,VariableRef}}()  #map linkconstraints to objective penalty terms

        #Original objective function
        obj = objective_function(subproblem)
        subgraph.ext[:original_objective] = obj

        #setup optimizer
        JuMP.set_optimizer(subgraph,optimizer)
    end

    #SETUP SUBPROBLEMS FOR ALGORITHM
    #indices
    x_in_indices = Dict{OptiGraph,Vector{Int64}}()   #map subproblem to its x_in_indices
    l_in_indices = Dict{OptiGraph,Vector{Int64}}()
    x_out_indices = Dict{OptiGraph,Vector{Int64}}()
    l_out_indices = Dict{OptiGraph,Vector{Int64}}()

    #initialize solution values
    schwarz_sol = SchwarzSolution(subgraphs)
    for i = 1:length(subgraphs)
        subgraph = subgraphs[i]
        for linkref in dual_links[i]
            link = constraint_object(linkref)
            edge = owner_model(linkref)

            #Initialize dual values
            if !(haskey(edge.dual_values,link))
                edge.dual_values[link] = 0.0
            end

            #INPUTS
            if edge.dual_values[link] != nothing
                push!(schwarz_sol.l_vals,edge.dual_values[link])
            else
                push!(schwarz_sol.l_vals,0.0)  #initial dual value
            end

            idx = length(schwarz_sol.l_vals)
            push!(l_in_indices[subgraph],idx)                                         #add index to subproblem l inputs

            #OUTPUTS
            vars = collect(keys(link.func.terms))                                     #variables in linkconsstraint
            external_vars = [var for var in vars if !(var in all_variables(subgraph)]
            external_node = getnode(external_vars[end])                               #get the target node for this link

            original_subgraph = node_subgraph_map[external_node]                        #get the restricted subgraph
            target_subgraph = expanded_subgraph_map[original_subgraph]          #the subproblem that owns this link_constraint
            #target_subproblem,target_map = target_subproblem_map
            push!(l_out_indices[target_subgraph],idx)                               #add index to target subproblem outputs

            push!(target_subgraph.ext[:l_out],link)   #map linkconstraint to target subproblem dual outputs
            _add_subproblem_dual_penalty!(subgraph,link,l_vals[idx])  #Add penalty to subproblem
        end

        for linkref in primal_links[i]
            link = constraint_object(linkref)
            vars = collect(keys(link.func.terms))                                       #variables in linkconsstraint
            external_vars = [var for var in vars if !(var in all_variables(subgraph))]  #variables not part of this subgraph

            for ext_var in external_vars
                #if external variable hasn't been counted yet
                if !(ext_var in keys(ext_var_index_map))
                    JuMP.start_value(ext_var) == nothing ? start = 1 : start = JuMP.start_value(ext_var)
                    push!(x_vals,start)                                                 #increment x_vals
                    idx = length(x_vals)                                                #get index
                    ext_var_index_map[ext_var] = idx                                    #map index for external variable

                    external_node = getnode(ext_var)                                    #get the node for this external variable
                    original_subgraph = node_subgraph_map[external_node]                  #get the restricted subgraph
                    source_subgraph = expanded_subgraph_map[original_subgraph]    #get the subproblem that owns this external variable

                    #OUTPUTS
                    push!(x_out_indices[source_subgraph],idx)                           #add index to source subproblem outputs
                    push!(source_subgraph.ext[:x_out],ext_var)     #map external variable to source problem primal outputs
                else
                    idx = ext_var_index_map[ext_var]
                end

                #If this subproblem needs to make a copy of the external variable
                if !(ext_var in keys(subgraph.ext[:varmap]))
                    #we don't always want to make a copy if this subproblem already has a copy of this variable
                    copyvar = _add_subproblem_var!(subproblem,ext_var)  #create local variable on subgraph subproblem

                    #INPUTS
                    push!(x_in_indices[subgraph],idx)
                end
            end
            #mapping = merge(submap.varmap,subgraph.ext[:varmap])
            _add_subproblem_constraint!(subgraph,link)  #subgraph.ext[:varmap]             #Add link constraint to the subproblem
        end
    end

end

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
                    target_node = collect(edge.nodes)[end]
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

function _find_boundaries(optigraph::OptiGraph,subgraphs::Vector{OptiGraph})

    boundary_linkedges_list = []
    hypergraph,hyper_map = Plasmo.graph_backend_data(optigraph)

    for subgraph in subgraphs
        subnodes = all_nodes(subgraph)
        hypernodes = [hyper_map[node] for node in subnodes]
        overlap_hnodes = hypernodes #Plasmo.neighborhood(hypergraph,hypernodes,overlap)
        boundary_hedges = Plasmo.incident_edges(hypergraph,overlap_hnodes)
        boundary_edges = [hyper_map[edge] for edge in boundary_hedges]
        push!(boundary_linkedges_list,boundary_edges)
    end

    return boundary_linkedges_list
end

#TODO: Figure out how to update these.  It might just be possible to fix certain variables in the optigraph
function _add_subproblem_var!(subgraph::OptiGraph,ext_var::VariableRef)
    ghost_node = @optinode(subgraph)
    newvar = @variable(ghost_node)
    JuMP.set_name(newvar,name(ext_var)*"ghost")
    JuMP.start_value(ext_var) == nothing ? start = 1 : start = JuMP.start_value(ext_var)
    JuMP.fix(newvar,start)     #we will fix this to a new value for each iteration
    subgraph.ext[:varmap][ext_var] = newvar
    push!(subproblem.ext[:x_in],newvar)
    return newvar
end

function _add_subproblem_constraint!(subgraph::OptiGraph,con::LinkConstraint)
    #Add local linkconstraint for this subproblem
    varmap = subgraph.ext[:varmap]
    new_con = Plasmo._copy_constraint(con,varmap)
    conref = JuMP.add_constraint(subproblem,new_con)
    push!(subproblem.ext[:added_constraints], conref)
    return conref
end

function _add_subproblem_dual_penalty!(subgraph::OptiGraph,con::LinkConstraint,l_start::Float64)
    push!(subproblem.ext[:l_in],con)

    vars = collect(keys(con.func.terms))
    local_vars = [var for var in vars if var all_variables(subgraph)]

    con_func = con.func  #need to create func containing only local vars
    terms = con_func.terms
    new_terms = OrderedDict([(var_ref,coeff) for (var_ref,coeff) in terms if var_ref in local_vars])
    new_func = JuMP.GenericAffExpr{Float64,JuMP.VariableRef}()
    new_func.terms = new_terms
    new_func.constant = con_func.constant

    subproblem.ext[:lmap][con] = new_func

    return new_func
end



# function _add_subproblem_var!(subproblem::OptiNode,ext_var::VariableRef)
#     newvar = @variable(subproblem)
#     JuMP.set_name(newvar,name(ext_var)*"ghost")
#     JuMP.start_value(ext_var) == nothing ? start = 1 : start = JuMP.start_value(ext_var)
#     JuMP.fix(newvar,start)     #we will fix this to a new value for each iteration
#     subproblem.ext[:varmap][ext_var] = newvar
#     push!(subproblem.ext[:x_in],newvar)
#     return newvar
# end



# function _add_subproblem_dual_penalty!(subproblem::OptiNode,mapping::Dict,con::LinkConstraint,l_start::Float64)
#
#     push!(subproblem.ext[:l_in],con)
#
#     vars = collect(keys(con.func.terms))
#     local_vars = [var for var in vars if var in keys(mapping)]
#
#     con_func = con.func  #need to create func containing only local vars
#     terms = con_func.terms
#     new_terms = OrderedDict([(mapping[var_ref],coeff) for (var_ref,coeff) in terms if var_ref in local_vars])
#     new_func = JuMP.GenericAffExpr{Float64,JuMP.VariableRef}()
#     new_func.terms = new_terms
#     new_func.constant = con_func.constant
#
#     subproblem.ext[:lmap][con] = new_func
#
#     return new_func
# end

# #NOTE: I don't think this is needed anymore with the new Plasmo.jl updates.  Variables and duals should be updated automatically after optimizing an optigraph
# function _update_graph_solution!(optigraph,subproblem_subgraph_map,node_subgraph_map)
#     #update node variable values
#     for subgraph in getsubgraphs(optigraph)
#         #subproblem,sub_map = subproblem_subgraph_map[subgraph]
#         # for node in all_nodes(subgraph)
#         #     for var in JuMP.all_variables(node)
#         #         node.variable_values[var] = value(sub_map[var])
#         #     end
#         # end
#         # for node in all_nodes(subgraph)
#         #     for var in JuMP.all_variables(node)
#         #         node.variable_values[var] = value(sub_map[var])
#         #     end
#         # end
#     end
#     #update link duals using owning subgraph
#     #NOTE: this doesn't really make sense at the boundaries
#     for edge in optigraph.linkedges
#         for linkcon in getlinkconstraints(edge)
#             node_end = getnode(collect(keys(linkcon.func.terms))[1])
#             subgraph = node_subgraph_map[node_end]
#             subproblem,sub_map = subproblem_subgraph_map[subgraph]
#             dual_value = dual(sub_map.linkconstraintmap[linkcon])
#             edge.dual_values[linkcon] = dual_value
#         end
#     end
#     return nothing
# end

# #Create schwarz subproblems
# function _setup_subproblems!(subproblems,x_out_vals,l_out_vals,optimizer)
#     for (subproblem,ref_map) in subproblems
#         # x_out_vals[subproblem] = Float64[]
#         # l_out_vals[subproblem] = Float64[]
#         #Primal data
#         subproblem.ext[:x_in] = VariableRef[]                        #variables into subproblem
#         subproblem.ext[:x_out] = VariableRef[]                       #variables out of subproblem
#         subproblem.ext[:varmap] = Dict{VariableRef,VariableRef}()    #map subgraph variables to created subproblem variables
#         subproblem.ext[:added_constraints] = ConstraintRef[]         #link constraints added to this subproblem
#
#         #Dual data
#         subproblem.ext[:l_in] = LinkConstraint[]                     #duals into subproblem
#         subproblem.ext[:l_out] = ConstraintRef[]                     #duals out of subproblem
#         subproblem.ext[:lmap] = Dict{LinkConstraint,GenericAffExpr{Float64,VariableRef}}()  #map linkconstraints to objective penalty terms
#
#         obj = objective_function(subproblem)
#         subproblem.ext[:original_objective] = obj
#
#         JuMP.set_optimizer(getmodel(subproblem),optimizer)
#     end
#     return nothing
# end

# function _modify_subproblems!(optigraph,subproblems,x_vals,x_in_indices,x_out_indices,l_vals,l_in_indices,l_out_indices,node_subgraph_map,subproblem_subgraph_map,
#     subgraph_in_edges,subgraph_out_edges,ext_var_index_map)

# function _modify_subproblems!(optigraph,subproblems,x_vals,x_in_indices,x_out_indices,l_vals,l_in_indices,l_out_indices,node_subgraph_map,subproblem_subgraph_map,
#     primal_links,dual_links,ext_var_index_map)
#
#     for i = 1:length(subproblems)
#         subproblem,submap = subproblems[i]  #get the corresponding subproblem
#
#         for linkref in dual_links[i]
#             edge = linkref.linkedge
#             #link = constraint_object(linkref) #TODO
#             link = linkref.linkedge.linkconstraints[linkref.idx]
#
#             if !(haskey(edge.dual_values,link))
#                 edge.dual_values[link] = 0.0
#             end
#
#             #INPUTS
#             if edge.dual_values[link] != nothing
#                 push!(l_vals,edge.dual_values[link])
#             else
#                 push!(l_vals,0.0)  #initial dual value
#             end
#
#             idx = length(l_vals)
#             push!(l_in_indices[subproblem],idx)                                      #add index to subproblem l inputs
#
#             #OUTPUTS
#             #NOTE: This doesn't make sense anymore without edge directions
#             #external_node = getnodes(link)[end]
#
#             vars = collect(keys(link.func.terms))                                   #variables in linkconsstraint
#             external_vars = [var for var in vars if !(var in keys(submap.varmap))]
#             external_node = getnode(external_vars[end])
#
#                                                                                       #get the target node for this link
#             target_subgraph = node_subgraph_map[external_node]                        #get the restricted subgraph
#             target_subproblem_map = subproblem_subgraph_map[target_subgraph]          #the subproblem that owns this link_constraint
#             target_subproblem,target_map = target_subproblem_map
#             push!(l_out_indices[target_subproblem],idx)                               #add index to target subproblem outputs
#             push!(target_subproblem.ext[:l_out],target_map.linkconstraintmap[link])   #map linkconstraint to target subproblem dual outputs
#             _add_subproblem_dual_penalty!(subproblem,submap.varmap,link,l_vals[idx])  #Add penalty to subproblem
#         end
#
#         for linkref in primal_links[i]
#             edge = linkref.linkedge
#             #link = constraint_object(linkref)
#             link = linkref.linkedge.linkconstraints[linkref.idx]
#             vars = collect(keys(link.func.terms))                                   #variables in linkconsstraint
#             external_vars = [var for var in vars if !(var in keys(submap.varmap))]  #variables not part of this subproblem
#
#             for ext_var in external_vars
#                 #if external variable hasn't been counted yet
#                 if !(ext_var in keys(ext_var_index_map))
#                     JuMP.start_value(ext_var) == nothing ? start = 1 : start = JuMP.start_value(ext_var)
#                     push!(x_vals,start)                                                 #increment x_vals
#                     idx = length(x_vals)                                                #get index
#                     ext_var_index_map[ext_var] = idx                                    #map index for external variable
#
#                     external_node = getnode(ext_var)                                    #get the node for this external variable
#                     source_subgraph = node_subgraph_map[external_node]                  #get the restricted subgraph
#                     source_subproblem_map = subproblem_subgraph_map[source_subgraph]    #get the subproblem that owns this external variable
#                     source_subproblem,source_map = source_subproblem_map
#
#                     #OUTPUTS
#                     push!(x_out_indices[source_subproblem],idx)                         #add index to source subproblem outputs
#                     push!(source_subproblem.ext[:x_out],source_map.varmap[ext_var])     #map external variable to source problem primal outputs
#                 else
#                     idx = ext_var_index_map[ext_var]
#                 end
#
#                 #If this subproblem needs to make a copy of the external variable
#                 if !(ext_var in keys(subproblem.ext[:varmap]))
#                     #we don't always want to make a copy if this subproblem already has a copy of this variable
#                     copyvar = _add_subproblem_var!(subproblem,ext_var)                  #create local variable on subproblem
#                     #INPUTS
#                     push!(x_in_indices[subproblem],idx)
#                 end
#             end
#             mapping = merge(submap.varmap,subproblem.ext[:varmap])
#             _add_subproblem_constraint!(subproblem,mapping,link)               #Add link constraint to the subproblem
#         end
#     end
#
#     return nothing
# end
