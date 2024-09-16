@kwdef mutable struct Timers
    start_time::Float64 = 0.0
    initialize_time::Float64 = 0.0
    eval_objective_time::Float64 = 0.0
    eval_primal_feasibility_time::Float64 = 0.0
    eval_dual_feasibility_time::Float64 = 0.0
    update_subproblem_time::Float64 = 0.0
    solve_subproblem_time::Float64 = 0.0
    total_time::Float64 = 0.0
end

@kwdef mutable struct Options
    tolerance::Float64=1e-4
    max_iterations::Int64=100
    mu::Float64=1.0
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    graph::OptiGraph
    subproblem_graphs::Vector{OptiGraph}
    subproblem_optimizer::Any

    # algorithm options
    options::Options

    # mapping information
    node_subgraph_map::Dict         # map optinodes to "owning" subgraph
    edge_subgraph_map::Dict
    expanded_subgraph_map::Dict
    incident_variable_map::Dict     # map variable indices in optigraphs to ghost variables
    incident_constraint_map::Dict
    subproblem_links::Vector        # link constraints that require dual updates

    # current status
    initialized::Bool
    status::MOI.TerminationStatusCode
    err_pr::Float64
    err_du::Float64
    objective_value::Union{Nothing,Float64}

    # current primal and dual values that get communicated to subproblems
    x_vals::Vector{Float64}
    l_vals::Vector{Float64}

    # algorithm iteration outputs
    iteration::Int64
    primal_error_iters::Vector{Float64}
    dual_error_iters::Vector{Float64}
    objective_iters::Vector{Float64}
    solve_time::Float64

    # timers
    timers::Timers

    # partial initializer
    function Optimizer()
        optimizer = new()
    
        # mapping information
        optimizer.node_subgraph_map = Dict()
        optimizer.edge_subgraph_map = Dict()
        optimizer.expanded_subgraph_map = Dict()
        # optimizer.subproblem_links = Vector{Vector{Plasmo.EdgeConstraintRef}}()
        optimizer.incident_variable_map = OrderedDict()
        optimizer.incident_constraint_map = OrderedDict()

        # current status
        optimizer.initialized = false
        optimizer.status = MOI.OPTIMIZE_NOT_CALLED
        optimizer.err_pr = Inf
        optimizer.err_du = Inf
        optimizer.objective_value = nothing
        optimizer.x_vals = Vector{Float64}()  # primal values for communication
        optimizer.l_vals = Vector{Float64}()  # dual values for communication
        
        # iteration output
        optimizer.iteration = 0
        optimizer.primal_error_iters = Float64[]
        optimizer.dual_error_iters = Float64[]
        optimizer.objective_iters = Float64[]

        # timers
        optimizer.timers = Timers()
        return optimizer
    end
end

function Optimizer(
    graph::OptiGraph,
    subproblem_graphs::Vector{OptiGraph};
    subproblem_optimizer=nothing,
    tolerance=1e-6,
    max_iterations=100,
    mu=1.0,
)
    optimizer = Optimizer()
    optimizer.graph = graph
    optimizer.options = Options(tolerance, max_iterations, mu)
    optimizer.subproblem_graphs = subproblem_graphs
    optimizer.subproblem_optimizer = subproblem_optimizer
    return optimizer
end

# function Optimizer(
#     graph::OptiGraph,
#     partition::Plasmo.Partition,
#     tolerance::Float64=1e-4,
#     max_iterations::Int64=100,
#     mu::Float64=1.0
# )
#     optimizer = Optimizer()
#     optimizer.graph = graph
#     optimizer.options = Options(tolerance, max_iterations, mu)

# end

function _check_valid_subproblems(optimizer::Optimizer)
    graph = optimizer.graph
    subgraphs = optimizer.subproblem_graphs
    subgraph_nodes = union(all_nodes.(subgraphs)...)

    # check number of nodes makes sense
    length(subgraph_nodes) == num_nodes(graph) || 
        error(
        """
        Invalid subgraph problems given to optimizer. The number of nodes in 
        subgraphs does not match the number of nodes in the optigraph.
        """
        )

    all((node) -> node in all_nodes(graph), subgraph_nodes) ||
        error(
            """
            Invalid subgraphs given to optimizer.  At least one provided subgraph
            constrains an optinode that is not in the optigraph.
            """
        )

    #TODO: check for overlap of at least 1.
    # _check_overlap

    #TODO: raise warning for non-contiguous partitions
    # _check_partitions

    return true
end

function _build_restricted_objective(subgraph::OptiGraph)
    return sum(objective_function(node) for node in all_nodes(subgraph))
end

# in cases where the total graph objective is not separable, we can use duplicate variables.
function _parse_non_separable_objective(objective_function)
end

function initialize!(optimizer::Optimizer)
    if optimizer.subproblem_optimizer == nothing
        error(
            "No optimizer set for the subproblems.  Please provide an optimizer constructor to use to solve subproblem optigraphs",
        )
    end

    return optimizer.timers.initialize_time = @elapsed begin
        _check_valid_subgraphs(optimizer)
        graph = optimizer.graph
        overlap_subgraphs = optimizer.subproblem_graphs
        n_subproblems = length(overlap_subgraphs)

        # map optinodes to their original subgraphs
        original_subgraphs = local_subgraphs(graph)
        for subgraph in original_subgraphs
            for node in all_nodes(subgraph)
                optimizer.node_subgraph_map[node] = subgraph
            end
            for edge in all_edges(subgraph)
                optimizer.edge_subgraph_map[edge] = subgraph
            end
        end

        # reformulate objective function if necessary
        # node_objectives = parse_graph_objective_to_nodes(graph)

        # map subgraphs to their expanded subgraphs
        for i in 1:length(overlap_subgraphs)
            original_subgraph = original_subgraphs[i]
            expanded_subgraph = overlap_subgraphs[i]

            # check subgraphs are valid
            @assert intersect(
                all_nodes(original_subgraph), 
                all_nodes(expanded_subgraph)
            ) == all_nodes(original_subgraph)

            optimizer.expanded_subgraph_map[original_subgraph] = expanded_subgraph
            expanded_subgraph.ext[:restricted_subgraph] = original_subgraph

            # TODO: make sure we actually have the node objectives
            expanded_subgraph.ext[:restricted_objective] = _build_restricted_objective(
                original_subgraph
            )
        end

        # gather the boundary links for each subproblem
        subgraph_boundary_edges = _find_boundaries(graph, overlap_subgraphs)
        subproblem_links = _gather_links(
            overlap_subgraphs, subgraph_boundary_edges
        )
        
        # optimizer.subproblem_links = subproblem_links
        @assert length(subproblem_links) == n_subproblems

        ########################################################
        # INITIALIZE SUBPROBLEM DATA
        ########################################################
        for (i, subgraph) in enumerate(overlap_subgraphs)
            # initialize primal-dual maps
            subgraph.ext[:x_map] = Dict{Int64,NodeVariableRef}()
            subgraph.ext[:l_map] = Dict{Int64,EdgeConstraintRef}()
            subgraph.ext[:incident_links] = subproblem_links[i]

            Plasmo.set_to_node_objectives(subgraph)
            subgraph.ext[:original_objective] = objective_function(subgraph)
        end

        #######################################################
        # INITIALIZE SUBPROBLEMS
        #######################################################
        for i in 1:length(optimizer.subproblem_graphs)
            subproblem_graph = optimizer.subproblem_graphs[i]
            sub_links = subproblem_graph.ext[:incident_links]

            #map each subproblem to the primals and duals it is responsible for updating
            for link_reference in sub_links
                # build primal information map
                link = constraint_object(link_reference)
                vars = collect(keys(link.func.terms))
                incident_variables = setdiff(vars, all_variables(subproblem_graph))

                for incident_variable in incident_variables
                    # if incident variable hasn't been counted yet, add to array
                    if !(incident_variable in keys(optimizer.incident_variable_map))
                        if JuMP.start_value(incident_variable) == nothing
                            start = 0
                        else
                            start = JuMP.start_value(incident_variable)
                        end
                        push!(optimizer.x_vals, start)
                        idx = length(optimizer.x_vals)
                        optimizer.incident_variable_map[incident_variable] = idx

                        incident_node = optinode(incident_variable)                                   # get the node for this external variable
                        original_subgraph = optimizer.node_subgraph_map[incident_node]                # get the restricted subgraph
                        exp_subgraph = optimizer.expanded_subgraph_map[original_subgraph]             # get the subproblem that owns this external variable
                        exp_subgraph.ext[:x_map][idx] = incident_variable                             # this graph is used to calculate this variable
                    end
                end

                # build dual information map
                if !(link_reference in keys(optimizer.incident_constraint_map))
                    dual_start = 0.0
                    push!(optimizer.l_vals, dual_start)
                    idx = length(optimizer.l_vals)
                    optimizer.incident_constraint_map[link_reference] = idx
                    link = constraint_object(link_reference)
                    vars = collect(keys(link.func.terms))

                    # if the edge is contained inside a subgraph
                    if !(link_reference in linkconstraints(graph))
                        original_subgraph = optimizer.edge_subgraph_map[link_reference.optiedge]
                    else
                        # the edge needs to be absorbed into one of the subgraphs
                        node = optinodes(link_reference)[1]
                        original_subgraph = optimizer.node_subgraph_map[node]
                    end
                    exp_subgraph = optimizer.expanded_subgraph_map[original_subgraph]                  #the subproblem that "owns" this link_constraint
                    #exp_subgraph.ext[:l_idx_map][link_reference] = idx
                    exp_subgraph.ext[:l_map][idx] = link_reference                                     # this graph is used to calculate this dual
                end
            end
        end
        optimizer.loaded = true
    end
end

function _update_subproblem!(optimizer, subproblem_graph::OptiGraph)
    set_objective_function(subproblem_graph, subproblem_graph.ext[:original_objective])
    for link in subproblem_graph.ext[:incident_links]
        _formulate_penalty!(optimizer, subproblem_graph, link)
    end
    return nothing
end

"""
    Formulate penalty term for each subproblem
"""
function _formulate_penalty!(
    optimizer::Optimizer, 
    subproblem_graph::OptiGraph, 
    link_reference::ConstraintRef
)
    link = constraint_object(link_reference)
    link_variables = collect(keys(link.func.terms))
    local_link_variables = intersect(link_variables, all_variables(subproblem_graph))

    external_variables = setdiff(link_variables, local_link_variables)
    external_var_inds = [optimizer.incident_variable_map[var] for var in external_variables]
    external_values = [optimizer.x_vals[x_idx] for x_idx in external_var_inds]

    local_link_coeffs = [link.func.terms[var] for var in local_link_variables]
    external_coeffs = [link.func.terms[var] for var in external_variables]

    l_idx = optimizer.incident_constraint_map[link_reference]
    l_link = optimizer.l_vals[l_idx]

    rhs = link.set.value

    penalty =
        local_link_coeffs' * local_link_variables + external_coeffs' * external_values - rhs
    augmented_penalty = penalty^2

    # TODO: use `add_to_expression!` since it should be more efficient. the following lines do not work however.
    #JuMP.add_to_expression!(objective_function(subproblem_graph), -1, l_link*penalty)
    #JuMP.add_to_expression!(objective_function(subproblem_graph), 0.5*optimizer.mu, augmented_penalty)
    set_objective_function(
        subproblem_graph,
        objective_function(subproblem_graph) - l_link * penalty +
        0.5 * optimizer.mu * augmented_penalty,
    )
    return nothing
end

function _do_iteration(subproblem_graph::OptiGraph)
    Plasmo.optimize!(subproblem_graph)
    term_status = termination_status(subproblem_graph)
    !(
        term_status in [
            MOI.TerminationStatusCode(4),
            MOI.TerminationStatusCode(1),
            MOI.TerminationStatusCode(10),
        ]
    ) && @warn("Suboptimal solution detected for subproblem with status $term_status")

    # primal and dual variables to update
    x_sub = subproblem_graph.ext[:x_map]
    l_sub = subproblem_graph.ext[:l_map]

    # primal and dual values for this subproblem
    xk = Dict(key => value(subproblem_graph, val) for (key, val) in x_sub)
    lk = Dict(key => dual(subproblem_graph, val) for (key, val) in l_sub)

    return xk, lk
end

function _calculate_objective_value(optimizer::Optimizer)
    # TODO: this should just be the graph objective value
    # evaluate the objective function with the current primal
    return sum(
        value(optimizer.subproblem_graphs[i].ext[:restricted_objective]) for
        i in 1:length(optimizer.subproblem_graphs)
    )
end

#TODO: cache mappings ahead of time.  This is bottlenecking.
function _calculate_primal_feasibility(optimizer::Optimizer)
    linkrefs = local_link_constraints(optimizer.graph)
    prf = []
    for linkref in linkrefs
        val = 0
        linkcon = constraint_object(linkref)

        # TODO: update for nonlinear links
        terms = linkcon.func.terms
        for (term, coeff) in terms
            node = get_node(term)
            graph = optimizer.node_subgraph_map[node]
            subproblem_graph = optimizer.expanded_subgraph_map[graph]
            val += coeff * value(subproblem_graph, term)
        end
        push!(prf, val - linkcon.set.value)
    end
    return prf
end

#NOTE: Need at least an overlap of one
function _calculate_dual_feasibility(optimizer::Optimizer)
    graph = optimizer.graph
    linkrefs = local_link_constraints(optimizer.graph)
    duf = []
    for linkref in linkrefs
        duals = []
        linkcon = constraint_object(linkref)

        graphs = []
        for node in collect_nodes(linkcon)
            graph = optimizer.node_subgraph_map[node]
            subproblem_graph = optimizer.expanded_subgraph_map[graph]
            push!(graphs, subproblem_graph)
        end

        #check each subproblem's dual value for this linkconstraint
        graphs = unique(graphs)
        for subproblem_graph in graphs
            l_val = dual(subproblem_graph, linkref)
            push!(duals, l_val)
        end

        # dual residual between subproblems
        dual_res = abs(maximum(duals) - minimum(duals))
        push!(duf, dual_res)
    end
    return duf
end

function run_algorithm!(optimizer::Optimizer)
    if !optimizer.loaded
        println("Initializing SchwarzOpt...")
        _initialize_optimizer!(optimizer)
    end

    println("###########################################################")
    println("Optimizing with SchwarzOpt v0.2.0 using $(Threads.nthreads()) threads")
    println("###########################################################")
    println()
    println("Number of variables: $(num_all_variables(optimizer.graph))")
    println(
        "Number of constraints: $(num_all_constraints(optimizer.graph) + num_all_linkconstraints(optimizer.graph))",
    )
    println("Number of subproblems: $(length(optimizer.subproblem_graphs))")
    println("Overlap: ")
    println("Subproblem variables:   $(num_all_variables.(optimizer.subproblem_graphs))")
    println("Subproblem constraints: $(num_all_constraints.(optimizer.subproblem_graphs))")
    println()

    optimizer.timers = Timers()
    optimizer.timers.start_time = time()

    for subgraph in optimizer.subproblem_graphs
        JuMP.set_optimizer(subgraph, optimizer.subproblem_optimizer)
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
            for subproblem_graph in optimizer.subproblem_graphs
                _update_subproblem!(optimizer, subproblem_graph)
            end
        end

        optimizer.timers.solve_subproblem_time += @elapsed begin
            Threads.@threads for subproblem_graph in optimizer.subproblem_graphs
                #Returns primal and dual information we need to communicate to other subproblems
                xk, lk = _do_iteration(subproblem_graph)

                #Updates primal and dual information for other subproblems.
                for (idx, val) in xk
                    optimizer.x_vals[idx] = val
                end
                for (idx, val) in lk
                    optimizer.l_vals[idx] = val
                end
            end
        end

        #Evaluate residuals
        optimizer.timers.eval_primal_feasibility_time += @elapsed prf = _calculate_primal_feasibility(
            optimizer
        )
        optimizer.timers.eval_dual_feasibility_time += @elapsed duf = _calculate_dual_feasibility(
            optimizer
        )
        optimizer.err_pr = norm(prf[:], Inf)
        optimizer.err_du = norm(duf[:], Inf)

        optimizer.timers.eval_objective_time += @elapsed optimizer.objective_value = _calculate_objective_value(
            optimizer
        )
        push!(optimizer.primal_error_iters, optimizer.err_pr)
        push!(optimizer.dual_error_iters, optimizer.err_du)
        push!(optimizer.objective_iters, optimizer.objective_value)

        #Print iteration
        if optimizer.iteration % 20 == 0 || optimizer.iteration == 1
            @printf "%4s | %8s | %8s | %8s" "Iter" "Obj" "Prf" "Duf\n"
        end
        @printf(
            "%4i | %7.2e | %7.2e | %7.2e\n",
            optimizer.iteration,
            optimizer.objective_value,
            optimizer.err_pr,
            optimizer.err_du
        )

        optimizer.timers.update_subproblem_time += @elapsed begin
            for subproblem in optimizer.subproblem_graphs
                JuMP.set_start_value.(
                    Ref(subproblem),
                    all_variables(subproblem),
                    value.(Ref(subproblem), all_variables(subproblem)),
                )
            end
        end
    end

    #Point variables to restricted solutions
    for (node, subgraph) in optimizer.node_subgraph_map
        subproblem_graph = optimizer.expanded_subgraph_map[subgraph]
        backend(node).last_solution_id = subproblem_graph.id
    end

    optimizer.timers.total_time = time() - optimizer.timers.start_time
    if optimizer.status != MOI.ITERATION_LIMIT
        optimizer.status = termination_status(optimizer.subproblem_graphs[1])
    end
    # optimizer.graph.optimizer = optimizer
    println()
    println("Number of Iterations: ", optimizer.iteration)
    @printf "%8s | %8s | %8s" "Obj" "Prf" "Duf\n"
    @printf(
        "%7.2e | %7.2e | %7.2e\n",
        _calculate_objective_value(optimizer),
        optimizer.err_pr,
        optimizer.err_du
    )
    println()
    println("Time spent in subproblems: ", optimizer.timers.solve_subproblem_time)
    println("Solution Time: ", optimizer.timers.total_time)
    return println("EXIT: SchwarzOpt Finished with status: ", optimizer.status)
end

# """
#     SchwarzOpt.optimize!(graph::OptiGraph;kwargs...)

# Optimize an optigraph with overlapping schwarz decomposition.
# """
# function run_algorithm!(
#     graph::OptiGraph;,
#     subproblem_optimizer=Ipopt.Optimizer,
#     overlap=1,
#     max_iterations=50,
#     mu=100.0,
# )
#     if length(subgraphs) == 0
#         println("Optimizing with overlap of $overlap")
#         subgraphs = expand(graph, subgraphs, overlap)
#     else
#         println("Optimizing with user provided overlap")
#     end

#     optimizer = Optimizer(
#         graph, subgraphs; subproblem_optimizer=subproblem_optimizer, max_iterations=max_iterations, mu=mu
#     )

#     run_algorithm!(optimizer)
#     return optimizer
# end
