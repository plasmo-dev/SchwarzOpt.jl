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
    tolerance::Float64=1e-4     # primal and dual tolerance measure
    max_iterations::Int64=1000  # maximum number of iterations
    mu::Float64=1.20            # augmented lagrangian penalty
end

mutable struct Optimizer{GT<:Plasmo.AbstractOptiGraph} <: MOI.AbstractOptimizer
    graph::GT
    subproblem_graphs::Vector{GT}
    subproblem_optimizer::Any

    # algorithm options
    options::Options

    # mapping information
    node_subgraph_map::OrderedDict{OptiNode, GT}
    edge_subgraph_map::OrderedDict{OptiEdge, GT}
    expanded_subgraph_map::OrderedDict{GT, GT}
    incident_variable_map::OrderedDict{Plasmo.NodeVariableRef, Int}
    incident_constraint_map::OrderedDict{Plasmo.EdgeConstraintRef, Int}

    # current status
    initialized::Bool
    status::MOI.TerminationStatusCode
    err_pr::Float64
    err_du::Float64
    objective_value::Union{Nothing, Float64}

    # current primal and dual values
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

    # Inner constructor
    function Optimizer(graph::GT, options::Options) where GT <: Plasmo.AbstractOptiGraph
        subproblem_graphs = Vector{GT}()
        subproblem_optimizer = nothing

        new{GT}(graph,
            subproblem_graphs,
            subproblem_optimizer,
            options,
            OrderedDict{OptiNode, GT}(),
            OrderedDict{OptiEdge, GT}(),
            OrderedDict{GT, GT}(),
            OrderedDict{Plasmo.NodeVariableRef, Int}(),
            OrderedDict{Plasmo.EdgeConstraintRef, Int}(),
            false,
            MOI.OPTIMIZE_NOT_CALLED,
            Inf, 
            Inf, 
            nothing,
            Float64[], 
            Float64[],
            0, 
            Float64[], 
            Float64[], 
            Float64[], 
            0.0,
            Timers()
        )
    end
end

# provide subproblem graphs directly
function Optimizer(
    graph::OptiGraph,
    subproblem_graphs::Vector{OptiGraph}; # overlapping subgraphs
    subproblem_optimizer=nothing,
    tolerance=1e-4,
    max_iterations=100,
    mu=1.0,
)
    options = Options(tolerance, max_iterations, mu)
    optimizer = Optimizer(graph, options)
    optimizer.subproblem_graphs = subproblem_graphs
    optimizer.subproblem_optimizer = subproblem_optimizer
    return optimizer
end

# TODO partition using default Metis implementation
function Optimizer(
    graph::OptiGraph;
    tolerance=1e-4,
    max_iterations=100,
    mu=1.0,
    n_partitions=2
)
    options = Options(tolerance, max_iterations, mu)
    optimizer = Optimizer(graph, options)
    return optimizer
end

# TODO provide a Plasmo.Partition
function Optimizer(
    graph::OptiGraph,
    partition::Plasmo.Partition,
    tolerance::Float64=1e-4,
    max_iterations::Int64=100,
    mu::Float64=1.0
)
    options = Options(tolerance, max_iterations, mu)
    optimizer = Optimizer(graph, options)
    return optimizer
end

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

    # TODO: check for an overlap of at least 1.
    # _check_overlap

    # TODO: raise warning for non-contiguous partitions
    # _check_partitions

    return true
end

function initialize!(optimizer::Optimizer)
    if optimizer.subproblem_optimizer == nothing
        error(
            "No optimizer set for the subproblems.  Please provide an optimizer constructor to use to solve subproblem optigraphs",
        )
    end

    return optimizer.timers.initialize_time = @elapsed begin
        _check_valid_subproblems(optimizer)
        
        graph = optimizer.graph
        n_subproblems = length(optimizer.subproblem_graphs)

        # TODO: reformulate objective function if necessary onto optinodes
        # node_objectives = parse_graph_objective_to_nodes(graph)

        ########################################################
        # SETUP GRAPH MAPPINGS
        ########################################################
        # map optinodes to their original subgraphs
        expanded_subgraphs = optimizer.subproblem_graphs
        original_subgraphs = local_subgraphs(graph)
        for subgraph in original_subgraphs
            for node in all_nodes(subgraph)
                optimizer.node_subgraph_map[node] = subgraph
            end
            for edge in all_edges(subgraph)
                optimizer.edge_subgraph_map[edge] = subgraph
            end
        end

        # map subgraphs to their expanded versions
        for i = 1:n_subproblems
            original_subgraph = original_subgraphs[i]
            expanded_subgraph = expanded_subgraphs[i]

            # check subgraphs are valid
            @assert intersect(
                all_nodes(original_subgraph), 
                all_nodes(expanded_subgraph)
            ) == all_nodes(original_subgraph)

            # map original subgraph to expanded subgraph
            optimizer.expanded_subgraph_map[original_subgraph] = expanded_subgraph

            # map expanded graph to original graph
            expanded_subgraph.ext[:restricted_subgraph] = original_subgraph

            # TODO: make sure the restricted objective is correct
            # expanded_subgraph.ext[:restricted_objective] = objective_function(original_subgraph)
        end

        # gather the boundary links for each subproblem
        projection = hyper_projection(graph)
        subproblem_incident_edges = _find_boundary_edges(projection, expanded_subgraphs)
        subproblem_incident_constraints = _get_boundary_constraints(
            subproblem_incident_edges
        )
        @assert length(subproblem_incident_constraints) == n_subproblems

        ########################################################
        # INITIALIZE SUBPROBLEM DATA
        ########################################################
        for i = 1:n_subproblems
            subproblem = expanded_subgraphs[i]

            # initialize primal-dual maps
            subproblem.ext[:x_map] = Dict{Int64,NodeVariableRef}()
            subproblem.ext[:l_map] = Dict{Int64,ConstraintRef}()

            # add reference to incident link constraints to each subproblem graph
            subproblem.ext[:incident_edges] = subproblem_incident_edges[i]
            subproblem.ext[:incident_constraints] = subproblem_incident_constraints[i]

            # the original objective function to start with on each subproblem graph
            subproblem.ext[:overlap_objective] = objective_function(subproblem)
        end

        #######################################################
        # INITIALIZE SUBPROBLEM MAPPINGS
        #######################################################
        #map each subproblem to the primals and duals it is responsible for updating
        for i = 1:n_subproblems
            subproblem = expanded_subgraphs[i]
            subproblem_links = subproblem.ext[:incident_constraints]
            subproblem_incident_edges = subproblem.ext[:incident_edges]
            for link_edge in subproblem_incident_edges
                linked_variables = all_variables(link_edge)
                incident_variables = setdiff(linked_variables, all_variables(subproblem))
                for incident_variable in incident_variables
                    # if incident variable has not been counted yet, add it to array
                    if !(incident_variable in keys(optimizer.incident_variable_map))
                        if Plasmo.start_value(incident_variable) == nothing
                            start = 0
                        else
                            start = Plasmo.start_value(incident_variable)
                        end
                        push!(optimizer.x_vals, start)
                        idx = length(optimizer.x_vals)
                        optimizer.incident_variable_map[incident_variable] = idx

                        # get the node for this external variable
                        incident_node = get_node(incident_variable)

                        # get the subgraph that contains this node                                   
                        original_subgraph = optimizer.node_subgraph_map[incident_node]

                        # get the subproblem graph that owns this external variable
                        exp_subgraph = optimizer.expanded_subgraph_map[original_subgraph]      

                        # map this subproblem graph to "own" this variable       
                        exp_subgraph.ext[:x_map][idx] = incident_variable                             
                    end
                end

                # build dual information map
                for link_constraint in all_constraints(link_edge)
                    # if link constraint has not been counted yet, add it to array
                    if !(link_constraint in keys(optimizer.incident_constraint_map))
                        if Plasmo.start_value(link_constraint) == nothing
                            dual_start = 0.0
                        else
                            dual_start = Plasmo.start_value(link_constraint)
                        end
                        push!(optimizer.l_vals, dual_start)
                        idx = length(optimizer.l_vals)
                        optimizer.incident_constraint_map[link_constraint] = idx

                        # assign edges to subproblem graphs
                        if !(link_constraint in local_link_constraints(graph))
                            # if the edge is contained inside a subgraph, use that subgraph
                            original_subgraph = optimizer.edge_subgraph_map[link_edge]
                        else
                            # the edge needs to be absorbed into a subgraph for dual calc
                            node = get_node(linked_variables[1])
                            original_subgraph = optimizer.node_subgraph_map[node]
                        end

                        # the subproblem graph that "owns" this link constraint
                        exp_subgraph = optimizer.expanded_subgraph_map[original_subgraph]                  

                        # this subproblem graph is used to calculate this dual
                        exp_subgraph.ext[:l_map][idx] = link_constraint                                     
                    end
                end
            end
        end
        optimizer.initialized = true
    end
end

function _begin_iterations(optimizer::Optimizer)
    optimizer.timers = Timers()
    optimizer.timers.start_time = time()

    # initialize subproblems
    for subgraph in optimizer.subproblem_graphs
        JuMP.set_optimizer(subgraph, optimizer.subproblem_optimizer)
    end

    optimizer.err_pr = Inf
    optimizer.err_du = Inf
    optimizer.iteration = 0

    return nothing
end

function do_iteration(optimizer::Optimizer)
    # do iteration for each subproblem
    optimizer.timers.update_subproblem_time += @elapsed begin
        for subproblem_graph in optimizer.subproblem_graphs
            _update_subproblem(optimizer, subproblem_graph)
        end
    end

    optimizer.timers.solve_subproblem_time += @elapsed begin
        Threads.@threads for subproblem_graph in optimizer.subproblem_graphs
            # returns primal and dual information to communicate to other subproblems
            xk, lk = _solve_subproblem(subproblem_graph)

            #Updates primal and dual information for other subproblems.
            for (idx, x_val) in xk
                optimizer.x_vals[idx] = x_val
            end
            for (idx, l_val) in lk
                optimizer.l_vals[idx] = l_val
            end
        end
    end
    return nothing
end


function _update_subproblem(optimizer::Optimizer, subproblem_graph::OptiGraph)
    set_objective_function(subproblem_graph, subproblem_graph.ext[:original_objective])
    for link in subproblem_graph.ext[:incident_links]
        _formulate_penalty!(optimizer, subproblem_graph, link)
    end
    return nothing
end

"""
    _formulate_penalty!(
        optimizer::Optimizer, 
        subproblem_graph::OptiGraph, 
        link_reference::ConstraintRef
    )

    Formulate penalty term for each subproblem
"""
function _formulate_penalty!(
    optimizer::Optimizer, 
    subproblem_graph::OptiGraph, 
    link_reference::ConstraintRef
)
    # get the link constraint object
    link = constraint_object(link_reference)

    # all variables in linking constraint
    link_variables = collect(keys(link.func.terms))
    local_link_variables = intersect(link_variables, all_variables(subproblem_graph))
    local_link_coeffs = [link.func.terms[var] for var in local_link_variables]

    # variables in other subproblem graphs
    external_variables = setdiff(link_variables, local_link_variables)
    external_coeffs = [link.func.terms[var] for var in external_variables]
    external_var_inds = [optimizer.incident_variable_map[var] for var in external_variables]
    external_values = [optimizer.x_vals[x_idx] for x_idx in external_var_inds]

    # dual value
    l_idx = optimizer.incident_constraint_map[link_reference]
    l_link_value = optimizer.l_vals[l_idx]

    # link rhs
    rhs = link.set.value

    # dual penalty
    penalty =
        local_link_coeffs' * local_link_variables + external_coeffs' * external_values - rhs
    
    # augmented dual penalty
    augmented_penalty = penalty^2

    # TODO: use `add_to_expression!` since it should be more efficient. the following lines do not work however.
    #JuMP.add_to_expression!(objective_function(subproblem_graph), -1, l_link*penalty)
    #JuMP.add_to_expression!(objective_function(subproblem_graph), 0.5*optimizer.mu, augmented_penalty)
    set_objective_function(
        subproblem_graph,
        objective_function(subproblem_graph) - l_link_value * penalty +
        0.5 * optimizer.mu * augmented_penalty,
    )
    return nothing
end

function _solve_subproblem(subproblem_graph::OptiGraph)
    Plasmo.optimize!(subproblem_graph)
    term_status = Plasmo.termination_status(subproblem_graph)
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

### check iteration values

function _calculate_objective_value(optimizer::Optimizer)
    func = Plasmo.objective_function(optimizer.graph)

    # grab variable values from corresponding subproblems
    vars = Plasmo._extract_variables(func)
    var_vals = Dict{NodeVariableRef,Float64}()
    for var in vars
        node = get_node(var)
        subgraph = optimizer.node_subgraph_map[node]
        subproblem_graph = optimizer.expanded_subgraph_map[subgraph]
        var_vals[var] = Plasmo.value(subproblem_graph, var)
    end
    objective_value = Plasmo.value(i -> get(var_vals, i, 0.0), func) 
    optimizer.objective_value = objective_value

    return nothing
    # # TODO: evaluate the total objective function using subgraph solutions
    # # should be same approach as calculating feasibility.
    # return sum(
    #     value(objective_function(subgraph)) for subgraph in local_subgraphs(optimizer.graph)
    # )
    # # return sum(
    # #     value(optimizer.subproblem_graphs[i].ext[:restricted_objective]) for
    # #     i in 1:length(optimizer.subproblem_graphs)
    # # )
end

#TODO: cache vap_map ahead of time.  This may be bottlenecking.
"""
    _calculate_primal_feasibility(optimizer::Optimizer)

    Evaluate the primal feasibility of the linking constraints defined over the 
    optimizer's graph.
"""
function _calculate_primal_feasibility(optimizer::Optimizer)
    link_constraint_refs = local_link_constraints(optimizer.graph)
    n_link_constraints = length(link_constraint_refs)
    primal_residual = zeros(n_link_constraints)
    for i = 1:n_link_constraints
        # get link constraint function and set
        link_ref = link_constraint_refs[i]
        link_constraint = constraint_object(link_ref)
        func = Plasmo.jump_function(link_constraint)
        set = Plasmo.moi_set(link_constraint)

        # grab variable values from corresponding subproblems
        vars = Plasmo._extract_variables(func)
        var_vals = Dict{NodeVariableRef,Float64}()
        for var in vars
            node = get_node(var)
            subgraph = optimizer.node_subgraph_map[node]
            subproblem_graph = optimizer.expanded_subgraph_map[subgraph]
            var_vals[var] = Plasmo.value(subproblem_graph, var)
        end

        # evaluate linking constraint using subproblem variable values
        constraint_value = Plasmo.value(i -> get(var_vals, i, 0.0), func) 
        primal_residual[i] = constraint_value - set.value
    end
    return primal_residual
end

#NOTE: Need at least an overlap of one for this to work
function _calculate_dual_feasibility(optimizer::Optimizer)
    graph = optimizer.graph
    link_constraint_refs = local_link_constraints(optimizer.graph)
    n_link_constraints = length(link_constraint_refs)
    dual_residual = zeros(n_link_constraints)
    for i = 1:n_link_constraints
        link_ref = link_constraint_refs[i]
        edge = Plasmo.owner_model(link_ref)

        graphs = Set{typeof(optimizer.graph)}()
        for node in all_nodes(edge)
            graph = optimizer.node_subgraph_map[node]
            subproblem_graph = optimizer.expanded_subgraph_map[graph]
            push!(graphs, subproblem_graph)
        end

        # check each subproblem's dual value for this linkconstraint
        duals = zeros(length(graphs))
        for i = 1:length(graphs)
            subproblem_graph = graphs[i]
            duals[i] = dual(subproblem_graph, link_ref)
        end

        # dual residual between subproblems
        dual_residual[i] = abs(maximum(duals) - minimum(duals))
    end
    return dual_residual
end

function _eval_and_save_iteration(optimizer::Optimizer)
    # evaluate residuals
    optimizer.timers.eval_primal_feasibility_time += @elapsed prf = _calculate_primal_feasibility(
        optimizer
    )
    optimizer.timers.eval_dual_feasibility_time += @elapsed duf = _calculate_dual_feasibility(
        optimizer
    )
    optimizer.err_pr = norm(prf[:], Inf)
    optimizer.err_du = norm(duf[:], Inf)

    # eval objective
    optimizer.timers.eval_objective_time += @elapsed optimizer.objective_value = _calculate_objective_value(
        optimizer
    )

    # save iteration data
    push!(optimizer.primal_error_iters, optimizer.err_pr)
    push!(optimizer.dual_error_iters, optimizer.err_du)
    push!(optimizer.objective_iters, optimizer.objective_value)
    return nothing
end

function run_algorithm!(optimizer::Optimizer)
    if !optimizer.initialized
        println("Initializing Optimizer...")
        initialize!(optimizer)
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

    _begin_iterations(optimizer)

    while optimizer.err_pr > optimizer.tolerance || optimizer.err_du > optimizer.tolerance
        optimizer.iteration += 1
        if optimizer.iteration > optimizer.max_iterations
            optimizer.status = MOI.ITERATION_LIMIT
            break
        end

        do_iteration(optimizer)

        _eval_and_save_iteration(optimizer)

        # print iteration
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

        # update start values
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

    # TODO: expose algorithm solution with value and dual.

    optimizer.timers.total_time = time() - optimizer.timers.start_time
    if optimizer.status != MOI.ITERATION_LIMIT
        optimizer.status = Plasmo.termination_status(optimizer.subproblem_graphs[1])
    end

    println()
    println("Number of Iterations: ", optimizer.iteration)
    @printf "%8s | %8s | %8s" "Obj" "Prf" "Duf\n"
    @printf(
        "%7.2e | %7.2e | %7.2e\n",
        optimizer.objective_value,
        optimizer.err_pr,
        optimizer.err_du
    )
    println()
    println("Time spent in subproblems: ", optimizer.timers.solve_subproblem_time)
    println("Solution Time: ", optimizer.timers.total_time)
    return println("EXIT: SchwarzOpt Finished with status: ", optimizer.status)
end