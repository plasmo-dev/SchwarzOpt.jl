@kwdef mutable struct Timers
    start_time::Float64 = 0.0
    initialize_time::Float64 = 0.0
    eval_objective_time::Float64 = 0.0
    eval_primal_feasibility_time::Float64 = 0.0
    eval_dual_feasibility_time::Float64 = 0.0
    communicate_time::Float64 = 0.0
    update_subproblem_time::Float64 = 0.0
    solve_subproblem_time::Float64 = 0.0
    total_time::Float64 = 0.0
end

@kwdef mutable struct Options
    tolerance::Float64 = 1e-4         # primal and dual tolerance measure
    max_iterations::Int64 = 1000      # maximum number of iterations
    mu::Float64 = 1.0                 # augmented lagrangian penalty
    overlap_distance::Int64 = 1
    use_node_objectives::Bool = true  # whether to ignore graph objective and use nodes
    subproblem_optimizer = nothing
end

mutable struct SubProblemData{GT<:Plasmo.AbstractOptiGraph}
    restricted_subgraph::GT

    # where a subproblem should look for primal and dual values
    incident_variable_map::OrderedDict{NodeVariableRef,GT}
    incident_constraint_map::OrderedDict{EdgeConstraintRef,GT}

    # the current primal and dual values
    primal_values::OrderedDict{NodeVariableRef,Float64}
    dual_values::OrderedDict{EdgeConstraintRef,Float64}

    # primal and dual parameters (used for updating the objective)
    primal_parameters::OrderedDict{NodeVariableRef,NodeVariableRef}
    dual_parameters::OrderedDict{EdgeConstraintRef,NodeVariableRef}

    # node objectives on subproblem
    node_objectives::OrderedDict{OptiNode,Plasmo.AbstractJuMPScalar}

    # the current total objective function (with penalties)
    objective_function::Union{Nothing,Plasmo.AbstractJuMPScalar}

    last_termination_status::MOI.TerminationStatusCode
end

function SubProblemData(restricted_subgraph::GT) where {GT<:Plasmo.AbstractOptiGraph}
    incident_variable_map = OrderedDict{NodeVariableRef,GT}()
    incident_constraint_map = OrderedDict{EdgeConstraintRef,GT}()
    primal_values = OrderedDict{NodeVariableRef,Float64}()
    dual_values = OrderedDict{EdgeConstraintRef,Float64}()
    primal_parameters = OrderedDict{NodeVariableRef,NodeVariableRef}()
    dual_parameters = OrderedDict{EdgeConstraintRef,NodeVariableRef}()
    node_objectives = OrderedDict{OptiNode,Plasmo.AbstractJuMPScalar}()
    return SubProblemData(
        restricted_subgraph,
        incident_variable_map,
        incident_constraint_map,
        primal_values,
        dual_values,
        primal_parameters,
        dual_parameters,
        node_objectives,
        nothing,
        MOI.OPTIMIZE_NOT_CALLED,
    )
end

mutable struct Algorithm{GT<:Plasmo.AbstractOptiGraph}
    graph::GT
    expanded_subgraphs::Vector{GT}              # i.e. subproblem graphs
    objective_func::Plasmo.AbstractJuMPScalar   # for evaluating total objective

    # algorithm options
    options::Options

    # current status
    initialized::Bool
    status::MOI.TerminationStatusCode
    err_pr::Union{Nothing,Float64}
    err_du::Union{Nothing,Float64}
    objective_value::Union{Nothing,Float64}

    # algorithm iteration outputs
    iteration::Int64
    primal_error_iters::Vector{Float64}
    dual_error_iters::Vector{Float64}
    objective_iters::Vector{Float64}
    solve_time::Float64

    # timers
    timers::Timers

    # inner constructor
    function Algorithm(graph::GT, options::Options) where {GT<:Plasmo.AbstractOptiGraph}
        overlapping_graphs = Vector{GT}()
        if options.use_node_objectives
            objective_func = sum(objective_function(node) for node in all_nodes(graph))
        else
            objective_func = Plasmo.objective_function(graph)
        end
        return new{GT}(
            graph,
            overlapping_graphs,
            objective_func,
            options,
            false,
            MOI.OPTIMIZE_NOT_CALLED,
            nothing,
            nothing,
            nothing,
            0,
            Float64[],
            Float64[],
            Float64[],
            0.0,
            Timers(),
        )
    end
end

Base.broadcastable(algorithm::Algorithm) = Ref(algorithm)

# constructors

"""
    Algorithm(graph::OptiGraph, expanded_subgraphs::Vector{OptiGraph}; kwargs...)

Create an algorithm instance by providing the subproblems directly.
"""
function Algorithm(graph::OptiGraph, expanded_subgraphs::Vector{OptiGraph}; kwargs...)
    options = Options(; kwargs...)
    algorithm = Algorithm(graph, options)
    algorithm.expanded_subgraphs = expanded_subgraphs
    return algorithm
end

"""
    Algorithm(graph::OptiGraph, partition::Plasmo.Partition; kwargs...)

Create an algorithm instance by providing a valid `Plasmo.Partition`.
"""
function Algorithm(graph::OptiGraph, partition::Plasmo.Partition; kwargs...)
    options = Options(; kwargs...)
    partitioned_graph = assemble_optigraph(partition)
    subgraphs = local_subgraphs(partitioned_graph)
    projection = hyper_projection(graph)
    algorithm = Algorithm(partitioned_graph, options)
    algorithm.expanded_subgraphs = expand.(projection, subgraphs, options.overlap_distance)
    return algorithm
end

function Algorithm(graph::OptiGraph; n_partitions=4, kwargs...)
    options = Options(; kwargs...)
    clique_proj = Plasmo.clique_projection(graph)
    metis_vector = Int64.(Metis.partition(clique_proj.projected_graph, n_partitions))
    metis_partition = Partition(clique_proj, metis_vector)
    partitioned_graph = assemble_optigraph(metis_partition)
    subgraphs = local_subgraphs(partitioned_graph)
    projection = hyper_projection(graph)
    algorithm = Algorithm(partitioned_graph, options)
    algorithm.expanded_subgraphs = expand.(projection, subgraphs, options.overlap_distance)
    return algorithm
end

function check_valid_problem(algorithm::Algorithm)
    graph = algorithm.graph
    restricted_subgraphs = local_subgraphs(graph)
    n_subproblems = length(algorithm.expanded_subgraphs)

    if algorithm.options.subproblem_optimizer == nothing
        error(
            "No algorithm set for the subproblems.  Please provide an algorithm constructor to use to solve subproblem optigraphs",
        )
    end

    # check subgraphs are valid
    for i in 1:n_subproblems
        restricted_subgraph = restricted_subgraphs[i]
        subproblem_graph = algorithm.expanded_subgraphs[i]
        if !(
            intersect(all_nodes(restricted_subgraph), all_nodes(subproblem_graph)) ==
            all_nodes(restricted_subgraph)
        )
            algorithm.status = MOI.INVALID_MODEL
            error("Invalid subproblems given to algorithm.")
        end
    end

    # TODO: check if objective is separable. need custom way to handle non-separable.
    if !(Plasmo.is_separable(objective_function(algorithm.graph)))
        algorithm.status = MOI.INVALID_MODEL
        error("Algorithm does not yet support non-separable objective functions.")
    end

    # TODO: check if graph is hierarchical. come up with way to handle 'parent' nodes.
    # _check_hierarchical

    # TODO: raise warning for non-contiguous partitions
    # _check_partitions

    # TODO: check for an overlap of at least 1.
    # _check_overlap

    return true
end

function initialize!(algorithm::Algorithm)
    if algorithm.initialized
        error("Algorithm already initialized. Create a new instance of 
        `Algorithm` if you want to restart the algorithm completely.")
    end

    return algorithm.timers.initialize_time = @elapsed begin
        check_valid_problem(algorithm)

        graph = algorithm.graph
        n_subproblems = length(algorithm.expanded_subgraphs)
        graph_type = typeof(graph)
        obj_type = objective_function_type(graph)

        ########################################################
        # SETUP NODE TO SUBPROBLEM-GRAPH MAPPINGS
        ########################################################
        original_subgraphs = local_subgraphs(algorithm.graph)
        for i in 1:n_subproblems
            restricted_subgraph = original_subgraphs[i]
            expanded_subgraph = algorithm.expanded_subgraphs[i]
            for node in all_nodes(restricted_subgraph)
                # algorithm.node_subgraph_map[node] = restricted_subgraph
                node[:subproblem_graph] = expanded_subgraph
            end
            for edge in all_edges(restricted_subgraph)
                # algorithm.edge_subgraph_map[edge] = restricted_subgraph
                edge[:subproblem_graph] = expanded_subgraph
            end
        end

        # assign cross edges that couple subgraphs
        for link_edge in local_edges(algorithm.graph)
            assigned_node = link_edge.nodes[1]
            link_edge[:subproblem_graph] = assigned_node[:subproblem_graph]
        end

        ########################################################
        # INITIALIZE SUBPROBLEM DATA
        ########################################################
        all_incident_edges = _find_boundary_edges(
            algorithm.graph, algorithm.expanded_subgraphs
        )
        all_incident_constraints = _extract_constraints(all_incident_edges)
        @assert length(all_incident_constraints) == n_subproblems

        # setup data structure for each subproblem
        for i in 1:n_subproblems
            restricted_subgraph = original_subgraphs[i]
            expanded_subgraph = algorithm.expanded_subgraphs[i]
            subproblem_incident_edges = all_incident_edges[i]
            subproblem_data = SubProblemData(restricted_subgraph)
            expanded_subgraph.ext[:subproblem_data] = subproblem_data

            # create a node to hold parameter variables
            parameter_node = add_node(expanded_subgraph)

            for linking_edge in subproblem_incident_edges
                linked_variables = all_variables(linking_edge)

                # map primal variables
                incident_variables = setdiff(
                    linked_variables, all_variables(expanded_subgraph)
                )
                for incident_variable in incident_variables
                    owning_node = get_node(incident_variable)
                    owning_subproblem = owning_node[:subproblem_graph]
                    subproblem_data.incident_variable_map[incident_variable] =
                        owning_subproblem
                    if Plasmo.start_value(incident_variable) == nothing
                        primal_start = 1.0
                    else
                        primal_start = Plasmo.start_value(incident_variable)
                    end
                    @variable(parameter_node, p in MOI.Parameter(primal_start))
                    subproblem_data.primal_parameters[incident_variable] = p
                    subproblem_data.primal_values[incident_variable] = primal_start
                end

                # build dual information map
                for link_constraint in all_constraints(linking_edge)
                    owning_edge = get_edge(link_constraint)
                    owning_subproblem = owning_edge[:subproblem_graph]
                    subproblem_data.incident_constraint_map[link_constraint] =
                        owning_subproblem
                    if Plasmo.start_value(link_constraint) == nothing
                        dual_start = 1.0
                    else
                        dual_start = Plasmo.start_value(link_constraint)
                    end
                    @variable(parameter_node, d in MOI.Parameter(dual_start))
                    subproblem_data.dual_parameters[link_constraint] = d
                    subproblem_data.dual_values[link_constraint] = dual_start
                end
            end
        end

        _initialize_subproblem_objectives(algorithm)

        reset_iterations(algorithm)

        algorithm.initialized = true
    end
end

function _initialize_subproblem_objectives(algorithm::Algorithm)
    if algorithm.options.use_node_objectives
        for expanded_subgraph in algorithm.expanded_subgraphs
            Plasmo.set_to_node_objectives(expanded_subgraph)
        end
    else
        # extract objective terms from graph if we are not using the node objectives
        # sets [:schwarz_objective] on each optinode
        _extract_node_objectives(algorithm)

        # set the objective on each subproblem
        for expanded_subgraph in algorithm.expanded_subgraphs
            set_objective_function(
                expanded_subgraph,
                sum(node[:schwarz_objective] for node in all_nodes(expanded_subgraph)),
            )
            set_objective_sense(expanded_subgraph, Plasmo.objective_sense(algorithm.graph))
        end
    end

    # add objective penalties
    for expanded_subgraph in algorithm.expanded_subgraphs
        _formulate_objective_penalty(expanded_subgraph, algorithm.options.mu)
    end
    return nothing
end

function _extract_node_objectives(algorithm::Algorithm)
    objective_func = Plasmo.objective_function(algorithm.graph)
    for expanded_subgraph in algorithm.expanded_subgraphs
        restricted_subgraph = expanded_subgraph.ext[:subproblem_data].restricted_subgraph
        node_objectives = Plasmo.extract_separable_terms(
            objective_func, restricted_subgraph
        )
        for node in all_nodes(restricted_subgraph)
            node[:schwarz_objective] = sum(node_objectives[node])
        end
    end
    return nothing
end

"""
    _formulate_objective_penalty(
        expanded_subgraph::OptiGraph,
        mu::Float64
    )

Formulate and add objective penalty term for the given subproblem.

Args:
    `expanded_subgraph`: The optimizer subproblem that contains overlap.
    `mu`: A hyperparameter for the augmented lagrangian penalty.
"""
function _formulate_objective_penalty(expanded_subgraph::OptiGraph, mu::Float64)
    subproblem_data = expanded_subgraph.ext[:subproblem_data]

    # variable swap function for generating penalty terms
    incident_variables = keys(subproblem_data.incident_variable_map)
    function swap_variable_func(nvref::Plasmo.NodeVariableRef)
        if nvref in incident_variables
            return subproblem_data.primal_parameters[nvref]
        else
            return nvref
        end
    end

    # start with subproblem objective
    subproblem_objective = Plasmo.objective_function(expanded_subgraph)

    # add penalty for each incident linking constraint
    for link_reference in keys(subproblem_data.incident_constraint_map)
        link_constraint = Plasmo.constraint_object(link_reference)
        link_func = Plasmo.jump_function(link_constraint)
        link_rhs = Plasmo.moi_set(link_constraint).value

        # generate penalty term by swapping incident terms with their values
        penalty_term = Plasmo.value(swap_variable_func, link_func) - link_rhs
        augmented_penalty_term = penalty_term^2

        # update objective with dual term and augmented penalty
        link_dual_parameter = subproblem_data.dual_parameters[link_reference]
        subproblem_objective += *(-1, link_dual_parameter * penalty_term)
        subproblem_objective += *(0.5 * mu, augmented_penalty_term)

        # NOTE: add_to_expression does not work on the nonlinear terms
        # Plasmo.add_to_expression!(obj, -1, link_dual*penalty_term)
        # Plasmo.add_to_expression!(obj, 0.5*mu, augmented_penalty_term)
    end
    set_objective_function(expanded_subgraph, subproblem_objective)
    return nothing
end

function reset_iterations(algorithm::Algorithm)
    algorithm.timers = Timers()
    algorithm.timers.start_time = time()

    # initialize subproblem algorithms
    for subgraph in algorithm.expanded_subgraphs
        Plasmo.set_optimizer(subgraph, algorithm.options.subproblem_optimizer)
    end

    algorithm.err_pr = Inf
    algorithm.err_du = Inf
    algorithm.iteration = 0
    return nothing
end

"""
    do_iteration(algorithm::Algorithm)

    Perform a single iteration of the algorithm. Solves each subproblem and communicates 
    solution values between them.
"""
function do_iteration(algorithm::Algorithm)
    # update based on current information
    algorithm.timers.solve_subproblem_time += @elapsed begin
        Threads.@threads for subproblem_graph in algorithm.expanded_subgraphs
            _solve_subproblem(subproblem_graph)
        end
    end

    # communicate primal and dual values
    algorithm.timers.communicate_time += @elapsed begin
        Threads.@threads for subproblem_graph in algorithm.expanded_subgraphs
            _retrieve_neighbor_values(subproblem_graph)
        end
    end

    algorithm.timers.update_subproblem_time += @elapsed begin
        Threads.@threads for subproblem_graph in algorithm.expanded_subgraphs
            _update_suproblem(subproblem_graph)
        end
    end

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
    subproblem_graph.ext[:subproblem_data].last_termination_status = term_status
    return nothing
end

function _retrieve_neighbor_values(subproblem_graph::OptiGraph)
    subproblem_data = subproblem_graph.ext[:subproblem_data]
    for (variable, remote_graph) in subproblem_data.incident_variable_map
        subproblem_data.primal_values[variable] = Plasmo.value(remote_graph, variable)
    end
    for (link_reference, remote_graph) in subproblem_data.incident_constraint_map
        subproblem_data.dual_values[link_reference] = Plasmo.dual(
            remote_graph, link_reference
        )
    end
    return nothing
end

function _update_suproblem(subproblem_graph::OptiGraph)
    _update_objective_penalty(subproblem_graph)
    return nothing
end

function _update_objective_penalty(expanded_subgraph::OptiGraph)
    subproblem_data = expanded_subgraph.ext[:subproblem_data]
    for variable in keys(subproblem_data.incident_variable_map)
        variable_primal_value = subproblem_data.primal_values[variable]
        variable_parameter = subproblem_data.primal_parameters[variable]
        Plasmo.set_parameter_value(variable_parameter, variable_primal_value)
    end

    for link_reference in keys(subproblem_data.incident_constraint_map)
        link_dual_value = subproblem_data.dual_values[link_reference]
        link_dual_parameter = subproblem_data.dual_parameters[link_reference]
        Plasmo.set_parameter_value(link_dual_parameter, link_dual_value)
    end
    return nothing
end

"""
    calculate_objective_value(algorithm::Algorithm)

Evaluate the objective value defined over the algorithm's graph.
"""
function calculate_objective_value(algorithm::Algorithm)
    # grab variable values from corresponding subproblems
    vars = Plasmo.extract_variables(algorithm.objective_func)
    var_vals = Dict{NodeVariableRef,Float64}()
    for var in vars
        node = get_node(var)
        subproblem_graph = node[:subproblem_graph]
        var_vals[var] = Plasmo.value(subproblem_graph, var)
    end

    # evaluate using restricted solutions
    objective_value = Plasmo.value(i -> get(var_vals, i, 0.0), algorithm.objective_func)
    algorithm.objective_value = objective_value
    return objective_value
end

"""
    calculate_primal_feasibility(algorithm::Algorithm)

Evaluate the primal feasibility of the linking constraints defined over the 
algorithm's graph.
"""
function calculate_primal_feasibility(algorithm::Algorithm)
    link_constraint_refs = local_link_constraints(algorithm.graph)
    n_link_constraints = length(link_constraint_refs)
    primal_residuals = Vector{Float64}(undef, n_link_constraints)
    for i in 1:n_link_constraints
        link_reference = link_constraint_refs[i]
        link_constraint = Plasmo.constraint_object(link_reference)
        func = Plasmo.jump_function(link_constraint)
        set = Plasmo.moi_set(link_constraint)

        # grab variable values from corresponding subproblems
        vars = Plasmo.extract_variables(func)
        var_vals = Dict{NodeVariableRef,Float64}()
        for var in vars
            node = get_node(var)
            subproblem_graph = node[:subproblem_graph]
            var_vals[var] = Plasmo.value(subproblem_graph, var)
        end

        # evaluate linking constraint using subproblem variable values
        constraint_value = Plasmo.value(i -> get(var_vals, i, 0.0), func)
        primal_residuals[i] = constraint_value - set.value
    end
    return primal_residuals
end

"""
    calculate_dual_feasibility(algorithm::Algorithm)

Evaluate the dual feasibility of the linking constraints defined over the 
algorithm's graph. NOTE: Need at least an overlap of one to calculate the dual.
"""
function calculate_dual_feasibility(algorithm::Algorithm)
    graph = algorithm.graph
    link_constraint_refs = local_link_constraints(algorithm.graph)
    n_link_constraints = length(link_constraint_refs)
    dual_residuals = Vector{Float64}(undef, n_link_constraints)
    for i in 1:n_link_constraints
        link_reference = link_constraint_refs[i]
        edge = Plasmo.get_edge(link_reference)

        # get all subproblems that contain this edge
        graphs = Set{typeof(algorithm.graph)}()
        for node in all_nodes(edge)
            subproblem_graph = node[:subproblem_graph]
            push!(graphs, subproblem_graph)
        end

        # check each subproblem's dual value for this linkconstraint
        subproblem_duals = Vector{Float64}(undef, length(graphs))
        for (i, graph) in enumerate(collect(graphs))
            subproblem_duals[i] = dual(graph, link_reference)
        end
        dual_residuals[i] = abs(maximum(subproblem_duals) - minimum(subproblem_duals))
    end
    return dual_residuals
end

"""
    eval_iteration(algorithm::Algorithm; save_iteration=true)

Evaluate the current iterate and store values. Save the iterate values internally if 
`save_iteration=true`.
"""
function eval_iteration(algorithm::Algorithm; save_iteration=true)
    # evaluate residuals
    algorithm.timers.eval_primal_feasibility_time += @elapsed prf = calculate_primal_feasibility(
        algorithm
    )
    algorithm.timers.eval_dual_feasibility_time += @elapsed duf = calculate_dual_feasibility(
        algorithm
    )
    algorithm.err_pr = norm(prf[:], Inf)
    algorithm.err_du = norm(duf[:], Inf)

    # eval objective
    algorithm.timers.eval_objective_time += @elapsed algorithm.objective_value = calculate_objective_value(
        algorithm
    )

    # save iteration data
    if save_iteration
        push!(algorithm.primal_error_iters, algorithm.err_pr)
        push!(algorithm.dual_error_iters, algorithm.err_du)
        push!(algorithm.objective_iters, algorithm.objective_value)
    end
    return nothing
end

function _check_tolerance(algorithm::Algorithm)
    if algorithm.err_pr > algorithm.options.tolerance
        return true
    elseif algorithm.err_du > algorithm.options.tolerance
        return true
    end
    return false
end

function run_algorithm!(algorithm::Algorithm)
    if !algorithm.initialized
        println("Initializing Algorithm...")
        initialize!(algorithm)
    end

    println("###########################################################")
    println("Running Algorithm: SchwarzOpt v0.3.0 using $(Threads.nthreads()) threads")
    println("###########################################################")
    println()
    println("Number of variables: $(num_variables(algorithm.graph))")
    println("Number of constraints: $(num_constraints(algorithm.graph))")
    println("Number of subproblems: $(length(algorithm.expanded_subgraphs))")
    println("Subproblem variables:   $(num_variables.(algorithm.expanded_subgraphs))")
    println("Subproblem constraints: $(num_constraints.(algorithm.expanded_subgraphs))")
    println()

    reset_iterations(algorithm)

    while _check_tolerance(algorithm)
        algorithm.iteration += 1
        if algorithm.iteration > algorithm.options.max_iterations
            algorithm.status = MOI.ITERATION_LIMIT
            break
        end

        do_iteration(algorithm)

        eval_iteration(algorithm)

        # print iteration
        if algorithm.iteration % 20 == 0 || algorithm.iteration == 1
            @printf "%4s | %8s | %8s | %8s" "Iter" "Obj" "Prf" "Duf\n"
        end
        @printf(
            "%4i | %7.2e | %7.2e | %7.2e\n",
            algorithm.iteration,
            algorithm.objective_value,
            algorithm.err_pr,
            algorithm.err_du
        )

        # update start values
        algorithm.timers.update_subproblem_time += @elapsed begin
            for subproblem in algorithm.expanded_subgraphs
                JuMP.set_start_value.(
                    Ref(subproblem),
                    all_variables(subproblem),
                    value.(Ref(subproblem), all_variables(subproblem)),
                )
            end
        end
    end

    algorithm.timers.total_time = time() - algorithm.timers.start_time
    if algorithm.status != MOI.ITERATION_LIMIT
        algorithm.status =
            algorithm.expanded_subgraphs[1].ext[:subproblem_data].last_termination_status
    end

    println()
    println("Number of Iterations: ", algorithm.iteration)
    @printf "%8s | %8s | %8s" "Obj" "Prf" "Duf\n"
    @printf(
        "%7.2e | %7.2e | %7.2e\n",
        algorithm.objective_value,
        algorithm.err_pr,
        algorithm.err_du
    )
    println()
    println("Time spent in subproblems: ", algorithm.timers.solve_subproblem_time)
    println("Solution Time: ", algorithm.timers.total_time)
    println("EXIT: SchwarzOpt Finished with status: ", algorithm.status)
    return nothing
end