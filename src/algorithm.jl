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
    tolerance::Float64 = 1e-4         # primal and dual tolerance measure
    max_iterations::Int64 = 1000      # maximum number of iterations
    mu::Float64 = 1.0                 # augmented lagrangian penalty
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

    # the dual variables parameters (used for updating the objective)
    dual_variables::OrderedDict{EdgeConstraintRef,NodeVariableRef}

    # the current objective function (with penalties)
    objective_function::Union{Nothing,Plasmo.AbstractJuMPScalar}
end

function SubProblemData(restricted_subgraph::GT) where {GT<:Plasmo.AbstractOptiGraph}
    incident_variable_map = OrderedDict{NodeVariableRef,GT}()
    incident_constraint_map = OrderedDict{EdgeConstraintRef,GT}()
    primal_values = OrderedDict{NodeVariableRef,Float64}()
    dual_values = OrderedDict{EdgeConstraintRef,Float64}()
    dual_variables = OrderedDict{EdgeConstraintRef,NodeVariableRef}()
    return SubProblemData(
        restricted_subgraph,
        incident_variable_map,
        incident_constraint_map,
        primal_values,
        dual_values,
        dual_variables,
        nothing,
    )
end

mutable struct Algorithm{GT<:Plasmo.AbstractOptiGraph}
    graph::GT
    expanded_subgraphs::Vector{GT}  # subproblem graphs

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

    # Inner constructor
    function Algorithm(graph::GT, options::Options) where {GT<:Plasmo.AbstractOptiGraph}
        overlapping_graphs = Vector{GT}()
        return new{GT}(
            graph,
            overlapping_graphs,
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

# Constructors

## provide subproblem graphs directly
function Algorithm(graph::OptiGraph, expanded_subgraphs::Vector{OptiGraph}; kwargs...)
    options = Options(; kwargs...)
    algorithm = Algorithm(graph, options)
    algorithm.expanded_subgraphs = expanded_subgraphs
    return algorithm
end

## TODO partition using default Metis implementation
function Algorithm(
    graph::OptiGraph; n_partitions=2, subproblem_algorithm=nothing, kwargs...
)
    options = Options(kwargs...)
    algorithm = Algorithm(graph, options)
    return algorithm
end

## TODO provide a Plasmo.Partition
function Algorithm(graph::OptiGraph, partition::Plasmo.Partition; kwargs...)
    options = Options(kwargs...)
    algorithm = Algorithm(graph, options)
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
    if !(is_objective_separable(algorithm))
        algorithm.status = MOI.INVALID_MODEL
        error("Algorithm does not yet support non-separable objective functions.")
    end

    # TODO: check if graph is hierarchical. come up with way to handle 'parent' nodes.

    # TODO: raise warning for non-contiguous partitions
    # _check_partitions

    # TODO: check for an overlap of at least 1.
    # _check_overlap

    return true
end

function is_objective_separable(algorithm::Algorithm)
    return _is_objective_separable(objective_function(algorithm.graph))
end

# """
#     Parse what the node objectives would be from the graph objective if it is seperable
# """
# function _parse_node_objectives(algorithm::Algorithm)
# end

function initialize!(algorithm::Algorithm)
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
                    subproblem_data.dual_values[link_constraint] = dual_start
                end
            end
        end

        _initialize_subproblem_objectives(algorithm)

        reset_iterations(algorithm)

        algorithm.initialized = true
    end
end

function _extract_node_objectives(algorithm::Algorithm)
    sense = Plasmo.objective_sense(algorithm.graph)
    for expanded_subgraph in algorithm.expanded_subgraphs
        node_objectives = _extract_node_objectives(objective_func, expanded_subgraph)
        for node in all_nodes(expanded_subgraph)
            set_objective_function(node, node_objectives[node])
            set_objective_sense(node, sense)
        end
    end
end

function _initialize_subproblem_objectives(algorithm::Algorithm)
    # need to extract objective terms from graph if we are not using the node objectives
    if !(algorithm.options.use_node_objectives)
        _extract_node_objectives(algorithm)
    end

    # set to the node objectives and add penalties
    for expanded_subgraph in algorithm.expanded_subgraphs
        set_to_node_objectives(expanded_subgraph)
        _formulate_objective_penalty(expanded_subgraph, algorithm.options.mu)
    end
    return nothing
end

"""
    _formulate_objective_penalty(
        expanded_subgraph::OptiGraph,
        mu::Float64
    )

    Formulate penalty term for each subproblem
"""
function _formulate_objective_penalty(expanded_subgraph::OptiGraph, mu::Float64)
    subproblem_data = expanded_subgraph.ext[:subproblem_data]

    # variable swap function for generating penalty terms
    incident_variables = keys(subproblem_data.incident_variable_map)
    function swap_variable_func(nvref::Plasmo.NodeVariableRef)
        if nvref in incident_variables
            return subproblem_data.primal_values[nvref]
        else
            return nvref
        end
    end

    # start with subproblem objective
    obj = Plasmo.objective_function(expanded_subgraph)
    # add penalty for each incident linking constraint
    for link_reference in keys(subproblem_data.incident_constraint_map)
        link_constraint = Plasmo.constraint_object(link_reference)
        link_func = Plasmo.jump_function(link_constraint)
        link_rhs = Plasmo.moi_set(link_constraint).value

        # generate penalty term by swapping current incident values
        penalty_term = Plasmo.value(swap_variable_func, link_func) - link_rhs

        # augmented penalty term
        augmented_penalty_term = penalty_term^2

        # the current dual multiplier for this constraint
        link_dual_value = subproblem_data.dual_values[link_reference]

        # create a parameter for the dual value that we can update
        dual_node = add_node(expanded_subgraph)
        @variable(dual_node, link_dual in MOI.Parameter(link_dual_value))
        subproblem_data.dual_variables[link_reference] = link_dual

        obj += *(-1, link_dual * penalty_term)
        obj += *(0.5 * mu, augmented_penalty_term)

        # NOTE: add_to_expression does not work on the nonlinear terms
        #Plasmo.add_to_expression!(obj, -1, link_dual*penalty_term)
        #Plasmo.add_to_expression!(obj, 0.5*mu, augmented_penalty_term)
    end
    set_objective_function(expanded_subgraph, obj)
    return nothing
end

function _update_objective_penalty(expanded_subgraph::OptiGraph)
    subproblem_data = expanded_subgraph.ext[:subproblem_data]
    for link_reference in keys(subproblem_data.incident_constraint_map)
        link_dual_value = subproblem_data.dual_values[link_reference]
        link_dual_parameter = subproblem_data.dual_variables[link_reference]
    end
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
        Threads.@threads for subproblem_graph in algorithm.subproblem_graphs
            _solve_subproblem(subproblem_graph)
        end
    end

    # communicate primal and dual values
    algorithm.timers.communicate_time += @elapsed begin
        for subproblem_graph in algorithm.subproblem_graphs
            _communicate_values(subproblem_graph)
        end
    end

    algorithm.timers.update_subproblem_time += @elapsed begin
        for subproblem_graph in algorithm.subproblem_graphs
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
    return nothing
end

function _communicate_values(subproblem_graph::OptiGraph) end

function _update_suproblem(subproblem_graph::OptiGraph)
    _update_objective_penalty(subproblem_graph)
    return nothing
end

# check iteration values

function calculate_objective_value(algorithm::Algorithm)
    func = Plasmo.objective_function(algorithm.graph)

    # grab variable values from corresponding subproblems
    vars = Plasmo._extract_variables(func)
    var_vals = Dict{NodeVariableRef,Float64}()
    for var in vars
        node = get_node(var)
        # subproblem_graph = _get_subproblem(node)
        subgraph = algorithm.node_subgraph_map[node]
        subproblem_graph = algorithm.expanded_subgraph_map[subgraph]
        var_vals[var] = Plasmo.value(subproblem_graph, var)
    end
    objective_value = Plasmo.value(i -> get(var_vals, i, 0.0), func)
    algorithm.objective_value = objective_value

    return nothing
end

#TODO: cache var_map ahead of time.  This may be bottlenecking.
"""
    _calculate_primal_feasibility(algorithm::Algorithm)

    Evaluate the primal feasibility of the linking constraints defined over the 
    algorithm's graph.
"""
function calculate_primal_feasibility(algorithm::Algorithm)
    link_constraint_refs = local_link_constraints(algorithm.graph)
    n_link_constraints = length(link_constraint_refs)
    primal_residual = zeros(n_link_constraints)
    for i in 1:n_link_constraints
        # get link constraint function and set
        link_ref = link_constraint_refs[i]
        link_constraint = Plasmo.constraint_object(link_ref)
        func = Plasmo.jump_function(link_constraint)
        set = Plasmo.moi_set(link_constraint)

        # grab variable values from corresponding subproblems
        vars = Plasmo._extract_variables(func)
        var_vals = Dict{NodeVariableRef,Float64}()
        for var in vars
            node = get_node(var)
            subproblem_graph = node[:expanded_subgraph]
            var_vals[var] = Plasmo.value(subproblem_graph, var)
        end

        # evaluate linking constraint using subproblem variable values
        constraint_value = Plasmo.value(i -> get(var_vals, i, 0.0), func)
        primal_residual[i] = constraint_value - set.value
    end
    return primal_residual
end

# NOTE: Need at least an overlap of one to calculate the dual.
## the pure gauss-seidel approach would require fixing the neighbor variables.
function calculate_dual_feasibility(algorithm::Algorithm)
    graph = algorithm.graph
    link_constraint_refs = local_link_constraints(algorithm.graph)
    n_link_constraints = length(link_constraint_refs)
    dual_residual = zeros(n_link_constraints)
    for i in 1:n_link_constraints
        link_ref = link_constraint_refs[i]
        edge = Plasmo.owner_model(link_ref)

        graphs = Vector{typeof(algorithm.graph)}()
        for node in all_nodes(edge)
            graph = algorithm.node_subgraph_map[node]
            subproblem_graph = algorithm.expanded_subgraph_map[graph]
            push!(graphs, subproblem_graph)
        end
        graphs = unique(graphs)

        # check each subproblem's dual value for this linkconstraint
        duals = zeros(length(graphs))
        for i in 1:length(graphs)
            subproblem_graph = graphs[i]
            duals[i] = dual(subproblem_graph, link_ref)
        end

        # dual residual between subproblems
        dual_residual[i] = abs(maximum(duals) - minimum(duals))
    end
    return dual_residual
end

function eval_iteration(algorithm::Algorithm; save_iteration=true)
    # evaluate residuals
    algorithm.timers.eval_primal_feasibility_time += @elapsed prf = _calculate_primal_feasibility(
        algorithm
    )
    algorithm.timers.eval_dual_feasibility_time += @elapsed duf = _calculate_dual_feasibility(
        algorithm
    )
    algorithm.err_pr = norm(prf[:], Inf)
    algorithm.err_du = norm(duf[:], Inf)

    # eval objective
    algorithm.timers.eval_objective_time += @elapsed algorithm.objective_value = _calculate_objective_value(
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
    println("Number of subproblems: $(length(algorithm.subproblem_graphs))")
    println("Subproblem variables:   $(num_variables.(algorithm.subproblem_graphs))")
    println("Subproblem constraints: $(num_constraints.(algorithm.subproblem_graphs))")
    println()

    reset_iterations(algorithm)

    while algorithm.err_pr > algorithm.tolerance || algorithm.err_du > algorithm.tolerance
        algorithm.iteration += 1
        if algorithm.iteration > algorithm.max_iterations
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
            for subproblem in algorithm.subproblem_graphs
                JuMP.set_start_value.(
                    Ref(subproblem),
                    all_variables(subproblem),
                    value.(Ref(subproblem), all_variables(subproblem)),
                )
            end
        end
    end

    # TODO: expose algorithm solution using value and dual.

    algorithm.timers.total_time = time() - algorithm.timers.start_time
    if algorithm.status != MOI.ITERATION_LIMIT
        algorithm.status = Plasmo.termination_status(algorithm.subproblem_graphs[1])
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
