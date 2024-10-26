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
end

mutable struct SubProblemData{GT<:Plasmo.AbstractOptiGraph}
    restricted_subgraph::GT

    # where a subproblem should look for primal and dual values
    incident_variable_map::OrderedDict{NodeVariableRef,GT}
    incident_constraint_map::OrderedDict{EdgeConstraintRef,GT}

    # the current primal and dual values
    primal_values::OrderedDict{NodeVariableRef,Float64}
    dual_values::OrderedDict{EdgeConstraintRef,Float64}

    # the current objective function (with penalties)
    objective_function::Plasmo.AbstractJuMPScalar
end

function SubProblemData(restricted_subgraph::GT, obj_type::Type{OT})
    {where GT <: Plasmo.AbstractOptiGraph, where OT<:Plasmo.AbstractJuMPScalar}
    incident_variable_map = OrderedDict{NodeVariableRef,GT}()
    incident_constraint_map = OrderedDict{EdgeConstraintRef,GT}()
    primal_values = OrderedDict{NodeVariableRef,Float64}()
    dual_values = OrderedDict{EdgeConstraintRef,Float64}()
    return SubproblemData(
        restricted_subgraph,
        incident_variable_map,
        incident_constraint_map,
        primal_values,
        dual_values,
        zero(obj_type),
    )
end

mutable struct Optimizer{GT<:Plasmo.AbstractOptiGraph}
    graph::GT
    expanded_subgraphs::Vector{GT}  # subproblem graphs
    subproblem_optimizer::Any

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
    function Optimizer(graph::GT, options::Options) where {GT<:Plasmo.AbstractOptiGraph}
        overlapping_graphs = Vector{GT}()
        subproblem_optimizer = nothing
        return new{GT}(
            graph,
            overlapping_graphs,
            subproblem_optimizer,
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

### Constructors

# provide subproblem graphs directly
function Optimizer(
    graph::OptiGraph,
    expanded_subgraphs::Vector{OptiGraph};
    subproblem_optimizer=nothing,
    tolerance=1e-4,
    max_iterations=100,
    mu=1.0,
)
    options = Options(tolerance, max_iterations, mu)
    optimizer = Optimizer(graph, options)
    optimizer.expanded_subgraphs = expanded_subgraphs
    optimizer.subproblem_optimizer = subproblem_optimizer
    return optimizer
end

# TODO partition using default Metis implementation
function Optimizer(
    graph::OptiGraph;
    n_partitions=2,
    subproblem_optimizer=nothing,
    tolerance=1e-4,
    max_iterations=100,
    mu=1.0,
)
    options = Options(tolerance, max_iterations, mu)
    optimizer = Optimizer(graph, options)
    return optimizer
end

# TODO provide a Plasmo.Partition
function Optimizer(
    graph::OptiGraph,
    partition::Plasmo.Partition;
    tolerance::Float64=1e-4,
    max_iterations::Int64=100,
    mu::Float64=1.0,
)
    options = Options(tolerance, max_iterations, mu)
    optimizer = Optimizer(graph, options)
    return optimizer
end

function check_valid_problem(optimizer::Optimizer)
    graph = optimizer.graph
    restricted_subgraphs = local_subgraphs(graph)
    n_subproblems = length(optimizer.overlapping_graphs)

    if optimizer.subproblem_optimizer == nothing
        error(
            "No optimizer set for the subproblems.  Please provide an optimizer constructor to use to solve subproblem optigraphs",
        )
    end

    # check subgraphs are valid
    for i in 1:n_subproblems
        restricted_subgraph = restricted_subgraphs[i]
        subproblem_graph = optimizer.overlapping_graphs[i]
        if intersect(all_nodes(restricted_subgraph), all_nodes(expanded_subgraph)) ==
            all_nodes(original_subgraph)
            optimizer.status = MOI.INVALID_MODEL
            error("Invalid subproblems given to optimizer.")
        end
    end

    # TODO: check if objective is separable. need custom way to handle non-separable.
    if !(_is_objective_separable(optimizer))
        optimizer.status = MOI.INVALID_MODEL
        error("Optimizer does not yet support non-separable objective functions.")
    end

    # TODO: check if graph is hierarchical. come up with way to handle 'parent' nodes.

    # TODO: raise warning for non-contiguous partitions
    # _check_partitions

    # TODO: check for an overlap of at least 1.
    # _check_overlap

    return true
end

function is_objective_separable(optimizer::Optimizer)
    return _is_objective_separable(objective_function(optimizer.graph))
end

# """
#     Parse what the node objectives would be from the graph objective if it is seperable
# """
# function _parse_node_objectives(optimizer::Optimizer)
# end

function initialize!(optimizer::Optimizer)
    return optimizer.timers.initialize_time = @elapsed begin
        _check_valid_problem(optimizer)

        graph = optimizer.graph
        n_subproblems = length(optimizer.subproblem_graphs)
        graph_type = typeof(graph)
        obj_type = objective_function_type(graph)

        # TODO: reformulate objective function if necessary onto optinodes.
        # track node objectives on subproblems
        # node_objectives = _parse_graph_objective_to_nodes(graph)

        ########################################################
        # SETUP NODE TO SUBPROBLEM-GRAPH MAPPINGS
        ########################################################
        original_subgraphs = local_subgraphs(optimizer.graph)
        for i in 1:n_subproblems
            expanded_subgraph = optimizer.expanded_subgraphs[i]
            for node in all_nodes(restricted_subgraph)
                # optimizer.node_subgraph_map[node] = restricted_subgraph
                node[:subproblem_graph] = expanded_subgraph
            end
            for edge in all_edges(restricted_subgraph)
                # optimizer.edge_subgraph_map[edge] = restricted_subgraph
                edge[:subproblem_graph] = expanded_subgraph
            end
        end

        # assign cross edges that couple subgraphs
        for link_edge in local_edges(optimizer.graph)
            assigned_node = link_edge.nodes[1]
            link_edge[:subproblem_graph] = assigned_node[:subproblem_graph]
        end

        ########################################################
        # INITIALIZE SUBPROBLEM DATA
        ########################################################
        all_incident_edges = _find_boundary_edges(
            optimizer.graph, optimizer.expanded_subgraphs
        )
        all_incident_constraints = _get_boundary_constraints(subproblem_incident_edges)
        @assert length(all_incident_constraints) == n_subproblems

        # setup data structure for each subproblem
        for i in 1:n_subproblems
            restricted_subgraph = original_subgraphs[i]
            expanded_subgraph = optimizer.expanded_graphs[i]
            subproblem_incident_edges = all_incident_edges[i]
            subproblem_data = SubProblemData(restricted_subgraph, obj_type)

            for linking_edge in subproblem_incident_edges
                linked_variables = all_variables(linking_edge)

                # map primal variables
                incident_variables = setdiff(linked_variables, all_variables(subproblem))
                for incident_variable in incident_variables
                    owning_node = get_node(incident_variable)
                    owning_subproblem = node.ext[:subproblem_graph]
                    subproblem_data.incident_variable_map[incident_variable] =
                        owning_subproblem
                    if Plasmo.start_value(incident_variable) == nothing
                        primal_start = 0.0
                    else
                        primal_start = Plasmo.start_value(incident_variable)
                    end
                    subproblem_data.primal_values[incident_variable] = primal_start
                end

                # build dual information map
                for link_constraint in all_constraints(linking_edge)
                    owning_edge = get_edge(link_constraint)
                    owning_subproblem = edge.ext[:subproblem_graph]
                    subproblem_data.incident_constraint_map[link_constraint] =
                        owning_subproblem
                    if Plasmo.start_value(link_constraint) == nothing
                        dual_start = 0.0
                    else
                        dual_start = Plasmo.start_value(link_constraint)
                    end
                    subproblem_data.dual_values[link_constraint] = dual_start
                end
            end
        end

        _initialize_subproblem_objectives(optimizer)

        optimizer.initialized = true
    end
end

function _extract_node_objectives(optimizer::Optimizer)
    sense = Plasmo.objective_sense(optimizer.graph)
    for expanded_subgraph in optimizer.expanded_subgraphs
        node_objectives = _extract_node_objectives(objective_func, expanded_subgraph)
        for node in all_nodes(expanded_subgraph)
            set_objective_function(node, node_objectives[node])
            set_objective_sense(node, sense)
        end
    end
end

function _initialize_subproblem_objectives(optimizer::Optimizer)
    # need to extract objective terms from graph if we are not using the node objectives
    if !(optimizer.options.use_node_objectives)
        _extract_node_objectives(optimizer)
    end

    # set to the node objectives and add penalties
    for expanded_subgraph in optimizer.expanded_subgraphs
        set_to_node_objectives(expanded_subgraph)
        _formulate_objective_penalty(expanded_subgraph)
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
    subproblem_data = expanded_subgraph.subproblem_data

    # variable swap function for generating penalty terms
    incident_variables = keys(subproblem_data.incident_variable_map)
    function swap_variable(nvref::Plasmo.NodeVariableRef)
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
        link_rhs = Plasmo.moi_set(link).value

        # generate penalty term by swapping current incident values
        penalty_term = Plasmo.value(swap_variable, link_func) - link_rhs

        # augmented penalty term
        augmented_penalty_term = penalty_term^2

        # the current dual multiplier for this constraint
        link_dual = subproblem_data.dual_values[link_reference]
        Plasmo.add_to_expression!(obj, -1, link_dual * penalty_term)
        Plasmo.add_to_expression!(obj, 0.5 * mu, augmented_penalty_term)

        # set_objective_function(
        #     subproblem_graph,
        #     objective_function(subproblem_graph) - l_link_value * penalty +
        #     0.5 * optimizer.mu * augmented_penalty,
        # )

        # TODO: need a way to swap out the multiplier
    end
    return nothing
end

# TODO: need a way to swap out the multiplier
function _update_objective_penalty() end

function reset_iterations(optimizer::Optimizer)
    optimizer.timers = Timers()
    optimizer.timers.start_time = time()

    # initialize subproblem optimizers
    for subgraph in optimizer.expanded_subgraphs
        Plasmo.set_optimizer(subgraph, optimizer.subproblem_optimizer)
    end

    optimizer.err_pr = Inf
    optimizer.err_du = Inf
    optimizer.iteration = 0
    return nothing
end

"""
    do_iteration(optimizer::Optimizer)

    Perform a single iteration of the optimizer. Solves each subproblem and communicates 
    solution values between them.
"""
function do_iteration(optimizer::Optimizer)
    # update based on current information
    optimizer.timers.solve_subproblem_time += @elapsed begin
        Threads.@threads for subproblem_graph in optimizer.subproblem_graphs
            _solve_subproblem(subproblem_graph)
        end
    end

    # communicate primal and dual values
    optimizer.timers.communicate_time += @elapsed begin
        _communicate_values(subproblem_graph)
    end
    return nothing
end

function _communicate_values(subproblem_graph::OptiGraph) end

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

### check iteration values

function calculate_objective_value(optimizer::Optimizer)
    func = Plasmo.objective_function(optimizer.graph)

    # grab variable values from corresponding subproblems
    vars = Plasmo._extract_variables(func)
    var_vals = Dict{NodeVariableRef,Float64}()
    for var in vars
        node = get_node(var)
        # subproblem_graph = _get_subproblem(node)
        subgraph = optimizer.node_subgraph_map[node]
        subproblem_graph = optimizer.expanded_subgraph_map[subgraph]
        var_vals[var] = Plasmo.value(subproblem_graph, var)
    end
    objective_value = Plasmo.value(i -> get(var_vals, i, 0.0), func)
    optimizer.objective_value = objective_value

    return nothing
end

#TODO: cache var_map ahead of time.  This may be bottlenecking.
"""
    _calculate_primal_feasibility(optimizer::Optimizer)

    Evaluate the primal feasibility of the linking constraints defined over the 
    optimizer's graph.
"""
function calculate_primal_feasibility(optimizer::Optimizer)
    link_constraint_refs = local_link_constraints(optimizer.graph)
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
            # subgraph = optimizer.node_subgraph_map[node]
            # subproblem_graph = optimizer.expanded_subgraph_map[subgraph]
            var_vals[var] = Plasmo.value(subproblem_graph, var)
        end

        # evaluate linking constraint using subproblem variable values
        constraint_value = Plasmo.value(i -> get(var_vals, i, 0.0), func)
        primal_residual[i] = constraint_value - set.value
    end
    return primal_residual
end

# NOTE: Need at least an overlap of one to calculate the dual.
# the pure gauss-seidel approach would require fixing the neighbor variables.
function calculate_dual_feasibility(optimizer::Optimizer)
    graph = optimizer.graph
    link_constraint_refs = local_link_constraints(optimizer.graph)
    n_link_constraints = length(link_constraint_refs)
    dual_residual = zeros(n_link_constraints)
    for i in 1:n_link_constraints
        link_ref = link_constraint_refs[i]
        edge = Plasmo.owner_model(link_ref)

        graphs = Vector{typeof(optimizer.graph)}()
        for node in all_nodes(edge)
            graph = optimizer.node_subgraph_map[node]
            subproblem_graph = optimizer.expanded_subgraph_map[graph]
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

function eval_iteration(optimizer::Optimizer; save_iteration=true)
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
    if save_iteration
        push!(optimizer.primal_error_iters, optimizer.err_pr)
        push!(optimizer.dual_error_iters, optimizer.err_du)
        push!(optimizer.objective_iters, optimizer.objective_value)
    end
    return nothing
end

function run_algorithm!(optimizer::Optimizer)
    if !optimizer.initialized
        println("Initializing Optimizer...")
        initialize!(optimizer)
    end

    println("###########################################################")
    println("Optimizing with SchwarzOpt v0.3.0 using $(Threads.nthreads()) threads")
    println("###########################################################")
    println()
    println("Number of variables: $(num_variables(optimizer.graph))")
    println("Number of constraints: $(num_constraints(optimizer.graph))")
    println("Number of subproblems: $(length(optimizer.subproblem_graphs))")
    println("Overlap: ")
    println("Subproblem variables:   $(num_variables.(optimizer.subproblem_graphs))")
    println("Subproblem constraints: $(num_constraints.(optimizer.subproblem_graphs))")
    println()

    reset_iterations(optimizer)

    while optimizer.err_pr > optimizer.tolerance || optimizer.err_du > optimizer.tolerance
        optimizer.iteration += 1
        if optimizer.iteration > optimizer.max_iterations
            optimizer.status = MOI.ITERATION_LIMIT
            break
        end

        do_iteration(optimizer)

        eval_and_save_iteration(optimizer)

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

    # TODO: expose algorithm solution using value and dual.

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
    println("EXIT: SchwarzOpt Finished with status: ", optimizer.status)
    return nothing
end
