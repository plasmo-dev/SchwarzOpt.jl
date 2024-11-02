[![CI](https://github.com/zavalab/SchwarzOpt.jl/workflows/CI/badge.svg)](https://github.com/plasmo-dev/SchwarzOpt.jl/actions)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

# SchwarzOpt.jl

## Overview
SchwarzOpt.jl implements overlapping Schwarz decomposition for graph-structured optimization problems using the algorithm outlined in this [paper](https://arxiv.org/abs/1810.00491).  
The package works with the graph-based algebraic modeling package [Plasmo.jl](https://github.com/plasmo-dev/Plasmo.jl) to formulate and solve problems.

## Installation
SchwarzOpt.jl can be installed using the following Julia Pkg command:

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/zavalab/SchwarzOpt.jl.git"))
```

## Simple Example
The following example solves a long-horizon optimal control problem where `x` are the states and `u` are the controls.

```julia
using Plasmo, Ipopt
using SchwarzOpt

T = 200             # number of time points
d = sin.(1:T)       # a disturbance vector
n_parts = 10        # number of partitions

# create the optigraph
graph = Plasmo.OptiGraph()
@optinode(graph, state[1:T])
@optinode(graph, control[1:(T - 1)])
for (i, node) in enumerate(state)
    @variable(node, x)
    @constraint(node, x >= 0)
    @objective(node, Min, x^2)
end
for node in control
    @variable(node, u)
    @constraint(node, u >= -1000)
    @objective(node, Min, u^2)
end

# initial condition
n1 = state[1]
@constraint(n1, n1[:x] == 0)

# dynamics
for i in 1:(T - 1)
    @linkconstraint(graph, state[i + 1][:x] == state[i][:x] + control[i][:u] + d[i])
end

# subproblem optimizer
sub_optimizer = Plasmo.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

# generate algorithm instance. the algorithm will use Metis to partition internally.
optimizer = SchwarzOpt.Algorithm(
    graph;
    n_partitions=n_parts,
    subproblem_optimizer=sub_optimizer,
    max_iterations=100,
    mu=10.0,            # augmented lagrangian penalty
)

# run the optimizer
SchwarzOpt.run_algorithm!(optimizer)

# check termination status
@show Plasmo.termination_status(optimizer)

# check objective value
@show Plasmo.objective_value(optimizer)

# check first state and control values
@show Plasmo.value.(optimizer, state[1][:x])
@show Plasmo.value.(optimizer, control[1][:u])
```

### Providing Custom Partitions
It is also possible for users to provide custom partitions (in the form of a `Plasmo.Partition`). 

```julia
using KaHyPar
imbalance = 0.1             # partition imbalance
overlap_distance = 2        # expansion distance

# create a hypergraph projection and partition with KaHyPar
projection = Plasmo.hyper_projection(graph)
partition_vector = KaHyPar.partition(projection, n_parts; imbalance=imbalance, configuration=:edge_cut)

# create a `Plasmo.Partition` object using produced vector
partition = Plasmo.Partition(projection, partition_vector)

# run optimizer using provided partition (overlap will run internally)
optimizer = SchwarzOpt.Algorithm(
    graph,
    partition;
    overlap_distance=overlap_distance,
    subproblem_optimizer=sub_optimizer,
    max_iterations=100,
    mu=10.0,            # augmented lagrangian penalty
)

SchwarzOpt.run_algorithm!(optimizer)

# check termination status
@show Plasmo.termination_status(optimizer)
```

### Providing Custom Subproblems
Users may further provide their own custom overlap and provide  subproblems directly to the algorithm.

```julia
# create custom partitioned optigraph
partitioned_graph = Plasmo.assemble_optigraph(partition)

# generate custom subproblems using overlap distance
subgraphs = Plasmo.local_subgraphs(partitioned_graph)
expanded_subgraphs = Plasmo.expand.(projection, subgraphs, overlap_distance)

# run optimizer using provided subproblem
optimizer = SchwarzOpt.Algorithm(
    partitioned_graph,
    expanded_subgraphs;
    subproblem_optimizer=sub_optimizer,
    max_iterations=100,
    mu=10.0,
)

SchwarzOpt.run_algorithm!(optimizer)

# check termination status
@show Plasmo.termination_status(optimizer)
```

## Important Notes
- SchwarzOpt.jl does not yet perform automatic overlap improvement. This means the user needs to provide sufficient overlap to obtain convergence.
- SchwarzOpt.jl is not meant for problems with integer decision variables.
- Convergence may fail if the user provides non-contiguous subproblems (partitions), which means a subproblem contains distinct sets of unconnected nodes.
