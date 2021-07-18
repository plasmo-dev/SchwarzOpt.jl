[![CI](https://github.com/zavalab/SchwarzOpt.jl/workflows/CI/badge.svg)](https://github.com/zavalab/SchwarzOpt.jl/actions)

# SchwarzOpt.jl

## Overview
SchwarzOpt.jl implements overlapping Schwarz decomposition for graph-structured optimization problems using the algorithm outlined in this [paper](https://arxiv.org/abs/1810.00491).  
The package works with the graph-based algebraic modeling package [Plasmo.jl](https://github.com/zavalab/Plasmo.jl) to formulate and solve problems.

## Installation
SchwarzOpt.jl can be installed using the following Julia Pkg command:

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/zavalab/SchwarzOpt.jl.git"))
```

## Simple Example
```julia
#Example demonstrating the use of overlap to solve a long horizon control problem
using Plasmo, Ipopt
using KaHyPar
using SchwarzOpt

T = 100              #number of time points
d = sin.(1:T)        #a disturbance vector
imbalance = 0.1      #partition imbalance
distance = 5         #expand distance
n_parts = 6          #number of partitions

#Create the model (an optigraph)
graph = OptiGraph()
@optinode(graph,state[1:T])
@optinode(graph,control[1:T-1])

for (i,node) in enumerate(state)
    @variable(node,x)
    @constraint(node, x >= 0)
    @objective(node,Min,0.001*x^2) #- 2*x*d[i])
end
for node in control
    @variable(node,u)
    @constraint(node, u >= -1000)
    @objective(node,Min,u^2)
end
n1 = state[1]
@constraint(n1,n1[:x] == 0)

for i = 1:T-1
    @linkconstraint(graph, state[i][:x] + control[i][:u] + d[i] == state[i+1][:x],attach = state[i+1])
end

#Partition the optigraph using recrusive bisection over a hypergraph
hypergraph,hyper_map = hyper_graph(graph) #create hypergraph object based on graph
partition_vector = KaHyPar.partition(hypergraph,n_parts,configuration = "cut_rKaHyPar_sea20.ini",imbalance = imbalance)
partition = Partition(hypergraph,partition_vector,hyper_map)
apply_partition!(graph,partition)

#Inspect the graph structure. It should be a RECURSIVE_GRAPH, which SchwarzOpt.jl supports.
println(Plasmo.graph_structure(graph))

#calculate subproblems using expansion distance
subgraphs = getsubgraphs(graph)
expanded_subgraphs = Plasmo.expand.(graph,subgraphs,distance)
sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"print_level" => 0)

#optimize using schwarz overlapping decomposition
SchwarzOpt.optimize!(graph;
subgraphs = expanded_subgraphs,
sub_optimizer = sub_optimizer,
max_iterations = 50)
```

## Important Notes
- SchwarzOpt.jl does not yet perform automatic overlap improvement.  This means the user needs to provide sufficient overlap such that the optimizer converges.
- Convergence may fail if the user provides non-contiguous subproblems (partitions), which means a subproblem contains distinct sets of unconnected nodes.
