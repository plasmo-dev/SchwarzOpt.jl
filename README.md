# SchwarzSolver

## Overview
SchwarzSolver.jl implements overlapping Schwarz decomposition to graph-structured optimization problems according to this [paper](https://arxiv.org/abs/1810.00491).  
The package works with the graph-based algebraic modeling package [Plasmo.jl](https://github.com/zavalab/Plasmo.jl).

## Installation
SchwarzSolver.jl can be installed using the following Julia Pkg command:

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/zavalab/SchwarzSolver.jl.git"))
```

## Simple Example
```julia
using Plasmo, KaHyPar, Ipopt
using SchwarzSolver

T = 3000          #number of time points
d = sin.(1:T)     #disturbance vector
overlap = 5       #overlap distance
imbalance = 0.1   #partition imbalance

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

@linkconstraint(graph,links[i = 1:T-1], state[i][:x] + control[i][:u] + d[i] == state[i+1][:x])

#Partition the problem
hypergraph,hyper_map = gethypergraph(graph) #create hypergraph object based on graph
partition_vector = KaHyPar.partition(hypergraph,8,configuration = :connectivity,imbalance = imbalance)
partition = Partition(hypergraph,partition_vector,hyper_map)
make_subgraphs!(graph,partition)

#Solve directly with expanded subgraphs
subgraphs = getsubgraphs(graph)
expanded_subs = expand.(Ref(graph),subgraphs,Ref(5))
schwarz_solve(graph,expanded_subs;sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"tol" => 1e-12,"print_level" => 0),max_iterations = 100,tolerance = 1e-10)
```
