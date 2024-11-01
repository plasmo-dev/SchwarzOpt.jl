#Example demonstrating the use of overlap to solve a long horizon control problem
using Plasmo, Ipopt
using KaHyPar
using SchwarzOpt

T = 200             #number of time points
d = sin.(1:T)       #a disturbance vector
imbalance = 0.1    #partition imbalance
distance = 2        #expand distance
n_parts = 10        #number of partitions

#Create the optigraph
graph = OptiGraph()
@optinode(graph, state[1:T])
@optinode(graph, control[1:(T - 1)])
for (i, node) in enumerate(state)
    @variable(node, x)
    @constraint(node, x >= 0)
    @objective(node, Min, x^2) #- 2*x*d[i])
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

projection = hyper_projection(graph)
partition_vector = KaHyPar.partition(projection, n_parts; configuration=:edge_cut)
partition = Partition(projection, partition_vector)
apply_partition!(graph, partition)

#calculate subproblems using expansion distance
subs = local_subgraphs(graph)
expanded_subgraphs = Plasmo.expand.(projection, subs, distance)
sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

#optimize using schwarz overlapping decomposition
optimizer = SchwarzOpt.Algorithm(
    graph,
    expanded_subgraphs;
    subproblem_optimizer=sub_optimizer,
    max_iterations=100,
    mu=10.0,
)

SchwarzOpt.run_algorithm!(optimizer)