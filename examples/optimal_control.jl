#Example demonstrating the use of overlap to solve a long horizon control problem
using Plasmo, Ipopt
using KaHyPar
using SchwarzOpt

T = 200             #number of time points
d = sin.(1:T)       #a disturbance vector
imbalance = 0.1    #partition imbalance
distance = 2        #expand distance
n_parts = 10        #number of partitions

#Create the model (an optigraph)
graph = OptiGraph()

@optinode(graph,state[1:T])
@optinode(graph,control[1:T-1])

for (i,node) in enumerate(state)
    @variable(node,x)
    @constraint(node, x >= 0)
    @objective(node, Min, x^2) #- 2*x*d[i])
end
for node in control
    @variable(node,u)
    @constraint(node, u >= -1000)
    @objective(node, Min, u^2)
end
n1 = state[1]
@constraint(n1,n1[:x] == 0)

for i = 1:T-1
    @linkconstraint(graph, state[i][:x] + control[i][:u] + d[i] == state[i+1][:x])
end

hypergraph,hyper_map = hyper_graph(graph) #create hypergraph object based on graph
partition_vector = KaHyPar.partition(
    hypergraph,
    n_parts,
    configuration = :edge_cut,
    imbalance = imbalance
)
partition = Partition(hypergraph, partition_vector, hyper_map)
apply_partition!(graph, partition)

#calculate subproblems using expansion distance
subgraphs = getsubgraphs(graph)
expanded_subgraphs = Plasmo.expand.(graph, subgraphs, distance)
sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

#optimize using schwarz overlapping decomposition
SchwarzOpt.optimize!(
    graph;
    subgraphs = expanded_subgraphs,
    sub_optimizer = sub_optimizer,
    max_iterations = 200,
    mu=0.0
)