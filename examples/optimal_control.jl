#Example demonstrating the use of overlap to solve a long horizon control problem
using Plasmo, Ipopt
using KaHyPar
using SchwarzOpt

T = 100              #number of time points
d = sin.(1:T)        #a disturbance vector
imbalance = 0.05      #partition imbalance
distance = 2      #expand distance
n_parts = 4          #number of partitions

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
    @linkconstraint(graph, state[i][:x] + control[i][:u] + d[i] == state[i+1][:x], attach = state[i+1])
end

#Partition the optigraph using recrusive bisection over a hypergraph
hypergraph,hyper_map = hyper_graph(graph) #create hypergraph object based on graph

partition_vector = KaHyPar.partition(
    hypergraph,
    n_parts,
    configuration = (@__DIR__)*"/cut_kKaHyPar_sea20.ini",
    imbalance = imbalance
)

partition = Partition(hypergraph, partition_vector, hyper_map)
apply_partition!(graph, partition)

#calculate subproblems using expansion distance
subgraphs = getsubgraphs(graph)
expanded_subgraphs = Plasmo.expand.(graph, subgraphs, distance)
sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)


# optimizer = SchwarzOpt.Optimizer(
#     graph,
#     expanded_subgraphs; 
#     sub_optimizer=sub_optimizer, 
#     max_iterations=50,
#     mu=1.0
# )
# SchwarzOpt._initialize_optimizer!(optimizer)

# for subgraph in optimizer.subproblem_graphs
#     JuMP.set_optimizer(subgraph, optimizer.sub_optimizer)
# end

# for subproblem_graph in optimizer.subproblem_graphs
#     SchwarzOpt._update_subproblem!(optimizer, subproblem_graph)
# end

# # sg1 = optimizer.subproblem_graphs[1]
# # xk,lk = SchwarzOpt._do_iteration(sg1)

# for subproblem_graph in optimizer.subproblem_graphs
#     xk,lk = SchwarzOpt._do_iteration(subproblem_graph)
#     println(xk)
#     println(lk)
# end

# inc_edges = SchwarzOpt._find_boundaries(graph, optimizer.subproblem_graphs)

#optimize using schwarz overlapping decomposition
SchwarzOpt.optimize!(
    graph;
    subgraphs = expanded_subgraphs,
    sub_optimizer = sub_optimizer,
    max_iterations = 200,
    mu=0.001
)
