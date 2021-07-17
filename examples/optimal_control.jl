#Example demonstrating the use of overlap to solve a long horizon control problem
using Plasmo, Ipopt
using KaHyPar
using SchwarzOpt

T = 100              #number of time points
d = sin.(1:T)        #disturbance vector
imbalance = 0.1
distance = 5
n_parts = 6

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

#Partition the problem
hypergraph,hyper_map = hyper_graph(graph) #create hypergraph object based on graph
partition_vector = KaHyPar.partition(hypergraph,n_parts,configuration = "cut_rKaHyPar_sea20.ini",imbalance = imbalance)
partition = Partition(hypergraph,partition_vector,hyper_map)
apply_partition!(graph,partition)
println(graph_structure(graph))

#or partition_to_subgraphs(KaHyPar.partition)

#calculate expanded subgraphs
subgraphs = getsubgraphs(graph)
expanded_subgraphs = Plasmo.expand.(graph,subgraphs,distance)
sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"print_level" => 0)

#optimize using the optimizer
# optimizer = SchwarzOpt.Optimizer(graph,expanded_subgraphs;
# sub_optimizer = sub_optimizer,
# max_iterations = 50)
# SchwarzOpt.optimize!(optimizer)

# set_optimizer(graph,Ipopt.Optimizer)
# Plasmo.optimize!(graph)


#optimize the graph
SchwarzOpt.optimize!(graph;
subgraphs = expanded_subgraphs,
sub_optimizer = sub_optimizer,
max_iterations = 50)
