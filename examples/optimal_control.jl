#Example demonstrating the use of overlap to solve a long horizon control problem
using Plasmo, Ipopt
using KaHyPar
using SchwarzOpt

T = 100              #number of time points
d = sin.(1:T)        #disturbance vector
imbalance = 0.01
distance = 1
n_parts = 4

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

#TODO: fix attached_node for macro
#@linkconstraint(graph,links[i = 1:T-1], state[i][:x] + control[i][:u] + d[i] == state[i+1][:x],attach = state[i])
for i = 1:T-1
    @linkconstraint(graph, state[i][:x] + control[i][:u] + d[i] == state[i+1][:x],attach = state[i+1])
end

#Partition the problem
hypergraph,hyper_map = hyper_graph(graph) #create hypergraph object based on graph
partition_vector = KaHyPar.partition(hypergraph,n_parts,configuration = :edge_cut,imbalance = imbalance)
partition = Partition(hypergraph,partition_vector,hyper_map)
apply_partition!(graph,partition)
println(graph_structure(graph))

#calculate expanded subgraphs
subgraphs = getsubgraphs(graph)
expanded_subgraphs = Plasmo.expand(graph,subgraphs,distance)

#This will initialize the first time
SchwarzOpt.optimize!(graph;
subgraphs = expanded_subgraphs,
sub_optimizer = sub_optimizer,
max_iterations = 50)
