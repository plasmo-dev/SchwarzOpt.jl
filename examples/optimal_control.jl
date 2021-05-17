#Example demonstrating the use of overlap to solve a long horizon control problem
using Plasmo, Ipopt
using KaHyPar
using SchwarzSolver

T = 3000          #number of time points
d = sin.(1:T)     #disturbance vector
imbalance = 0.01
distance = 15

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

#TODO: fix attached_node
#@linkconstraint(graph,links[i = 1:T-1], state[i][:x] + control[i][:u] + d[i] == state[i+1][:x],attach = state[i])
for i = 1:T-1
    @linkconstraint(graph, state[i][:x] + control[i][:u] + d[i] == state[i+1][:x],attach = state[i])
end

#Some default convenience methods

# partition_to_subgraphs!(graph,KaHyPar.partition,8;configuration = :connectivity,imbalance = imbalance)
# partition_to_subgraphs!(graph,8;configuration = :connectivity,imbalance = imbalance)


#Partition the problem
hypergraph,hyper_map = gethypergraph(graph) #create hypergraph object based on graph
partition_vector = KaHyPar.partition(hypergraph,16,configuration = :connectivity,imbalance = imbalance)
partition = Partition(hypergraph,partition_vector,hyper_map)
apply_partition!(graph,partition)

#
#supported_structures(SchwarzOpt.Optimizer)
#OR:
#@partition_to_subgraphs KaHyPar.partition(graph,8,configuration = :connectivity,imbalance = imbalance)
#partition_to_subgraphs!(graph,KaHyPar.partition,8,configuration = :connectivity,imbalance = imbalance)
#graph_structure(graph)

#Provide expanded subgraphs
subgraphs = getsubgraphs(graph)
expanded_subgraphs = expand.(Ref(graph),subgraphs,Ref(distance))

# ipopt = optimizer_with_attributes(Ipopt.Optimizer,"print_level" => 0)
# set_optimizer(graph,ipopt)
# optimize!(graph)
#set an optigraph optimizer
set_optimizer(graph,SchwarzSolver.SchwarzOptimizer(graph,expanded_subgraphs;sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"print_level" => 0)))
#set_optimizer(graph,SchwarzOpt.Optimizer(graph,distance))
optimize!(graph)
