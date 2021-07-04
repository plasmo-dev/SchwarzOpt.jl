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

#supported_structures(SchwarzOpt.Optimizer)

# Some default convenience methods
# partition_to_subgraphs!(graph,KaHyPar.partition,8;configuration = :connectivity,imbalance = imbalance)
# partition_to_subgraphs!(graph,8;configuration = :connectivity,imbalance = imbalance) #Metis by default
# partition_to_tree!(graph,8) #edge_graph
# partition_to_linked_tree!(graph)

#Partition the problem
hypergraph,hyper_map = hyper_graph(graph) #create hypergraph object based on graph
partition_vector = KaHyPar.partition(hypergraph,n_parts,configuration = :edge_cut,imbalance = imbalance)
partition = Partition(hypergraph,partition_vector,hyper_map)
apply_partition!(graph,partition)

println(graph_structure(graph))

#Provide expanded subgraphs
subgraphs = getsubgraphs(graph)
expanded_subgraphs = expand.(Ref(graph),subgraphs,Ref(distance))
#expand(graph,subgraphs,5)

# Test ipopt solution
# ipopt = optimizer_with_attributes(Ipopt.Optimizer,"print_level" => 0)
# set_optimizer(graph,ipopt)

#optimize with optimizer directly to use custom subproblems
sub_optimizer = optimizer_with_attributes(Ipopt.Optimizer,"print_level" => 0)
optimizer = SchwarzOpt.Optimizer(graph,expanded_subgraphs;
sub_optimizer = sub_optimizer,
max_iterations = 50)
optimize!(optimizer)


#use default settings
optimizer = SchwarzOpt.Optimizer
set_optimizer(graph,optimizer)
optimize!(graph)



#optimizer = SchwarzOpt.Optimizer
#set_optimizer(graph,optimizer) #will call Optimizer(graph,kwargs...)
#optimizer = optimizer_with_attributes(SchwarzOpt.Optimizer,)
#set_optimizer(graph,optimizer)
#optimize!(graph)


# set_optimizer(graph,SchwarzOpt.Optimizer(graph,expanded_subgraphs;
# sub_optimizer = optimizer_with_attributes(Gurobi.Optimizer),
# max_iterations = 50)
# )

#TODO: there shouldn't be more than two subgraphs on the dual check.  something is wrong there
#non-contiguous partitions can cause problems
