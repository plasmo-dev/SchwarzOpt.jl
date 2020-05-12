using Plasmo
using KaHyPar
using Ipopt
using SchwarzSolver



n_nodes = 100  #number of intervals
N = 3000
k = round(Int64,N/n_nodes)
d = sin.(1:N)

disturbance_matrix = reshape(d, (k, div(length(d), k)))

graph = ModelGraph()
@node(graph,nodes[1:n_nodes])
for (i,node) in enumerate(nodes)
    node_disturbance = disturbance_matrix[:,i]
    M = length(node_disturbance)
    @variable(node, x[1:M+1])
    @variable(node, u[1:M])
    @constraint(node, dynamics[i in 1:M], x[i+1] == x[i] + u[i] + d[i])
    @objective(node, Min, 0.001*sum(x[i]^2 - 2*x[i]*d[i] for i in 1:M) + sum(u[i]^2 for i in 1:M))
end

n1 = getnode(graph,1)
@constraint(n1,n1[:x][1] == 0)                      #First node in planning horizon has initial condition

links = []
for i = 1:n_nodes - 1
    ni = getnode(graph,i)
    nj = getnode(graph,i+1)
    lref = @linkconstraint(graph, ni[:x][end] == nj[:x][1])  #last state in partition i is first state in partition
    push!(links,lref)
end

#Partition the problem
hypergraph,hyper_map = gethypergraph(graph) #create hypergraph object based on graph
partition_vector = KaHyPar.partition(hypergraph,8,configuration = :connectivity,imbalance = imbalance)
partition = Partition(hypergraph,partition_vector,hyper_map)
make_subgraphs!(graph,partition)



overlap = 1
# schwarz_solve(graph,overlap;sub_optimizer = with_optimizer(Ipopt.Optimizer,tol = 1e-12,print_level = 0),max_iterations = 100,tolerance = 1e-10)

schwarz_solve(graph,overlap;sub_optimizer = with_optimizer(Ipopt.Optimizer,tol = 1e-12,print_level = 0),max_iterations = 100,tolerance = 1e-10,
dual_links = [],primal_links = [])

# schwarz_solve(graph,overlap;sub_optimizer = with_optimizer(Gurobi.Optimizer,OutputFlag=0,BarConvTol=1e-10,FeasibilityTol=1e-9,OptimalityTol= 1e-9),max_iterations = 100,tolerance = 1e-10)
