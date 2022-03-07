# Trans_Prob
# Transshipment Problem - SQG method
# Author: Paul Tu

using JuMP, CPLEX

# Costs
h=[1,2,3,4,5,6,7]  # h_i = unit cost of holding inventory
c=[0 1 1 1 1 1 1; 1 0 1 1 1 1 1; 1 0 0 1 1 1 1; 1 0 1 0 1 1 1; 1 0 1 1 0 1 1; 1 0 1 0 0 0 1; 1 0 1 1 1 1 0]  # c_ij = unit cost of transshipment
p=[7,6,5,4,3,2,1]  # p_i = penalty cost for shortage

n = 7  # number of retailers


function get_dtc(s, d)
    # Dual Problem

    
    # create JuMP Model
    model = Model(CPLEX.Optimizer)
    set_silent(model)

    # @variables(model, begin
    #     B[1:n]
    #     M[1:n]
    #     R
    #     E[1:n]
    # end)

    @variable(model, B[1:n])
    @variable(model, M[1:n])
    @variable(model, R)
    @variable(model, E[1:n])

    # constraints
    @constraint(model, a6[i=1:n] ,B[i] + E[i] <= h[i])
    # @constraint(model, b6[i=1:n], B[i] + M[i] <= 0)
    @constraint(model, c6[i=1:n, j=1:n], B[i] + M[j] <= c[i,j])
    @constraint(model, d6[i=1:n], M[i] + R <= p[i])
    @constraint(model, e6[i=1:n], R + E[i] <= 0)

    # the objective
    @objective(model, Max,
    sum(s.*B) + sum(d.*M) + sum(d.*R) + sum(s.*E)
    )

    # call optimizer
    optimize!(model)

    return JuMP.value.(B)+JuMP.value.(E)
end

function get_averaged_dtc(s)
    # Generate M realizations of demands D[1], D[2], ... D[m]
    # ...
    # ...
    M = length(D)
    dTC = [0,0,0,0,0,0,0]
    for i in 1:M
        d = D[i]
        dTC = dTC + get_dtc(s,d)
    end
    return dTC/M
end


# Main

s=[100.0, 200, 150, 170, 180, 170, 170]  # initial stock level = mean

# Scenario generation for the transshipment problem
# 
using Distributions

mu = [100.0, 200, 150, 170, 180, 170, 170]
sigma = [20.0, 50, 30, 50, 40, 30, 50]

q = [
    quantile(Normal(mu[i], sigma[i]), [0.25, 0.5, 0.75])
    for i in 1:7
]

D = vec(collect(Iterators.product(q...)))

# Model
for i in 1:10
    s = s - (10/i)*get_averaged_dtc(s)
    s = max([0,0,0,0,0,0,0], s)
end

print(s)
