# Trans_Prob
# Transshipment Problem - SQG method
# Author: Paul Tu

using JuMP, CPLEX, LinearAlgebra, StatsBase, Random

Random.seed!(533)
n = 7  # number of retailers
quantiles = true

Iter_Num = 100
s0 = [100.0, 200, 150, 170, 180, 170, 170]  # initial stock level = mean
# s0 = [300,300,300,300,300,300,300]
# s0 = [0,0,0,0,0,0,0]
a = 10  # constant in alpha (step size): (Iter_Num/a)/i

# Costs
# h_i = unit cost of holding inventory
# h = [1,1,1,1,1,1,1]
h = 1
h = [h for i in 1:n]

# c_ij = unit cost of transshipment
# c=[0 0.1 0.1 0.1 0.1 0.1 0.1;
#    0.1 0 0.1 0.1 0.1 0.1 0.1;
#    0.1 0.1 0 0.1 0.1 0.1 0.1;
#    0.1 0.1 0.1 0 0.1 0.1 0.1;
#    0.1 0.1 0.1 0.1 0 0.1 0.1;
#    0.1 0.1 0.1 0.1 0.1 0 0.1;
#    0.1 0.1 0.1 0.1 0.1 0.1 0]
c = 0.1*ones(7,7)-Diagonal(0.1*ones(7,7))

# p_i = penalty cost for shortage
# p = [4,4,4,4,4,4,4]
p = 4
p = [p for i in 1:n]

function get_dtc(s, d)
    # Solve dual problem, return B+E

    # create JuMP Model
    model = Model(CPLEX.Optimizer)
    set_silent(model)

    @variables(model, begin
        B[1:n]
        M[1:n]
        R
        E[1:n]
    end)

    # constraints
    @constraint(model, a6[i=1:n] ,B[i] + E[i] <= h[i])
    @constraint(model, b6[i=1:n], B[i] + M[i] <= 0)
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
    # Calculate averaged dTC using M random demands, return dTC/M
    # Generate m realizations of demands D[1], D[2], ... D[m]
    m = 100
    dTC = [0,0,0,0,0,0,0]
    index = sample(1:3^n,m,replace=false)
    for i in index
        if quantiles
            d = D[i]
        else
            d = D[i,:]
        end
        dTC = dTC + get_dtc(s,d)
    end
    return dTC/m
end


# Main


# Scenario generation for the transshipment problem
# 
using Distributions

mu = [100.0, 200, 150, 170, 180, 170, 170]
sigma = [20.0, 50, 30, 50, 40, 30, 50]

if quantiles
    q = [quantile(Normal(mu[i], sigma[i]), [0.25, 0.5, 0.75]) for i in 1:n]
    D = vec(collect(Iterators.product(q...)))
else
    using DelimitedFiles
    D=open(readdlm,"C:/Users/10447/Desktop/Truncated_Normal_Samples_Space.txt")

    # D = Array{Float64}(undef, 3^n, n)
    # for i in 1:n
    # D[:,i] .= rand(TruncatedNormal(mu[i], sigma[i], mu[i]-3*sigma[i], mu[i]+3*sigma[i]), 3^n)
    # end
end

# Model - One replication
# s = s0
# for i in 1:Iter_Num
#     print("Iteration",i)
#     s = s - ((Iter_Num/a)/i)*get_averaged_dtc(s)
#     for j in 1:length(s)
#         s[j] = max(0, s[j])
#     end
#     # print(s)
# end
# print(s)



# Model - Output for .csv format
# s = s0
# for i in 1:Iter_Num
#     s = s - ((Iter_Num/a)/i)*get_averaged_dtc(s)
#     for j in 1:length(s)
#         s[j] = max(0, s[j])
#     end
#     # print s
#     for k in s[1:n-1]
#         print(k,',')
#     end
#     print(s[n],'\n')
# end


# Model - Repeat 100 times for confidence interval
sums = [0,0,0,0,0,0,0]
for r in 1:100
    s = s0  # initial stock level
    # print('Epoch ',r)
    for i in 1:Iter_Num
        s = s - ((Iter_Num/a)/i)*get_averaged_dtc(s)
        for j in 1:length(s)
            s[j] = max(0, s[j])
        end
    end
    # print s
    for k in s[1:n-1]
        print(k,',')
    end
    print(s[n],'\n')
    sums = sums + s
end
print("Mean = ",sums/100)
s=sums/100



















# All in one for validation
using LinearAlgebra, Distributions, StatsBase, Random

Random.seed!(533)
n = 7  # number of retailers
M = 3^n
validation = true

if validation
    val = s
end

# Costs
# h_i = unit cost of holding inventory
# h = [1,1,1,1,1,1,1]
h = 1
h = [h for i in 1:n]

# c_ij = unit cost of transshipment
# c=[0 0.1 0.1 0.1 0.1 0.1 0.1;
#    0.1 0 0.1 0.1 0.1 0.1 0.1;
#    0.1 0.1 0 0.1 0.1 0.1 0.1;
#    0.1 0.1 0.1 0 0.1 0.1 0.1;
#    0.1 0.1 0.1 0.1 0 0.1 0.1;
#    0.1 0.1 0.1 0.1 0.1 0 0.1;
#    0.1 0.1 0.1 0.1 0.1 0.1 0]
c = 0.1*ones(7,7)-Diagonal(0.1*ones(7,7))

# p_i = penalty cost for shortage
# p = [4,4,4,4,4,4,4]
p = 4
p = [p for i in 1:n]

# Scenario generation for the transshipment problem
mu = [100.0, 200, 150, 170, 180, 170, 170]
sigma = [20.0, 50, 30, 50, 40, 30, 50]


if quantiles
    q = [quantile(Normal(mu[i], sigma[i]), [0.25, 0.5, 0.75]) for i in 1:n]
    D = vec(collect(Iterators.product(q...)))
else
    D = Array{Float64}(undef, 3^n, n)
    for i in 1:n
    D[:,i] .= rand(TruncatedNormal(mu[i], sigma[i], mu[i]-3*sigma[i], mu[i]+3*sigma[i]), 3^n)
    end
end

# using DelimitedFiles
# D=open(readdlm,"C:/Users/10447/Desktop/Truncated_Normal_Samples_Space.txt")

# ----------  Model  ----------
using JuMP, GLPK

# create JuMP Model
model_val = Model(GLPK.Optimizer)
set_silent(model_val)

# variables
@variables(model_val, begin
    S[1:7] >= 0  # stock level
    e[1:M, 1:7] >= 0  # ending inventory
    f[1:M, 1:7] >= 0  # inventory at retailer i used to satisfy demand at retailer i
    q[1:M, 1:7] >= 0  # inventory increase
    r[1:M, 1:7] >= 0  # amount of shortage met after replenishment at retailer i
    t[1:M, 1:7, 1:7] >= 0  # transshipment from i to j
end)

# the objective
@objective(model_val, Min,
    (sum(sum(h.*e[i,:] for i in 1:M)) + sum(sum(c.*t[i,:,:] for i in 1:M)) + sum(sum(p.*r[i,:] for i in 1:M)))/M
)

# constraints
# @constraint(model, t_ii[k=1:M, i=1:n], t[k,i,i] == 0)  # t_ii = 0
@constraint(model_val, a1[k=1:M, i=1:n], f[k,i] + sum(t[k,i,j] for j in 1:7) + e[k,i] == S[i])  # 1a
if quantiles
    @constraint(model_val, b1[k=1:M, i=1:n], f[k,i] + sum(t[k,j,i] for j in 1:7) + r[k,i] == D[k][i])  # 1b
    @constraint(model_val, c1[k=1:M], sum(r[k, 1:7]) + sum(q[k, 1:7]) == sum(D[k]))  # 1c
else
    @constraint(model_val, b1[k=1:M, i=1:n], f[k,i] + sum(t[k,j,i] for j in 1:7) + r[k,i] == D[k,i])  # 1b
    @constraint(model_val, c1[k=1:M], sum(r[k, 1:7]) + sum(q[k, 1:7]) == sum(D[k,:]))  # 1c
end
@constraint(model_val, d1[k=1:M, i=1:n], e[k, i] + q[k, i] == S[i])  # 1d
# constraint for given demand
if validation
    @constraint(model_val, S .== val)
end

# call optimizer
optimize!(model_val)

@show objective_value(model_val)
@show value.(S)
