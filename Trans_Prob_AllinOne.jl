# Trans_Prob
# Transshipment Problem - All-in-one method
# Author: Paul Tu

using LinearAlgebra, Distributions, StatsBase, Random

Random.seed!(533)
n = 7  # number of retailers
M = 3^n
quantiles = false
validation = false

if validation
    val = [161.22375627724787,163.46865053269502,161.8924481741567,162.69945380294257,162.92200182092424,162.53926016507106,162.70437999498716]
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
model = Model(GLPK.Optimizer)
set_silent(model)

# variables
@variables(model, begin
    s[1:7] >= 0  # stock level
    e[1:M, 1:7] >= 0  # ending inventory
    f[1:M, 1:7] >= 0  # inventory at retailer i used to satisfy demand at retailer i
    q[1:M, 1:7] >= 0  # inventory increase
    r[1:M, 1:7] >= 0  # amount of shortage met after replenishment at retailer i
    t[1:M, 1:7, 1:7] >= 0  # transshipment from i to j
end)

# the objective
@objective(model, Min,
    (sum(sum(h.*e[i,:] for i in 1:M)) + sum(sum(c.*t[i,:,:] for i in 1:M)) + sum(sum(p.*r[i,:] for i in 1:M)))/M
)

# constraints
# @constraint(model, t_ii[k=1:M, i=1:n], t[k,i,i] == 0)  # t_ii = 0
@constraint(model, a1[k=1:M, i=1:n], f[k,i] + sum(t[k,i,j] for j in 1:7) + e[k,i] == s[i])  # 1a
if quantiles
    @constraint(model, b1[k=1:M, i=1:n], f[k,i] + sum(t[k,j,i] for j in 1:7) + r[k,i] == D[k][i])  # 1b
    @constraint(model, c1[k=1:M], sum(r[k, 1:7]) + sum(q[k, 1:7]) == sum(D[k]))  # 1c
else
    @constraint(model, b1[k=1:M, i=1:n], f[k,i] + sum(t[k,j,i] for j in 1:7) + r[k,i] == D[k,i])  # 1b
    @constraint(model, c1[k=1:M], sum(r[k, 1:7]) + sum(q[k, 1:7]) == sum(D[k,:]))  # 1c
end
@constraint(model, d1[k=1:M, i=1:n], e[k, i] + q[k, i] == s[i])  # 1d
# constraint for given demand
if validation
    @constraint(model, s .== val)
end

# call optimizer
optimize!(model)

@show objective_value(model)
@show value.(s)



# Result - All in one
# ERROR: CPLEX Error  1016: Community Edition. Problem size limits exceeded. Purchase at http://ibm.biz/error1016.
# Quantiles - All in one Result
# objective_value(model) = 85.99173758969357
# [100.0, 200.0, 150.0, 170.0, 206.97959000784329, 190.23469250588246, 176.74489750196076]  # sum = 1193.959


# Truncated Normal - All in one Result
# objective_value(model) = 149.95661954263775
# value.(s) = [105.851137399196, 215.5923521980182, 160.19279618485416, 187.30244582769106, 191.78963107120327, 179.327281230806, 184.7710442994231]


# Result - validation







# Result - Quantile (seed = 533)
# objective_value(model) = 85.99173758968595
# value.(s) = [100.0, 200.0, 150.0, 170.0, 186.7448975019608, 183.48979500392159, 203.72448750980408]



# Result - Truncated Normal Distribution
# All-in-one result (seed = 533)
# objective_value(model) = 149.95661954263775
# value.(s) = [105.851137399196, 215.5923521980182, 160.19279618485416, 187.30244582769106, 191.78963107120327, 179.327281230806, 184.7710442994231]

# SGD result(1 replication, seed = 533) in all-in-one
# [111.82438128008175, 212.36882802126004, 161.99856844481647, 182.4036459657501, 192.1945118560511, 182.05006167390692, 182.261717485734]
# objective_value(model) = 150.01465995156818

# SGD result(100 replications, seed = 533) in all-in-one
# [111.7651, 212.3338, 162.1081, 182.3973, 192.2231, 182.051, 182.3198]
# objective_value(model) = 150.01511885773476