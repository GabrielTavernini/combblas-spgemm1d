include("combblas.jl")
using SparseArrays
using MatrixMarket

dpath = ""
hostname = gethostname()
if startswith(hostname, "login") || startswith(hostname,"nid") 
    dpath = "/global/cfs/cdirs/m1982/yuxihong/spgemm1ddataset/"
else
    dpath = "/home/hongy0a/graphclustering/dataset/"
end
@time t = combblas.mmread(joinpath(dpath, "G43.mtx"))
# mgtriplet = combblas.triplet_matrix()
@time I = combblas.getrows(t)
@time J = combblas.getcols(t)
@time V = combblas.getvalues(t)
@time sparray = sparse(I,J,V) # in CSC format

m,n = combblas.getshape(t)
println(m,n)

# println(sparray.colptr)

mutable struct MyMutableStruct
    data::Vector{Int}
end


a = MyMutableStruct([1,2,3]);
println(a.data)
a.data[1] = 444;
println(a.data)
# mgcsc = sparse(combblas.getrows(mgtriplet),combblas.getcols(mgtriplet),combblas.getvalues(mgtriplet))

# matshape = triplet_matrix.shape()
# println(matshape)
