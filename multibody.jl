using CSV, DataFrames
using DynamicPolynomials
using MomentOpt
using MosekTools
using LinearAlgebra
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["frame_id","x","y","vx","vy"]]) |>
    (d -> d .- [0 1000 1000 0 0]) |>
    (d -> d ./ [1 20 20 20 20]) |>
    (d -> filter(e -> -1 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> -1 <= e["y"] <= 1, d))

frames = D[:,"frame_id"] .|> Int
counts = [sum(D[:,"frame_id"] .== f) for f in frames]
weight = binomial.(counts, 2)

d = 3
@polyvar t x[1:8] x1[1:4] x2[1:4]
M = sum(DiracMeasure(x1,s) for s in collect.(eachrow(D[:,["x","y","vx","vy"]]))) / length(unique(frames))
K0 = let v0 = monomials(x1,0:d);
    Q = integrate.(v0*v0',[M]);
    v1 = monomials(x1,0:d);
    v2 = monomials(x2,0:d);
    v1'*inv(Q+1e-4I)*v2
end
K =  subs(K0, (x1 .=> x[[1,2,5,6]])..., (x2 .=> x[[3,4,7,8]])...)
Λ1 = subs(K0, (x1 .=> x[[1,2,5,6]])..., (x2 .=> x[[1,2,5,6]])...)
Λ2 = subs(K0, (x1 .=> x[[3,4,7,8]])..., (x2 .=> x[[3,4,7,8]])...)

F = frames[counts .> 1]
f0 = first(F)
X0 = filter(e -> e["frame_id"] == f0, D)
allpairs(d) = [[d[1,"x"],d[1,"y"],d[j,"x"],d[j,"y"],d[1,"vx"],d[1,"vy"],d[j,"vx"],d[j,"vy"]] for j=2:size(d,1)]
x0 = allpairs(X0)[1]
ρ0 = DiracMeasure([t;x],[0;x0])

ϕ = monomials([t;x[1:4]],0:2d)
m = GMPModel(Mosek.Optimizer)
@variable m ρ  Meas([t;x],support=@set([t;x]'*[t;x]<=10))
@variable m ρT Meas([t;x],support=@set([t;x]'*[t;x]<=10 && t==3))
@objective m Min Mom(2K+Λ1+Λ2,ρ)
@constraint m Mom.(differentiate(ϕ,[t;x[1:4]])*[1;x[5:8]],ρ) - Mom.(ϕ,ρT) .== -integrate.(ϕ,ρ0)
optimize!(m)

q1 = let v = monomials(x[1:2],0:d);
    Q = integrate.(v*v',[ρ]);
    v'*inv(Q+1e-4I)*v
end
q2 = let v = monomials(x[3:4],0:d);
    Q = integrate.(v*v',[ρ]);
    v'*inv(Q+1e-4I)*v
end

save("multibody.pdf", Axis([
    Plots.Image((x,y)->1/q1(x,y)+1/q2(x,y),(-1,1),(-1,1)),
    Plots.Quiver(
        D[1:50:end,"x"],    D[1:50:end,"y"],
        D[1:50:end,"vx"]/3, D[1:50:end,"vy"]/3,
        style="-stealth, no markers, blue"
    ),
    Plots.Scatter(reshape(x0[1:4],(2,2))),
    Plots.Scatter(reshape(integrate.(x[1:4],[ρT]),(2,2)),style="red"),
],xmin=-1,xmax=1,ymin=-1,ymax=1))
