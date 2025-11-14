using CSV, DataFrames
using DynamicPolynomials
using MomentOpt
using MosekTools
using LinearAlgebra
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["frame_id","track_id","x","y","vx","vy"]]) |>
    (d -> d .- [0 0 1000 1000 0 0]) |>
    (d -> d ./ [1 1 20 20 20 20]) |>
    (d -> filter(e -> -1 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> -1 <= e["y"] <= 1, d))

frames = D[:,"frame_id"] .|> Int |> unique
counts = [sum(D[:,"frame_id"] .== f) for f in frames]
weight = binomial.(counts, 2)

d = 2
@polyvar t x[1:8] x1[1:4] x2[1:4] u[1:4]
M = sum(DiracMeasure(x1,s) for s in collect.(eachrow(D[:,["x","y","vx","vy"]]))) / length(frames)
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
for (fi,f0) in enumerate(F[1:10:100])
X0 = filter(e -> e["frame_id"] == f0, D)
allpairs(d) = [[d[1,"x"],d[1,"y"],d[j,"x"],d[j,"y"],d[1,"vx"],d[1,"vy"],d[j,"vx"],d[j,"vy"]] for j=2:size(d,1)]
x0 = allpairs(X0)
ρ0 = [DiracMeasure([t;x;u],[0;_x0;0;0;0;0]) for _x0 in x0]

σ = Diagonal(2*[0.003,0.003,0.03,0.03,0.003,0.003,0.03,0.03])
ϕ = monomials([t;x],0:2d)
m = GMPModel(Mosek.Optimizer)
@variable m ρ[i=1:length(x0)]  Meas([t;x;u],support=@set([t;x;u]'*[t;x;u]<=10))
@variable m ρT[i=1:length(x0)] Meas([t;x;u],support=@set([t;x;u]'*[t;x;u]<=10 && t==3))
@objective m Min Mom(2K + Λ1 + Λ2*length(x0) + u'u, sum(ρ)/length(x0))
@constraint m [i=1:length(x0),j=1:length(ϕ)] Mom(differentiate(ϕ[j],[t;x])'*[1;x[5:8];u] + 0.5*tr(σ*σ'*differentiate(differentiate(ϕ[j],x),x)),ρ[i]) - Mom(ϕ[j],ρT[i]) == -integrate(ϕ[j],ρ0[i])
let v = monomials([t;x[[1,2,5,6]];u[[1,2]]],0:2d); @constraint m [i=2:length(x0)] Mom.(v,ρ[i])  .== Mom.(v,ρ[1]) end
let v = monomials([t;x[[1,2,5,6]];u[[1,2]]],0:2d); @constraint m [i=2:length(x0)] Mom.(v,ρT[i]) .== Mom.(v,ρT[1]) end
optimize!(m)

q1 = let v = monomials(x[1:2],0:d);
    Q = integrate.(v*v',[ρ[1]]);
    v'*inv(Q+1e-4I)*v
end
q2 = [let v = monomials(x[3:4],0:d);
    Q = integrate.(v*v',[ρ[i]]);
    v'*inv(Q+1e-4I)*v
end for i=1:length(x0)]

f1 = f0 + 3*10
id0 = D |>
    (m -> filter(e -> e["frame_id"] == f0, m)) |>
    (m -> m[:,"track_id"]) |> unique
X1 = D |> 
    (m -> filter(e -> e["frame_id"] == f1, m)) |>
    (m -> filter(e -> e["track_id"] ∈ id0, m))

save("multibody2-$(fi).pdf", Axis([
    Plots.Image((x,y)->1/q1(x,y)+sum(1/q(x,y) for q in q2),(-1,1),(-1,1)),
    Plots.Quiver(
        D[1:50:end,"x"],    D[1:50:end,"y"],
        D[1:50:end,"vx"]/3, D[1:50:end,"vy"]/3,
        style="-stealth, no markers, blue"
    ),
    Plots.Scatter(X0),
    Plots.Scatter([integrate.(x[1:2],[ρT[1]]) integrate.(x[3:4],transpose(ρT))],mark="x",style="red"),
    Plots.Scatter(X1,mark="x",style="green"),
],xmin=-1,xmax=1,ymin=-1,ymax=1))
end
