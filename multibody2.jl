using CSV, DataFrames
using DynamicPolynomials
using MomentOpt
using MosekTools
using LinearAlgebra
using PGFPlots
using XML
include("vehicle.jl")

istype(way, type) = any(
    e.tag == "tag" && e["k"] == "type" && e["v"] == type for e in way.children)

map_node = XML.read("DR_DEU_Roundabout_OF.osm_xy", Node) |>
    (m -> m.children) |> first |>
    (m -> filter(e -> e.tag == "node", m.children)) .|>
    (m -> (m["id"] => [parse(Float64, m["x"]), parse(Float64, m["y"])])) |>
    Dict

map_ways = XML.read("DR_DEU_Roundabout_OF.osm_xy", Node) |>
    (m -> m.children) |> first |>
    (m -> filter(e -> e.tag == "way", m.children)) |>
    (m -> filter(e -> istype(e, "curbstone"), m)) .|>
    (w -> filter(e -> e.tag == "nd", w.children)) .|>
    (w -> stack(map_node[n["ref"]] for n in w)) .|>
    (w -> w .- [1000, 1000]) .|>
    (w -> w ./ 20)

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["frame_id","track_id","x","y","vx","vy"]]) |>
    (d -> d .- [0 0 1000 1000 0 0]) |>
    (d -> d ./ [1 1 20 20 20 20]) |>
    (d -> filter(e -> -1 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> -1 <= e["y"] <= 1, d))

frames = D[:,"frame_id"] .|> Int |> unique
counts = [sum(D[:,"frame_id"] .== f) for f in frames]

d = 3
@polyvar t x[1:8] x1[1:4] x2[1:4]
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
ρ0 = [DiracMeasure([t;x],[0;_x0]) for _x0 in x0]

ϕ = monomials([t;x[1:4]],0:2d)
m = GMPModel(Mosek.Optimizer)
@variable m ρ[i=1:length(x0)]  Meas([t;x],support=@set([t;x]'*[t;x]<=10))
@variable m ρT[i=1:length(x0)] Meas([t;x],support=@set([t;x]'*[t;x]<=10 && t==3))
@objective m Min Mom(2K + Λ1 + Λ2*length(x0), sum(ρ)/length(x0) + sum(ρT)/length(x0))
@constraint m [i=1:length(x0)] Mom.(differentiate(ϕ,[t;x[1:4]])*[1;x[5:8]],ρ[i]) - Mom.(ϕ,ρT[i]) .== -integrate.(ϕ,ρ0[i])
let v = monomials([t;x[[1,2,5,6]]],0:2d); @constraint m [i=2:length(x0)] Mom.(v,ρ[i])  .== Mom.(v,ρ[1]) end
let v = monomials([t;x[[1,2,5,6]]],0:2d); @constraint m [i=2:length(x0)] Mom.(v,ρT[i]) .== Mom.(v,ρT[1]) end
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

xT = [integrate.(x,[ρT[1]]) integrate.(x,ρT')]
save("multibody2-$(fi).pdf", Axis([
    Plots.Image((x,y)->1/q1(x,y)+sum(1/q(x,y) for q in q2),(-1,1),(-1,1));
    Plots.Quiver(
        D[1:50:end,"x"],    D[1:50:end,"y"],
        D[1:50:end,"vx"]/3, D[1:50:end,"vy"]/3,
        style="-stealth, no markers, blue"
    );
    [Plots.Linear(vehicle(x0...,5/20,2/20), style="brown, no markers, solid") for x0 in eachrow(X0[:,["x","y","vx","vy"]]) .|> collect];
    Plots.Linear(vehicle(xT[1],xT[2],xT[5],xT[6],5/20,2/20), style="red, no markers, solid");
    [Plots.Linear(vehicle(xt[3],xt[4],xt[7],xt[8],5/20,2/20), style="red, no markers, solid") for xt in eachcol(xT[:,2:end])];
    [Plots.Linear(vehicle(x1...,5/20,2/20), style="green, no markers, solid") for x1 in eachrow(X1[:,["x","y","vx","vy"]]) .|> collect];
    [Plots.Linear(m, style="white, no markers, solid") for m in map_ways]
],xmin=-1,xmax=1,ymin=-1,ymax=1,xlabel="Easting (20\\,m)",ylabel="Northing (20\\,m)",style="colorbar style={title=Density}"))
end
