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
    (d -> d[:,["frame_id","x","y","vx","vy"]]) |>
    (d -> d .- [0 1000 1000 0 0]) |>
    (d -> d ./ [1 20 20 20 20]) |>
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
f0 = 177
X0 = filter(e -> e["frame_id"] == f0, D)
allpairs(d) = [[d[i,"x"],d[i,"y"],d[j,"x"],d[j,"y"],d[i,"vx"],d[i,"vy"],d[j,"vx"],d[j,"vy"]] for i=1:size(d,1)-1 for j=i+1:size(d,1)]
x0 = allpairs(X0)[8]
ρ0 = DiracMeasure([t;x],[0;x0])

X1 = filter(e -> e["frame_id"] == f0+3*10, D)
x1 = allpairs(X1)[5]

ϕ = monomials([t;x[1:4]],0:2d)
m = GMPModel(Mosek.Optimizer)
@variable m ρ  Meas([t;x],support=@set([t;x]'*[t;x]<=10))
@variable m ρT Meas([t;x],support=@set([t;x]'*[t;x]<=10 && t==3))
@objective m Min Mom(2K+Λ1+Λ2,ρ+ρT)
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

xT = integrate.(x,[ρT])
save("multibody.pdf", Axis([
    Plots.Image((x,y)->1/q1(x,y)+1/q2(x,y),(-1,1),(-1,1));
    Plots.Quiver(
        D[1:50:end,"x"],    D[1:50:end,"y"],
        D[1:50:end,"vx"]/3, D[1:50:end,"vy"]/3,
        style="-stealth, no markers, blue"
    );
    Plots.Linear(vehicle(x0[1],x0[2],x0[5],x0[6],5/20,2/20), style="brown, no markers, solid");
    Plots.Linear(vehicle(x0[3],x0[4],x0[7],x0[8],5/20,2/20), style="brown, no markers, solid");
    Plots.Linear(vehicle(xT[1],xT[2],xT[5],xT[6],5/20,2/20), style="red, no markers, solid");
    Plots.Linear(vehicle(xT[3],xT[4],xT[7],xT[8],5/20,2/20), style="red, no markers, solid");
    Plots.Linear(vehicle(x1[1],x1[2],x1[5],x1[6],5/20,2/20), style="green, no markers, solid");
    Plots.Linear(vehicle(x1[3],x1[4],x1[7],x1[8],5/20,2/20), style="green, no markers, solid");
    [Plots.Linear(m, style="white, no markers, solid") for m in map_ways]
],xmin=-1,xmax=1,ymin=-1,ymax=1))
