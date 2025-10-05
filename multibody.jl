using CSV, DataFrames
using DynamicPolynomials
using MomentOpt
using MosekTools
using LinearAlgebra
using PGFPlots
using Random
Random.seed!(1)

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["frame_id","x","y","vx","vy"]]) |>
    (d -> d .- [0 1000 1000 0 0]) |>
    (d -> d ./ [1 20 20 20 20]) |>
    (d -> filter(e -> -1 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> -1 <= e["y"] <= 1, d))

frames = unique(D[:,"frame_id"]) .|> Int
counts = [sum(D[:,"frame_id"] .== f) for f in frames]
mframe = frames[counts .> 1]

function sample()
    rf = rand(mframe)
    rd = filter(e -> e["frame_id"] == rf, D)
    ra = rd[randperm(size(rd,1))[1:2],:]
    return ra
end
D2 = [sample() for _ in 1:100] .|>
    (d -> [d[1,"x"],d[1,"y"],d[2,"x"],d[2,"y"],d[1,"vx"],d[1,"vy"],d[2,"vx"],d[2,"vy"]]) |>
    stack

d = 2
@polyvar x[1:8]
M2 = sum(DiracMeasure(x,collect(s)) for s in eachcol(D2)) / size(D2,2)
Λ2 = let v = monomials(x,0:d);
    Q2 = integrate.(v*v',[M2]);
    v'*inv(Q2+1e-4I)*v
end

x0 = D2[:,1]
ϕ = monomials(x[1:4],0:2d)
m = GMPModel(Mosek.Optimizer)
@variable m ρ  Meas(x,support=@set(x'x<=10))
@variable m ρT Meas(x,support=@set(x'x<=10))
ρ0 = DiracMeasure(x,x0)
@objective m Min Mom(Λ2,ρ)
@constraint m Mom.(differentiate(ϕ,x[1:4])*x[5:8],ρ) - Mom.(ϕ,ρT) .== -integrate.(ϕ,ρ0)
@constraint m Mom(1,ρ) == 1
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
        D2[1,:],   D2[2,:],
        D2[5,:]/3, D2[6,:]/3,
        style="-stealth, no markers, blue"
    ),
    Plots.Scatter(reshape(x0[1:4],(2,2)))
],xmin=-1,xmax=1,ymin=-1,ymax=1))
