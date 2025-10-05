using CSV, DataFrames
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["frame_id","x","y","vx","vy"]]) |>
    (d -> d .- [0 1000 1000 0 0]) |>
    (d -> d ./ [1 20 20 20 20]) |>
    (d -> filter(e -> -1 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> -1 <= e["y"] <= 1, d))

frames = unique(D[:,"frame_id"]) .|> Int
counts = [sum(D[:,"frame_id"] .== f) for f in frames]
mframe = frames[counts .> 1]
save("agents.pdf", Plots.Histogram(counts))

function sample()
    rf = rand(mframe)
    rd = filter(e -> e["frame_id"] == rf, D)
    ra = rd[randperm(size(rd,1))[1:2],:]
    return ra
end
D2 = [sample() for _ in 1:1000] .|>
    (d -> [d[1,"x"],d[1,"y"],d[2,"x"],d[2,"y"],d[1,"vx"],d[1,"vy"],d[2,"vx"],d[2,"vy"]]) |>
    stack

save("multibody.pdf", Axis([
    Plots.Quiver(
        D2[1,:],   D2[2,:],
        D2[5,:]/3, D2[6,:]/3,
        style="-stealth, no markers, blue"
    ),
    Plots.Quiver(
        D2[3,:],   D2[4,:],
        D2[7,:]/3, D2[8,:]/3,
        style="-stealth, no markers, red"
    ),
]))
