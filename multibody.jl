using CSV, DataFrames
using PGFPlots

D = CSV.read("vehicle_tracks_000.csv", DataFrame) |>
    (d -> d[:,["x","y","vx","vy"]]) |>
    (d -> d .- [1000 1000 0 0]) |>
    (d -> d ./ 20) |>
    (d -> filter(e -> -1 <= e["x"] <= 1, d)) |>
    (d -> filter(e -> -1 <= e["y"] <= 1, d))

save("multibody.pdf", Plots.Quiver(
    D[1:50:end,"x"],    D[1:50:end,"y"],
    D[1:50:end,"vx"]/3, D[1:50:end,"vy"]/3,
    style="-stealth, no markers, blue"
))
