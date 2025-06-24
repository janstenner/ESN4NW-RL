using PlotlyJS
using FileIO, JLD2



# FileIO.save("optimal.jld2","optimal",optimal)

optimal = FileIO.load("optimal.jld2","optimal")


xx = collect(0:(1/12):23+(11/12))

layout = Layout(
    plot_bgcolor = "white",
    font=attr(
        family="Arial",
        size=16,
        color="black"
    ),
    showlegend = true,
    xaxis = attr(gridcolor = "#E0E0E0FF",
                linecolor = "#888888"),
    yaxis = attr(gridcolor = "#E0E0E0FF",
                linecolor = "#888888",
                range=[0,1]),
    title= "Simulation Episode with Actions from the Optimizer output",
    )




# wind = [generate_wind() for i in 1:n_turbines]
# grid_price = generate_grid_price()

# to_plot = [scatter(x=xx, y=wind[1]) for i in 1:1]

# plot(Vector{AbstractTrace}(to_plot), layout)



p = plot(scatter(x=xx, y=optimal["wind"], name="Wind Power"), layout)
add_trace!(p, scatter(x=xx, y=optimal["grid_price"], name="Grid Price"))
add_trace!(p, scatter(x=xx, y=optimal["actions"], name="Optimal Actions"))

display(p)