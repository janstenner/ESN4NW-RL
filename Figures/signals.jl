using PlotlyJS
using FileIO, JLD2



# FileIO.save("signals.jld2","signals",signals)

signals = FileIO.load("signals.jld2","signals")


xx = collect(0:(1/12):23+(11/12))

layout = Layout(
    plot_bgcolor = "white",
    font=attr(
        family="Arial",
        size=16,
        color="black"
    ),
    showlegend = false,
    )




# wind = [generate_wind() for i in 1:n_turbines]
# grid_price = generate_grid_price()

# to_plot = [scatter(x=xx, y=wind[1]) for i in 1:1]

# plot(Vector{AbstractTrace}(to_plot), layout)




layout_p1 = Layout( xaxis = attr(gridcolor = "#aaaaaa",
                                linecolor = "#888888"),
                    yaxis = attr(gridcolor = "#aaaaaa",
                                linecolor = "#888888",
                                range=[0,1]),
                    title= "Wind Power",
)

p1 = plot(scatter(x=xx, y=signals[1]), layout_p1)
add_trace!(p1, scatter(x=xx, y=signals[2]))
add_trace!(p1, scatter(x=xx, y=signals[3]))


layout_p2 = Layout( xaxis = attr(gridcolor = "#aaaaaa",
                                linecolor = "#888888"),
                    yaxis = attr(gridcolor = "#aaaaaa",
                                linecolor = "#888888",
                                range=[0,1]),
                    title= "Grid Price",
)

p2 = plot(scatter(x=xx, y=signals[4]), layout_p2)
add_trace!(p2, scatter(x=xx, y=signals[5]))
add_trace!(p2, scatter(x=xx, y=signals[6]))


p = [p1 p2]
relayout!(p, layout.fields)
display(p)