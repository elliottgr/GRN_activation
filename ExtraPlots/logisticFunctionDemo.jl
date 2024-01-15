using Plots


Logistic(x, α = 1.0, β = 0.0, γ = 1.0) = γ/(1+exp(-α * (x - β)))
LeNagardExp(x, α=sqrt(1/2), β=0.0, γ = 1.0) = 1 - (γ * exp(-((x-β)^2/(2*(α^2)))))
xs = collect(-1:0.01:1)
α_set = collect(0:0.1:10)
σ_set = log.(1 .+ collect(0:0.0333:3)) .+ 0.0001
fontsize = 15
α_anim = @gif for i in [α_set; reverse(α_set)]
    plot(xs, Logistic.(xs, i), ylim = (0.0, 1.0), xlabelfontsize = fontsize, ylabelfontsize = fontsize, color = :black, xlabel = "Activation Signal", ylabel = "Gene Expression", legend = :bottomright, label = false, legend_title_font_pointsize = fontsize, legendtitle = "α = $i")
end
σ_anim = @animate for i in [σ_set; reverse(σ_set)]
    plot(xs, LeNagardExp.(xs, i), ylim = (0.0, 1.0), xlabelfontsize = fontsize, ylabelfontsize = fontsize, color = :black, xlabel = "Activation Signal", ylabel = "Gene Expression", legend = :bottomright, label = false, legend_title_font_pointsize = fontsize, legendtitle = "σ = $(round(i, digits = 3))")
end
gif(σ_anim, "invGaussianScale.gif", fps = 30)