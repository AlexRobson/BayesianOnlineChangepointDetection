# Bayesian Online ChangePoint

Implements BOCP. 

This is presently under construction. The intention is to take in a datastream and infer the changepoints and run lengths. 
For example, producing plots like the following to identify change points in a time series. 

![BOCP](./notebooks/bocp.png, "BOCP")


The interface is likely to change, however, at present the following works. 


```
    DIM = 2
    TLENGTH = 10
    data = rand(TLENGTH, DIM) # Generate some random data
    μ = zeros(DIM)
    Σ = Matrix{Float64}(I, 2, 2)
    R, maxes, obslikelihood, pred, dists, L = offline_changepoint_detection(
        data,
        ConstantHazard(2.0),
        ConjugateModel{MvNormal, Float64}(μ, Σ, 2.0, 2.0), # Use (a) conjugate prior to MvNormal 
    ),
)
```

See the notebook script on reproducing the plot.



