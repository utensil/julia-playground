Julia Playground
====================

A series of experiments with Julia, to familiarize with Julia and look for
potential applications.

Install Python, INotebook
-----------------------------------------

Install [MiniConda](http://conda.pydata.org/miniconda.html) and then:

```bash
conda install numpy scipy sympy ipython jupyter matplotlib
jupyter notebook
```

Install Julia and IJulia
-----------------------------------------

```bash
# On Mac:
# See https://caskroom.github.io/
brew tap caskroom/cask
brew cask install julia

# On Windows:
# Just install from http://julialang.org/downloads/
```

```julia
Pkg.add("IJulia")
using IJulia
notebook()
```

Julia-Python bridge
-----------------------------------------

```julia
julia> Pkg.add("PyCall")
INFO: Initializing package repository /Users/utensil/.julia/v0.4
INFO: Cloning METADATA from git://github.com/JuliaLang/METADATA.jl
# The first time takes a long time to do git clone......
julia> using PyCall
INFO: Precompiling module PyCall...

julia> @pyimport numpy.random as nr

julia> nr.rand(3,4)
3x4 Array{Float64,2}:
 0.104454  0.847609  0.810417  0.939161
 0.266673  0.186059  0.651118  0.861648
 0.720235  0.164573  0.448754  0.183331
```

Install most installed packages
-----------------------------------------

The following are some of the most installed packages from [Julia Package Pulse](http://pkg.julialang.org/pulse.html).

```julia
# Math
for pkg in ["Mocha", "Distributions", "DataFrames", "JuMP", "Graphs", "GLM", "GeneticAlgorithms", "ControlSystems", "DiscriminantAnalysis", "HTSLIB"]
    Pkg.add(pkg)
end

# Faster
for pkg in ["ParallelAccelerator","NLopt", "Optim"]
    Pkg.add(pkg)
end

# Vis
for pkg in ["Escher", "Plots", "Gadfly", "PyPlot", "GR", "Immerse", "UnicodePlots", "Qwt", "PlotlyJS", "Interact", "Mux", "GLVisualize", "Blink"]
    Pkg.add(pkg)
end

# Utility
# brew install llvm
for pkg in ["Reactive", "Maker", "FactCheck", "BuildExecutable", "Clang"]
    Pkg.add(pkg)
end
```

Note:

* `Clang.jl` is not working yet, it's looking for a missing `llvm-config`.

CUDA related
-----------------------------------------

See [this gitst](https://gist.github.com/mlhales/5785725) for CUDA installation
instructions.

```julia
Pkg.add("CUDArt")
using CUDArt
Pkg.test("CUDArt")
```

Note: Some MacBook Pro are using Intel Iris graphic card, which doesn't support CUDA, try
them somewhere else, with a Nvidia graphic card.

Related links
-----------------------------------------

### Learn

* https://github.com/svaksha/Julia.jl
* https://github.com/utensil-star/awesome-julia
* http://learnxinyminutes.com/docs/julia
* http://rogerluo.cc/slides/contents/lqcc.html#/

### Community

* http://pkg.julialang.org/pulse.html
* http://julialang.cn/

### Misc

* http://www.mkdocs.org/
*
http://stackoverflow.com/questions/2607425/is-google-s-cdn-for-jquery-available-in-china/22060903#22060903
