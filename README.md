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

# On Ubuntu:
sudo add-apt-repository ppa:staticfloat/juliareleases
sudo add-apt-repository ppa:staticfloat/julia-deps
sudo apt-get update
sudo apt-get install julia

# On Windows:
# Just install from http://julialang.org/downloads/
```

Start a Jupyter Notebook for Julia
------------------------------------

```julia
using Pkg
Pkg.add("IJulia")
using IJulia
notebook(dir=pwd())
```

A more interesting way:

```bash
# Start Julia REPL
$ julia
julia>

# Enter Pkg REPL-mode
# Hit ]
pkg> add IJulia

# Back to Julia REPL
# Hit Backspace
julia>

# Enter Shell mode or exit to shell
# Hit ;
shell> julia -e "using IJulia;notebook(dir=pwd())"
# or Enter exit()
$ julia -e "using IJulia;notebook(dir=pwd())"
```

Julia-Python bridge
-----------------------------------------

```bash
pkg> add PyCall
  Updating registry at `~/.julia/registries/General`
  Updating git-repo `https://github.com/JuliaRegistries/General.git`
 Resolving package versions...
 Installed MacroTools ─ v0.4.5
 Installed PyCall ───── v1.90.0
  Updating `~/.julia/environments/v1.1/Project.toml`
  [438e738f] + PyCall v1.90.0
  Updating `~/.julia/environments/v1.1/Manifest.toml`
  [1914dd2f] + MacroTools v0.4.5
  [438e738f] + PyCall v1.90.0
  Building PyCall → `~/.julia/packages/PyCall/RQjD7/deps/build.log`

julia> using PyCall
[ Info: Precompiling PyCall [438e738f-606a-5dbb-bf0a-cddfbfd45ab0]

julia> nr = pyimport("numpy.random")
PyObject <module 'numpy.random' from '~/.julia/conda/3/lib/python3.7/site-packages/numpy/random/__init__.py'>

julia> nr.rand(3,4)
3×4 Array{Float64,2}:
 0.729467  0.404296  0.62498   0.239982
 0.229927  0.291334  0.793212  0.655154
 0.212725  0.715652  0.600691  0.985202
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
