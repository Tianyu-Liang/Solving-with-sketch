# Solving-with-sketch

Installation/launch instructions:

1. Download julia from https://julialang.org/downloads/
2. launch julia with 8 threads: julia -t 8
3. After launching julia, to install a package, such as LoopVectorization, do the following:
   ```
    julia> using Pkg
    julia> Pkg.add("LoopVectorization")
    julia> using LoopVectorization
   ```


Quick example run:
```
include("solving_with_sketch.jl")
test_pipeline("matrices/rail2586.mat", "qr", "simple")
```


**The first parameter is the path to the matrix**, the second parameter is the factorization method ("qr" or "svd"), and the third parameter is the multiplication method to use. 
The four multiplication methods are ("simple", "simplepm", "advanced", "advancedpm"), which corresponds to algorithm 3 (-1, 1), algorithm 3 (+- 1), algorithm 4 (-1, 1), and algorithm 4 (+- 1).

See http://sparse.tamu.edu/ for a collection of sparse matrices.

**Important note: running functions in julia for the first time may be slow due to just-in-time compilation. It's recommended that everytime the user makes a change to the file and executes include("filename"), run the function twice.**

Paper link:
https://arxiv.org/abs/2310.15419
