using LinearAlgebra
using Random
using Distributions
using LoopVectorization
using RandomNumbers
using SparseArrays
using Random123
using MAT
using IterativeSolvers

struct SparseMatrixBlockedCSC{Tv,Ti<:Integer}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    rowptr::Vector{Ti}      # row j is in rowptr[j]:(rowptr[j+1]-1)
    colval::Vector{Ti}      
    nzval::Vector{Tv}       
    block_count::Int
    block_idx_range::Vector{Ti}
end



function multithread_blocked_dense_sparse_pm1!(A, A_hat, d, outer_n=100, outer_d=15000)

    # rng
    #d = Binomial(1, 0.5)
    #brng = BasicRandomNumberGenerator(VSL_BRNG_MT19937, 12345);
    #u = Uniform(brng, 0.0, 1.0);
    

    # size parameters
    m, n = size(A)
    A_hat .= 0
    
    

    #blocking sizes
    # outer_n = 600 #3000
    # outer_d = 10000 #3000

    #temporary variables and storage
    rowval = A.rowval
    nzval = A.nzval
    
    
    s2 = 0

    t1 = time()
    @inbounds Threads.@threads for i = 1 : outer_n : n
        @inbounds for j = 1 : outer_d : d
        
        
            xo_seed = ones(UInt64, 4)
            r = Xoshiro(xo_seed)
        
            #space = zeros(Int32, outer_d);
            space = zeros(Int8, min(d - j + 1, outer_d));
            @inbounds for i_outer_n = i : min(i + outer_n - 1, n)
                @inbounds for spidx = A.colptr[i_outer_n] : A.colptr[i_outer_n + 1] - 1
                
                    #t2 = time()
                    Random.setstate!(r, UInt64(j), UInt64(rowval[spidx]), UInt64(1), UInt64(1))
                    dbound = min(d - j + 1, outer_d)
                    rand!(r, space)
                    #vmap!(f_scale, space, space)
                    @turbo space .= (space .>> 7) .* 2 .+ 1 
                    #s2 += time() - t2
                        
                    
                
                    @inbounds @turbo for k = 1 : dbound
                        A_hat[j + k - 1, i_outer_n] += space[k] * nzval[spidx]
                    end
                end

            end

            
        end
    end

    # restore the state
    elasped_time = time() - t1
    println("blocked algorithm flops: ", nnz(A) * d * 2)
    println("blocked algorithm elapsed time: ", elasped_time)
    println("blocked algorithm Gflops: ", nnz(A) * d * 2 / elasped_time / 1000000000)
    println("s2: ", s2)
end

function multithread_blocked_dense_sparse!(A, A_hat, d, outer_n=100, outer_d=15000)

    # rng
    #d = Binomial(1, 0.5)
    #brng = BasicRandomNumberGenerator(VSL_BRNG_MT19937, 12345);
    #u = Uniform(brng, 0.0, 1.0);
    

    # size parameters
    m, n = size(A)
    
    
    

    #blocking sizes
    # outer_n = 300 #3000
    # outer_d = 5000 #3000

    #temporary variables and storage
    rowval = A.rowval
    slot = zeros(Int8, n)
    
    
    s2 = 0

    t1 = time()
    nzval = zeros(length(A.nzval))
    for i = 1 : length(nzval)
        nzval[i] = A.nzval[i]
    end
    @tturbo nzval ./= 2147483647
    @turbo A_hat .= 0
    @inbounds Threads.@threads for i = 1 : outer_n : n
        
        xo_seed = ones(UInt64, 4)
        r = Xoshiro(xo_seed)
        
        @inbounds for j = 1 : outer_d : d
            #space = zeros(Int32, outer_d);
            space = zeros(Int32, min(d - j + 1, outer_d));
            @inbounds for i_outer_n = i : min(i + outer_n - 1, n)
                @inbounds for spidx = A.colptr[i_outer_n] : A.colptr[i_outer_n + 1] - 1
                
                    #t2 = time()
                    Random.setstate!(r, UInt64(j), UInt64(rowval[spidx]), UInt64(1), UInt64(1))
                    dbound = min(d - j + 1, outer_d)
                    rand!(r, space)
                    #s2 += time() - t2
                        
                    
                
                    @inbounds @turbo for k = 1 : dbound
                        A_hat[j + k - 1, i_outer_n] += space[k] * nzval[spidx]
                    end
                end

            end

            
        end
    end

    elasped_time = time() - t1
    println("blocked algorithm flops: ", nnz(A) * d * 2)
    println("blocked algorithm elapsed time: ", elasped_time)
    println("blocked algorithm Gflops: ", nnz(A) * d * 2 / elasped_time / 1000000000)
    println("s2: ", s2)
end



# blocked csc format, where each block is stored in csr (quite memory intensive)
function produce_blocked_csc(A::SparseMatrixCSC, block_size::Int)
    rowval = A.rowval
    nzval = A.nzval
    colptr = A.colptr
    n = A.n
    m = A.m


    # preallocate memory for new data structure
    block_num = Int64(ceil(n / block_size))
    new_rowptr = zeros(Int64, (m + 1) * block_num)
    new_colval = zeros(Int64, length(rowval))
    new_nzval = zeros(length(nzval))

    
    # calculate the index range for the new blocks
    block_idx_range = zeros(Int64, block_num + 1)
    block_idx_range[1] = 1
    for j = 1 : block_num
        block_idx_range[j + 1] = colptr[1 + min(block_size * j, n)] - colptr[1 + block_size * (j - 1)]
    end
    cumsum!(block_idx_range, block_idx_range)
    total_threads = Threads.nthreads()

    Threads.@threads for outer_thread = 1 : total_threads
        block_per_thread = Int64(ceil(block_num / total_threads))

        list_of_array_col = Vector{Vector{Int64}}(undef, m)
        list_of_array_val = Vector{Vector{Float64}}(undef, m)
        for i in eachindex(list_of_array_col)
            list_of_array_col[i] = zeros(Int64, 0)
            list_of_array_val[i] = zeros(Float64, 0)
        end
        #for j = 1 : block_size : n
        for j = ((outer_thread - 1) * block_per_thread) * block_size + 1 : block_size : min(outer_thread * block_per_thread * block_size, n)
            # create list of list to store the new col indices and values
            
            for i in eachindex(list_of_array_col)
                resize!(list_of_array_col[i], 0)
                resize!(list_of_array_val[i], 0)
            end
            
            # add the values to list of list (essentially transposing)
            for i = j : min(j + block_size - 1, n)
                for ctr = colptr[i] : colptr[i + 1] - 1
                    push!(list_of_array_col[rowval[ctr]], i)
                    push!(list_of_array_val[rowval[ctr]], nzval[ctr])
                end
            end
            # accumulate all into new_colval and new_nzval
            new_rowptr_idx = Int64((j - 1) / block_size) * (m + 1) + 1
            my_idx = block_idx_range[Int64((j - 1) / block_size + 1)]
            new_rowptr[new_rowptr_idx] = 1
            new_rowptr_idx += 1
            for i in eachindex(list_of_array_col)
                temp_veccol = list_of_array_col[i]
                temp_vecval = list_of_array_val[i]
                for v in eachindex(temp_veccol)
                    new_colval[my_idx] = temp_veccol[v]
                    new_nzval[my_idx] = temp_vecval[v]
                    my_idx += 1
                end
                new_rowptr[new_rowptr_idx] = length(temp_veccol) + new_rowptr[new_rowptr_idx - 1]
                new_rowptr_idx += 1
            end

        end
    end

    return SparseMatrixBlockedCSC(m, n, new_rowptr, new_colval, new_nzval, block_num, block_idx_range)
    

end


function advanced_blocked_dense_sparse!(A, A_hat, d, outer_n=2000, outer_d=1200)

    # rng

    # size parameters
    m, n = size(A)
    
    # 800, 3000
    # 2000, 1200
    

    #blocking sizes
    # outer_n = 3000 # 200
    # outer_d = 1000 # 4000
    #S = rand(Int32, d, m)
    #temporary variables and storage
    produce_time = time()
    blocked_csc = produce_blocked_csc(A, outer_n)
    println("produce blocked time: ", time() - produce_time)
    colval = blocked_csc.colval
    nzval = blocked_csc.nzval
    
    slot = zeros(Int8, n)
    c1 = 0
    num_sampled = 0
    
    t1 = time()
    @turbo A_hat .= 0
    @tturbo nzval ./= 2147483647
    copy_list = [zeros(outer_d, outer_n) for i = 1 : Threads.nthreads()]
    @inbounds Threads.@threads for i = 1 : outer_n : n
        
        copy_block = copy_list[Threads.threadid()]
        #copy_block = zeros(outer_d, outer_n)
        xo_seed = ones(UInt64, 4)
        r = Xoshiro(xo_seed)
        block_idx = Int64(Int64((i - 1) / outer_n)) + 1
        offset = blocked_csc.block_idx_range[Int64((i - 1) / outer_n) + 1]
        

        @inbounds for j = 1 : outer_d : d
            #space = zeros(Int32, outer_d);
            space = zeros(Int32, min(d - j + 1, outer_d));

            # perform copy optimization
            @inbounds for o1 = 1 : min(n - i + 1, outer_n)
                for o2 = 1 : min(d - j + 1, outer_d)
                    copy_block[o2, o1] = A_hat[j + o2 - 1, i + o1 - 1]
                end       
            end

            @inbounds for k = 1 : m

                
                dbound = min(d - j + 1, outer_d)
                startp = blocked_csc.rowptr[(block_idx - 1) * (m + 1) + k]
                endp = blocked_csc.rowptr[(block_idx - 1) * (m + 1) + k + 1] - 1
                
                if endp >= startp 
                    #d1 = time()
                    Random.setstate!(r, UInt64(j), UInt64(k), UInt64(1), UInt64(1))
                    rand!(r, space)
                    #@views space .= S[j : j + length(space) - 1, k]
                    #c1 += time() - d1
                    #num_sampled += length(space)
                    @inbounds @turbo for spidx = startp : endp 
                        actual_idx = offset + spidx - 1
                         for k = 1 : dbound
                            #A_hat[j + k - 1, colval[actual_idx]] += space[k] * nzval[actual_idx]
                            copy_block[k, colval[actual_idx] - i + 1] += space[k] * nzval[actual_idx]
                        end
                    end
                    
                end
                
                
                
                
                

            end

            # copy back
            @inbounds for o1 = 1 : min(n - i + 1, outer_n)
                for o2 = 1 : min(d - j + 1, outer_d)
                    A_hat[j + o2 - 1, i + o1 - 1] = copy_block[o2, o1]
                end       
            end

            
        end
    end

    # restore the state
    #@turbo nzval .*= 2147483647
    elasped_time = time() - t1
    println("sample time: ", c1)
    println("number sampled: ", num_sampled)
    println("blocked algorithm flops: ", nnz(A) * d * 2)
    println("blocked algorithm elapsed time: ", elasped_time)
    println("blocked algorithm Gflops: ", nnz(A) * d * 2 / elasped_time / 1000000000)

end


function advanced_blocked_dense_sparse_pm1!(A, A_hat, d, outer_n=2000, outer_d=1200)

    # rng

    # size parameters
    m, n = size(A)
    
    
    

    #blocking sizes
    # outer_n = 3000 # 200
    # outer_d = 1000 # 4000

    #temporary variables and storage
    blocked_csc = produce_blocked_csc(A, outer_n)
    colval = blocked_csc.colval
    nzval = blocked_csc.nzval
    c1 = 0
    num_sampled = 0
    
    t1 = time()
    @turbo A_hat .= 0
    copy_list = [zeros(outer_d, outer_n) for i = 1 : Threads.nthreads()]
    @inbounds Threads.@threads for i = 1 : outer_n : n
        
        copy_block = copy_list[Threads.threadid()]
        #copy_block = zeros(outer_d, outer_n)
        xo_seed = ones(UInt64, 4)
        r = Xoshiro(xo_seed)
        block_idx = Int64(Int64((i - 1) / outer_n)) + 1
        offset = blocked_csc.block_idx_range[Int64((i - 1) / outer_n) + 1]
        

        @inbounds for j = 1 : outer_d : d
            #space = zeros(Int32, outer_d);
            space = zeros(Int8, min(d - j + 1, outer_d));

            # perform copy optimization
            @inbounds for o1 = 1 : min(n - i + 1, outer_n)
                for o2 = 1 : min(d - j + 1, outer_d)
                    copy_block[o2, o1] = A_hat[j + o2 - 1, i + o1 - 1]
                end       
            end

            @inbounds for k = 1 : m

                
                dbound = min(d - j + 1, outer_d)
                startp = blocked_csc.rowptr[(block_idx - 1) * (m + 1) + k]
                endp = blocked_csc.rowptr[(block_idx - 1) * (m + 1) + k + 1] - 1
                
                if endp >= startp 
                    #d1 = time()
                    Random.setstate!(r, UInt64(j), UInt64(k), UInt64(1), UInt64(1))
                    rand!(r, space)
                    @turbo space .= (space .>> 7) .* 2 .+ 1 
                    #c1 += time() - d1
                    #num_sampled += length(space)
                    @inbounds @turbo for actual_idx = startp + offset - 1 : endp + offset - 1 
                        #actual_idx = offset + spidx - 1
                         for k = 1 : dbound
                            #A_hat[j + k - 1, colval[actual_idx]] += space[k] * nzval[actual_idx]
                            copy_block[k, colval[actual_idx] - i + 1] += space[k] * nzval[actual_idx]
                        end
                    end
                    
                end
                
                
                
                
                

            end

            # copy back
            @inbounds for o1 = 1 : min(n - i + 1, outer_n)
                for o2 = 1 : min(d - j + 1, outer_d)
                    A_hat[j + o2 - 1, i + o1 - 1] = copy_block[o2, o1]
                end       
            end

            
        end
    end

    # restore the state
    elasped_time = time() - t1
    println("sample time: ", c1)
    println("number sampled: ", num_sampled)
    println("blocked algorithm flops: ", nnz(A) * d * 2)
    println("blocked algorithm elapsed time: ", elasped_time)
    println("blocked algorithm Gflops: ", nnz(A) * d * 2 / elasped_time / 1000000000)

end



mutable struct matrix_operator{}

    A::SparseMatrixCSC{Float64}
    R::Union{UpperTriangular{Float64}, Matrix{Float64}}
    adj::Bool

end

import LinearAlgebra.mul!
function mul!(y::AbstractVector, M::matrix_operator, x::AbstractVector)
    if M.adj == 0
        #y .= M.A * (M.R \ x)
        mul!(y, M.A, BLAS.trsv('U', 'N', 'N', M.R, x))
    else
        #y .= M.R' \ (M.A' * x)
        mul!(y, M.A', x)
        BLAS.trsv!('U', 'T', 'N', M.R, y)
    end
end


function Base.:(size)(M::matrix_operator, dim::Int64)
    if M.adj == 0
        if dim == 1
            return size(M.A, 1)
        elseif dim == 2
            return size(M.R, 2)
        else
            error("undefined dimension")
        end
    else 
        if dim == 1
            return size(M.R, 2)
        elseif dim == 2
            return size(M.A, 1)
        else
            error("undefined dimension")
        end
    end
end

function Base.:(size)(M::matrix_operator)
    if M.adj == 0
        return (size(M.A, 1), size(M.R, 2))
    else 
        return (size(M.R, 2), size(M.A, 1))
    end
end

function Base.:(eltype)(M::matrix_operator)
    return eltype(M.A)
end

function Base.:(*)(M::matrix_operator, x)
    if M.adj == 0
        #return M.A * (M.R \ x) 
        return M.A * BLAS.trsv('U', 'N', 'N', M.R, x)
    else 
        #return M.R' \ (M.A' * x)
        BLAS.trsv('U', 'T', 'N', M.R, (M.A' * x))
    end
end

function Base.:(adjoint)(M::matrix_operator)
    return matrix_operator(M.A, M.R, !M.adj)
end


mutable struct matrix_operator_svd{}

    A::SparseMatrixCSC{Float64}
    V::AbstractMatrix{Float64}
    S::Vector{Float64}
    adj::Bool

end

import LinearAlgebra.mul!
function mul!(y::AbstractVector, M::matrix_operator_svd, x::AbstractVector)

    if M.adj == 0
        
        y .= M.A * (M.V * (M.S .* x))
    else
        
        y .= M.S .* (M.V' * (M.A' * x))
    end
end


function Base.:(size)(M::matrix_operator_svd, dim::Int64)
    if M.adj == 0
        if dim == 1
            return size(M.A, 1)
        elseif dim == 2
            return size(M.S, 1)
        else
            error("undefined dimension")
        end
    else 
        if dim == 1
            return size(M.S, 1)
        elseif dim == 2
            return size(M.A, 1)
        else
            error("undefined dimension")
        end
    end
end

function Base.:(size)(M::matrix_operator_svd)
    if M.adj == 0
        return (size(M.A, 1), size(M.S, 1))
    else 
        return (size(M.S, 1), size(M.A, 1))
    end
end

function Base.:(eltype)(M::matrix_operator_svd)
    return eltype(M.A)
end

function Base.:(*)(M::matrix_operator_svd, x)

    if M.adj == 0
 
        return M.A * (M.V * (M.S .* x))
    else 

        return M.S .* (M.V' * (M.A' * x))
    end
end

function Base.:(adjoint)(M::matrix_operator_svd)
    return matrix_operator_svd(M.A, M.V, M.S, !M.adj)
end


function transpose_sparse(A_sparse, m, n)
    extract_transpose = produce_blocked_csc(A_sparse, n)
    A_sparse = SparseMatrixCSC(n, m, extract_transpose.rowptr, extract_transpose.colval, extract_transpose.nzval)
    return A_sparse
end

function sketch_and_solve(A_sparse, b, factor, reduce_factor, method::String, initial_guess, sketch_method::String)
    m, n = size(A_sparse)
    d = factor * n
    A_hat = zeros(d, n)
    
    if sketch_method == "simple"
        multithread_blocked_dense_sparse!(A_sparse, A_hat, d, 50, 15000)
    elseif sketch_method == "simplepm"
        multithread_blocked_dense_sparse_pm1!(A_sparse, A_hat, d, 50, 15000)
    elseif sketch_method == "advanced"
        advanced_blocked_dense_sparse!(A_sparse, A_hat, d, 50, 15000)
    else 
        advanced_blocked_dense_sparse_pm1!(A_sparse, A_hat, d, 50, 15000)
    end
 
    if method == "qr"

        x = deepcopy(initial_guess)
  
        t_qr = @timed fac = qr!(A_hat);
        R = fac.R
        println("qr time: ", t_qr.time)
        precond = matrix_operator(A_sparse, R, 0)
 
        t_lsqr = @timed out = lsqr!(x, precond, b; atol=1e-14, btol=1e-14, log=true, maxiter=1000)
        println("lsqr time: ", t_lsqr.time)
        println(out[2])
        println("atol: ", out[2][:atol], " btol: ", out[2][:btol], " ctol: ", out[2][:ctol])
        #println(out[2][:resnorm])
        return R \ x

    elseif method == "svd"
        t_svd = @timed F = LAPACK.gesdd!('O', A_hat);
        S = F[2]
        V = F[3]'
        #t_svd = @timed _, S, V = svd(A_hat)
        
        println("svd time: ", t_svd.time)
        #bound = max(1, Int64(ceil(n / reduce_factor)))
        max_sigma = S[1]
        bound = searchsortedfirst(S, 1e-12 * max_sigma, rev=true)
        bound -= 1
        S = S[1 : bound]
        V = V[:, 1 : bound]
        x = zeros(bound)
        precond = matrix_operator_svd(A_sparse, V, inv.(S), 0)
        t_lsqr = @timed out = lsqr!(x, precond, b; atol=1e-14, btol=1e-14, log=true, maxiter=1000)
        println("lsqr time: ", t_lsqr.time)
        println(out[2])
        println("atol: ", out[2][:atol], " btol: ", out[2][:btol], " ctol: ", out[2][:ctol])
        #println(out[2][:resnorm])
        return @views V * (inv.(S) .* x)

    else 
        error("method not recognized")
    end
    
    
    
end



function test_pipeline(matrix_name, method, sketch_method)
    file = matopen(matrix_name)
    info=read(file);
    A_sparse = info["Problem"]["A"];
    println(matrix_name, "------------------------------------------------------------------------------------------------")
    m, n = size(A_sparse)
    println("original size: ", size(A_sparse))
    if m < n
        A_sparse = transpose_sparse(A_sparse, m, n)
        m, n = size(A_sparse)
    end

    println("new size: ", size(A_sparse))
    println("number of nonzeros: ", nnz(A_sparse))


    
    x = rand(n); b = A_sparse * x + randn(m);  res = zeros(n);
   
    println("solve with sketching in span of A_sparse plus gaussian noise")
    sketch_factor = 2
    t_sap = @timed res = sketch_and_solve(A_sparse, b, sketch_factor, 1, method, zeros(n), sketch_method);
    temp = A_sparse * res - b
    println("total time for sap with overhead: ", t_sap.time - t_sap.gctime)
    println("memory used: ", sketch_factor * n * n * 8 / 1000^2)
    println("total sap memory allocated for julia portion: ", t_sap.bytes / 1000^2)
    println("||A’(Ax-b)||/ (||A||_F * ||Ax-b||): ", norm(A_sparse' * temp) / norm(temp) / norm(A_sparse, 2))
    println()
    println()

end
