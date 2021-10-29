
module ClassicalAndQuantumFIMs

using LinearAlgebra, ThreadsX, QuantumOpticsBase, OQSmodels
using Base.Threads: @threads, @spawn




#Some common Morozova-Cenkov functions to use in generalizedQFIM below
export SLD_mc_func, RLD_mc_func
SLD_mc_func(t) = 0.5*(t+1) #SLD-QFI
RLD_mc_func(t) = 2*t/(t+1) #RLD-QFI



export ClassicalFIM, QuantumFIM, MultiFIM 


#Version where derivatives ∂P/∂η have already been calculated
function ClassicalFIM(pops::Vector{T} where T <: Real, deriv_list::Vector{T} where T)

    FIM_dim = length(deriv_list)
    FIM = zeros(FIM_dim, FIM_dim)
    for i in 1:FIM_dim, j in i:FIM_dim
        FIM[i, j] = sum(deriv_list[i] .* deriv_list[j] ./ pops)
    end
    FIM = Symmetric(FIM) |> Array #Copy upper half to lower half but don't keep symmetric typing since eigen(::Symmetric{::ComplexF64}) method doesn't exist

    return FIM
end

#Version which calculates the derivativs ∂P/∂η
function ClassicalFIM(m::OQSmodel, FoM, diff_params::Vector{T} where T <: DiffParam)

    Ps = FoM(m)
    # if !(eltype(Ps) <: Real) #If so, we have likely forgotten to wrap FoM in call to populations(FoM(m)) so do that here
    if !(typeof(Ps) <: Vector) #If so, we have likely forgotten to wrap FoM in call to populations(FoM(m)) so do that here
        Ps = populations(Ps)
        FoM = populations ∘ FoM
    end
    deriv_list = ModelGradient(m, FoM, diff_params).grad_vec

    return ClassicalFIM(Ps, deriv_list)
end

#Version where we already have state's eigen-system and its various derivatives (for faster calc with multiple MC functions)
function QuantumFIM(vals, vecs, d_eigvals, d_eigvecs, MC_func)

    #Construct FIM
    state_dim = length(vals)
    FIM_dim = length(d_eigvals)
    QFIM = zeros(ComplexF64, FIM_dim, FIM_dim)

    # function single_el(I, a, b)
    #     i, j = Tuple(I)
    #     return i == j ? 0 : 1/(vals[i]*MC_func(vals[j]/vals[i])) * (vals[j] - vals[i])^2 * dot(vecs[j], d_eigvecs[a][i]) * dot(d_eigvecs[b][i], vecs[j])
    # end

    #Can this loop be optimized further?
    @threads for μ in 1:FIM_dim #Multi-threaded outer loop
        for ν in μ:FIM_dim

            QFIM[μ, ν] += sum(dμ*dν/P for (dμ, dν, P) in zip(d_eigvals[μ], d_eigvals[ν], vals) if P != 0) #First term in QFIM def (changes in eigvals)

            # QFIM[μ, ν] += mapreduce(I -> single_el(I, μ, ν), +, CartesianIndices(state))

            for i in 1:state_dim, j in 1:state_dim #Second term in QFIM def (changes in eigvecs)
                if i != j
                    QFIM[μ, ν] += 1/(vals[i]*MC_func(vals[j]/vals[i])) * (vals[j] - vals[i])^2 * dot(vecs[j], d_eigvecs[μ][i]) * dot(d_eigvecs[ν][i], vecs[j])
                end
            end

        end
    end

   return Symmetric(QFIM) |> Array #Copy upper half to lower half but don't keep symmetric typing since eigen(::Symmetric{::ComplexF64}) method doesn't exist
end


#Version where we need to calculate derivatives 
function QuantumFIM(m::OQSmodel, FoM, diff_params::Vector{T} where T <: DiffParam; MC_func=SLD_mc_func)

    state = FoM(m)
    state_dim = size(state, 1)
    
    if typeof(state) <: DataOperator #If so, we have likely forgotten to extract op's data field in FoM so fix that here
        state = state.data
        FoM = data ∘ FoM
    end

    vals, U = eigen(state)
    vecs = collect(eachcol(U))
    deriv_list = ModelGradient(m, FoM, diff_params).grad_vec


    #Use (1st order) perturbation theory equations to get derivatives of eigenvalues and eigenvectors 
    # (see https://www.wikiwand.com/en/Perturbation_theory_(quantum_mechanics)#/First_order_corrections)
    dvecs = [[sum(dot(vecs[j], dρ, vecs[i])/(vals[i]-vals[j])*vecs[j] for j in 1:state_dim if vals[i] != vals[j]) for i in 1:state_dim] for dρ in deriv_list]
    dvals = [[real(dot(v, dρ, v)) for v in eachcol(U)] for dρ in deriv_list]

    return QuantumFIM(vals, vecs, dvals, dvecs, MC_func)

end



# This method seems slightly slower than the more general one, so needs some more work before it's useful
function QuantumFIM_fastSLD(m::OQSmodel, FoM, diff_params::Vector{T} where T <: DiffParam)

    state = FoM(m)
    state_dim = size(state, 1)
    
    if typeof(state) <: DataOperator #If so, we have likely forgotten to extract op's data field in FoM so fix that here
        state = state.data
        FoM = data ∘ FoM
    end

    vals, U = eigen(state)
    vecs = collect(eachcol(U))
    deriv_list = ModelGradient(m, FoM, diff_params).grad_vec

    function single_el(I::CartesianIndex, a, b)
        i, j = Tuple(I)
        return 2 * real( dot(vecs[i],  deriv_list[a],  vecs[j]) * dot(vecs[j],  deriv_list[b], vecs[i]) ) /  ( vals[i] + vals[j] )
    end

    #Combine all elements to form QFIM
    FIM_dim = length(deriv_list)
    QuantumFIM = zeros(FIM_dim, FIM_dim)
    Threads.@threads for a in 1:FIM_dim
        for b in a:FIM_dim #Loop over QFIM elements 
            # for i in 1:state_dim, j in 1:state_dim #Loop over state eigenvalues / vectors
            #     real(vals[i] + vals[j]) < 1e-16 && continue #Skip any divergent terms in sum
            #     QuantumFIM[a, b] += 2 * real( dot(vecs[i],  deriv_list[a],  vecs[j]) * dot(vecs[j],  deriv_list[b], vecs[i]) ) /  ( vals[i] + vals[j] )
            # end
            QuantumFIM[a, b] = mapreduce(I -> single_el(I, a, b), +, CartesianIndices(state))
        end
    end
    QuantumFIM = Symmetric(QuantumFIM) # Copy upper half to lower half to save doubling up the computation

end





# Calculate multiple FIMs from the same derivative list (FIM_types can be either 'CFIM' or a MC func)
function MultiFIM(m::OQSmodel, FoM, diff_params::Vector{T} where T <: DiffParam, FIM_types)

    state = steady_state(m)
    state_dim = size(state, 1)
    FIM_dim = length(diff_params)
    
    if typeof(state) <: DataOperator #If so, we have likely forgotten to extract op's data field in FoM so fix that here
        state = state.data
        FoM = data ∘ FoM
    end

    vals, U = eigen(state)
    vecs = collect(eachcol(U))
    deriv_list = ModelGradient(m, FoM, diff_params).grad_vec

    #Use (1st order) perturbation theory equations to get derivatives of eigenvalues and eigenvectors (needed for all QFIM types)
    # (see https://www.wikiwand.com/en/Perturbation_theory_(quantum_mechanics)#/First_order_corrections)
    dvecs = [[sum(dot(vecs[j], dρ, vecs[i])/(vals[i]-vals[j])*vecs[j] for j in 1:state_dim if vals[i] != vals[j]) for i in 1:state_dim] for dρ in deriv_list]
    dvals = [[real(dot(v, dρ, v)) for v in eachcol(U)] for dρ in deriv_list]


    FIM_list = fill(zeros(FIM_dim, FIM_dim), length(FIM_types))
    for i in eachindex(FIM_types)
        if FIM_types[i] == "CFIM" || FIM_types[i] == "ClassicalFIM"
            FIM_list[i] = ClassicalFIM(populations(state), populations.(deriv_list))
        else
            MC_func = FIM_types[i]
            FIM_list[i] = QuantumFIM(vals, vecs, dvals, dvecs, MC_func)
        end
    end

    return FIM_list

end





function fast_QFIM_trace(m::OQSmodel, FoM, diff_params::Vector{T} where T <: DiffParam)

    state = FoM(m)
    state_dim = size(state, 1)
    
    if typeof(state) <: DataOperator #If so, we have likely forgotten to extract op's data field in FoM so fix that here
        state = state.data
        FoM = data ∘ FoM
    end

    vals, U = eigen(state)
    vecs = collect(eachcol(U))
    deriv_list = ModelGradient(m, FoM, diff_params).grad_vec

    QFI(vals, vecs, dρ) = 2*sum(abs2(dot(vecs[i], dρ, vecs[j]))/(vals[i]+vals[j]) for i in 1:state_dim for j in 1:state_dim)

    return ThreadsX.mapreduce(d -> QFI(vals, vecs, d), +, deriv_list)
end



end #Module 

