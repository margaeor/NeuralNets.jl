# select some random subset of the data of size b
function batch(b::Int,x,t)
    subset = rand(1:size(x,2),b)
    return x[:,subset],t[:,subset]
end

# Train a MLP using gradient decent with momentum.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function gdmtrain(mlp::MLP,
                  x,
                  t,
                  x_valid=nothing,
                  t_valid=nothing;
                  batch_size=size(x,2),
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate=.03,
                  momentum_rate=.6,
                  eval::Int=10,
                  show_trace::Bool=false)

    valid = !(isnothing(x_valid) || isnothing(t_valid)) # validation set present?

    η, c, m, b = learning_rate, tol, momentum_rate, batch_size
    i = e_old = Δw_old = 0
    converged::Bool = false

    e_train = loss(prop(mlp,x),t)
    e_valid = valid ? loss(prop(mlp,x_valid),t_valid) : 0

    n = size(x,2)
    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        ∇ = backprop(mlp.net,x_batch,t_batch)
        Δw_new = η*∇ .+ m.*Δw_old         # calculate Δ weights
        mlp.net = mlp.net .- Δw_new      # update weights
        Δw_old = Δw_new

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_train
            e_train = loss(prop(mlp,x),t)

            if !valid
                converged = abs(e_train - e_old) < c # check if converged
            else
              e_old_valid = e_valid
              e_valid = loss(prop(mlp,x_valid),t_valid)
              converged = abs(e_train - e_old_valid) < c # check if converged
            end

            diagnostic_trace!(mlp.report, e_train, e_valid, valid, show_trace)
        end
    end
    converged ? println("\nTraining converged.") : println("\nTraining did not converge.")

    mlp
end

# Train a MLP using Adagrad
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# c:        convergence criterion
# ε:        small constant for numerical stability
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function adatrain(mlp::MLP,
                  x,
                  t,
                  x_valid=Void,
                  t_valid=Void;
                  batch_size=size(x,2),
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate=.3,
                  lambda=1e-6,
                  eval::Int=10,
                  store_trace::Bool=false,
                  show_trace::Bool=false)

    valid = !(x_valid == Void || t_valid == Void) # validation set present?

    η, c, λ, b = learning_rate, tol, lambda, batch_size
    i = e_old = Δnet = sumgrad = 0
    e_train = loss(prop(mlp.net,x),t)
    e_valid = valid ? loss(prop(mlp,x_valid),t_valid) : 0
    n = size(x,2)
    converged::Bool = false
    e_list = []
    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        ∇ = backprop(mlp.net,x_batch,t_batch)
        sumgrad += ∇ .^ 2       # store sum of squared past gradients
        Δw = η * ∇ ./ (λ .+ (sumgrad .^ 0.5))   # calculate Δ weights
        mlp.net = mlp.net .- Δw                 # update weights

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_train
            e_train = loss(prop(mlp,x),t)

            if !valid
              converged = abs(e_train - e_old) < c # check if converged
            else
              e_old_valid = e_valid
              e_valid = loss(prop(mlp,x_valid),t_valid)
              converged = abs(e_train - e_old) < c # check if converged
            end

            diagnostic_trace!(mlp.report, e_train, e_valid, valid, show_trace)
        end
    end
    convgstr = converged ? "converged" : "didn't converge"
    println("Training $convgstr in less than $i iterations; average error: $(round(((valid ? e_valid : e_train)/n),4)).")
    println("* learning rate η = $η")
    println("* convergence criterion c = $c")
    return mlp, e_list
end
