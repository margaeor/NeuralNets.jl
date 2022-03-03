using NeuralNets
N  = 100
M = 10

x = rand(M, N)
t = rand(1, N)
# xor training data
# x = [
#     0.0 1.0 0.0 1.0
#     0.0 0.0 1.0 1.0
#     ]

# t = [
#     0.0 1.0 1.0 0.0
#     ]

# network topology
layer_sizes = [M, 100, 3, 1]
act = [NeuralNets.relu,  NeuralNets.relu,  NeuralNets.logis]
actd = [NeuralNets.relud, NeuralNets.relud, NeuralNets.logisd]

print(typeof(NeuralNets.relu))

# initialize net
mlp = NeuralNets.MLP(randn, layer_sizes, act, actd)
NeuralNets.gdmtrain(mlp, x, t, x, t; show_trace=true)