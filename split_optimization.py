import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from mnist_cnn import MNIST
from model_splitting import split_model
from lat_acc_test_funcs import test_split_latency_accuracy, latencies_with_bw, get_flops, get_read_writes
from skcriteria import mkdm
from skcriteria.agg import similarity
from skcriteria.pipeline import mkpipe
from skcriteria.preprocessing import scalers, invert_objectives


# class SplitPointProblem(ElementwiseProblem):
#     def __init__(self, L, E, T):
#         super().__init__(n_var=1, n_obj=2, xl=1, xu=L, type_var=int)
#         self.L = L
#         self.energy = E
#         self.latency = T

#     def _evaluate(self, x, out, *args, **kwargs):
#         # print(x)
#         s = str(int(x[0]))
#         out["F"] = [self.energy[s], self.latency[s]]  # Your actual cost functions

# kinda sucks
def topsis(X, weights, benefit_criteria):
    # Normalize
    scaler = MinMaxScaler()
    norm_X = scaler.fit_transform(X)

    # Weighted normalized matrix
    V = norm_X * weights

    # Identify ideal and anti-ideal solutions
    ideal = np.min(V, axis=0)
    # Calculate distances
    d_pos = np.linalg.norm(V - ideal, axis=1)

    # Compute relative closeness to ideal solution
    scores = d_pos
    return scores


def store_latencies(model, testloader):
    latencies = {}
    L = len(list(model.model.children()))

    for l in range(L+1):
        print(f"Testing split at layer {l}")
        head, tail = split_model(model, l)
        _, lat = test_split_latency_accuracy(head, tail.to('mps'), testloader)
        latencies[l] = lat

    with open('latencies.json', 'w') as f:
        json.dump(latencies, f)

def store_latencies_bw(model, testloader, bw_bps):
    latencies = {}
    L = len(list(model.children()))

    for l in range(L+1):
        print(f"Testing split at layer {l}")
        head, tail = split_model(model, l)
        lat = latencies_with_bw(head, tail, bw_bps, testloader)
        latencies[l] = lat

    with open('latencies.json', 'w') as f:
        json.dump(latencies, f)

def store_flops(model, input_size):
    L = len(list(model.children()))

    flops = get_flops(model, input_size=input_size)

    cumulative_flops = {0: 0}
    total = 0
    for l in range(L):
        total += flops[l]
        cumulative_flops[l+1] = total

    print(cumulative_flops)
    with open('flops.json', 'w') as f:
        json.dump(cumulative_flops, f)

def store_read_writes(model, input_size):
    L = len(list(model.children()))

    reads, writes = get_read_writes(model, input_size=input_size)

    cumulative_reads = {0: 0}
    total = 0
    for l in range(L):
        total += reads[l]
        cumulative_reads[l+1] = total

    with open('reads.json', 'w') as f:
        json.dump(cumulative_reads, f)


    cumulative_writes = {0: 0}
    total = 0
    for l in range(L):
        total += writes[l]
        cumulative_writes[l+1] = total

    with open('writes.json', 'w') as f:
        json.dump(cumulative_writes, f)

def eval_topsis(lat, flops, weights=[0.5, 0.5], max_flops=torch.inf, max_lat=torch.inf):
    F = []
    alts = []
    for l in range(len(lat.values())):
        label = str(l)
        if flops[label] < max_flops and lat[label] < max_lat:
            F.append([lat[label], flops[label]])
            alts.append(l)

    if len(F) == 0:
        print("No valid splits under constraint.")
        return
    elif len(F) == 1:
        print("Only one valid option:", alts[0], F[0])
        return

    dm = mkdm(F, objectives=["min", "min"], weights=weights, alternatives=alts)


    pipe = mkpipe(
        invert_objectives.NegateMinimize(),
        scalers.VectorScaler(target="matrix"),  # this scaler transform the matrix
        scalers.SumScaler(target="weights"),  # and this transform the weights
        similarity.TOPSIS(),
    )
    
    res_topsis = pipe.evaluate(dm)
    best_index = res_topsis.rank_.argmin()

    print(res_topsis)
    print("Best index by TOPSIS:", best_index)
    print("Optimal Values:", F[best_index])


def get_max_flops(max_energy_J, energy_per_flop_J):
    return max_energy_J / energy_per_flop_J


if __name__ == "__main__":
    batch_size = 64

    _, _, testset = MNIST()
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

   

    store_latencies_bw(model, testloader, 10**6)
    store_flops(model.model, input_size=(batch_size, 1, 28, 28))

    with open('latencies.json', 'r') as f:
        latencies = json.load(f)

    with open('flops.json', 'r') as f:
        cumulative_flops = json.load(f)

    eval_topsis(latencies, cumulative_flops)#, max_flops=10**6, max_lat=0.025)



    # store_latencies(model, testloader)
    # store_flops(model, input_size=(batch_size, 1, 28, 28))

    # with open('latencies.json', 'r') as f:
    #     latencies = json.load(f)

    # with open('flops.json', 'r') as f:
    #     cumulative_flops = json.load(f)

    # eval_topsis(latencies, cumulative_flops)



# problem = SplitPointProblem(L, cumulative_flops, latencies)

# algorithm = NSGA2(pop_size=20,
#                 sampling=IntegerRandomSampling(),
#                 crossover=SBX(repair=RoundingRepair(), vtype=int),
#                 mutation=PM(repair=RoundingRepair(), vtype=int),
#                 eliminate_duplicates=True)

# scores = topsis(res.F, weights, benefit_criteria)

# print("mniimizing...")
# res = minimize(problem,
#                algorithm,
#                ('n_gen', 50),
#                seed=1,
#                verbose=True)

# print(res.F)
# print(res.X)

# weights = np.array([0.5, 0.5])  # Equal importance
# benefit_criteria = np.array([0, 0])  # Both are costs

# scores = topsis(res.F, weights, benefit_criteria)

# print(scores)
# # Find best solution
# best_index = np.argmax(scores)
# best_split_point = res.X[best_index]

# print(f"Best split (by TOPSIS): {best_split_point}, Score: {scores[best_index]}")



# 65 nm chip
# 1 mm^2 area

# VLSI Symposium
# ISSCC
# JSSC (journal)
# ESCIRC
# CICC
# TCAS

#split points
#data width 8,16,32
#bottleneck widths

# memory weights + inputs size

# read write count -> energy per access
# - write intermediate values then read
# - 3 Reads, 1 Write per MAC op

# area = area from macs and memory(?)

# latency from cycles (keep cycle types separate (add, mult, read, write, etc.))
# cycles of edge + cycles of cloud + fixed cycles for tx, rx + bw * transmission size



