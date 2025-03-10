import torch
from torch import Tensor

parallelism=[3,3,2]

dim_0_split = (
    torch.tensor([[15, 24,  2],
                  [ 4,  9, 20],
                  [15, 28, 14]], dtype=torch.int32),
    torch.tensor([[20, 11, 12],
                  [ 0,  5,  9],
                  [16, 22, 25]], dtype=torch.int32),
    torch.tensor([[ 5,  9, 21],
                  [29, 12, 27],
                  [13, 17, 30]], dtype=torch.int32),
    torch.tensor([[ 1,  5, 11],
                  [ 9, 29,  5],
                  [ 8,  4,  1]], dtype=torch.int32),
)


dim_1_split = [x.split(parallelism[1], dim=1) for x in dim_0_split]
print (dim_1_split)

blocks =[]
for i in range(len(dim_1_split)):
    for j in range(len(dim_1_split[i])):
        print ("dim_1_split[i][j]",dim_1_split[i][j])
        blocks.append(dim_1_split[i][j].flatten().tolist())

# dim_2_split = [x.split(parallelism[2], dim=1) for x in dim_1_split]
# print ()

print (len([1, -9, -4, 4, -2, -6, -8, -16, -9, -14, -18, -11, -4, -5, -10, -5, -11, -2, 0, 3, -1, -1, -7, 2, -1, -1, 4, 4, 6, 3, -5, -7, 3, -3, -9, -11, -2, -16, -14, -3, 3, -16, -17, -15, -10, -15, -9, -5, -13, -1, -6, -2, -15, 1, 1, -5, 0, 5, 1, 4, 5, -2, 1, -1]))

