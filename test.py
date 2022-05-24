import numpy as np
data = np.load("./musicnet.npz", allow_pickle=True, encoding='bytes')
print(data["1788"].shape)
print(data["1788"][0].shape)


