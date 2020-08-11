import torch
torch.classes.load_library("build/libdcgan.so")
print(torch.classes.loaded_libraries)
#print(dcgan.forward)
