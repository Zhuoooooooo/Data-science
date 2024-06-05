# https://github.com/pytorch/pytorch/commit/fdfef759a676ee7a853872e347537bc1e4b51390
import torch
import numpy as np
print('pytorch_version:',torch.__version__)

#Scalar
scalar = torch.tensor(7)
scalar
# check the dimension of scalar
scalar.ndim
# Get the Python number within a tensor (only works with one-element tensors)
scalar.item()

# Vector
vector = torch.tensor([5, 7, 9])
vector
vector.ndim
# Check shape of vector
vector.shape

# Matrix
MATRIX = torch.tensor([[7, 8], 
                       [9, 10],
                       [1, 2]])
MATRIX
MATRIX.ndim
MATRIX.shape

# Tensor
tensor = torch.tensor([[[1, 2, 3, 4],
                        [4, 5, 6, 7],
                        [7, 8, 9, 10],
                        [10, 11, 12, 13]]])
tensor
tensor.ndim
tensor.shape

# Create random Tensor
random_tensor = torch.rand(size = (2, 3, 2))
random_tensor
random_tensor.ndim

# Create zero tensor
zeros = torch.zeros(size = (2, 3, 4))
zeros.ndim
# Create ones tensor
ones = torch.ones(size = (5, 3))
ones.ndim

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
zero_to_ten.ndim

# Can also create a tensor of zeros similar to another tensor
tensor_to_zero = torch.zeros_like(input = zero_to_ten)
tensor_to_zero.ndim

#############################################################

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0])
float_32_tensor.dtype
float_32_tensor.device

float_16_tensor = torch.tensor([3.0, 4.0, 5.0],
                               dtype = torch.float16,
                               device = None,
                               requires_grad = False)
float_16_tensor.device

# check tensor info
ran_tensor = torch.rand(3, 5)
print(ran_tensor)
print(f'shape of tensor: {ran_tensor.shape}')
print(f'dtype of tensor: {ran_tensor.dtype}')
print(f'device of tensor: {ran_tensor.device}')

# Tensor Basic operations
tensor = torch.tensor([1, 2, 3])
tensor + 10
tensor * 2
# Can also use function
torch.multiply(tensor, 10)
# Element-wise multiplication (each element multiplies its equivalent, index 0*0, 1*1, 2*2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)

# Matrix multiplication
torch.matmul(tensor,tensor)
tensor @ tensor

tensor_a = torch.rand(3, 2)
tensor_b = torch.rand(3, 2)
# (3,2) @ (3,2) --> error
torch.matmul(tensor_a, tensor_b)
# need to transpose
tensor_a
tensor_a.T
torch.matmul(tensor_a.T, tensor_b)
tensor_a.T @ tensor_b

# linear model test
torch.manual_seed(42)
linear = torch.nn.Linear(in_features = 2, out_features = 7)
x = tensor_a
output = linear(x)
print(f'input shape: {x.shape}\n')
print(f'output:\n {output}\noutput shape: {output.shape}')

# Finding the min, max, mean, sum
x = torch.arange(0, 100, 10)
print(f'Minimum: {x.min()}')
print(f'Maximum: {x.max()}')
print(f'Mean: {x.type(torch.float32).mean()}')
print(f'Sum: {x.sum()}')

# Find max/min position
x = torch.arange(10,100,10)
print(f'Index of Max: {x.argmax()}')
print(f'Index of Min: {x.argmin()}')

# Change tensor datatype
x = torch.arange(10., 100., 10.)
x.dtype
x_16 = x.type(torch.float16)
x_16.dtype
x_int16 = x.type(torch.int16)
x_int16

x = torch.arange(1., 8.)
x
# Reshape
x_reshape = x.reshape(1, 7)
x_reshape.ndim
x_reshape
# View
x_view = x.view(1, 7)
x_view
# Change view will chage original tensor, too
x_view[:,0] = 9
x_view, x
# stack
x_stack = torch.stack([x,x], dim = 1)
x_stack

# Remove extra dimension
print(f'previous: {x_reshape}')
print(f'previous shape: {x_reshape.shape}')
x_squeezed = x_reshape.squeeze()
x_squeezed
# Add extra dimension
x_unsqueezed = x_reshape.unsqueeze(dim = 2)
x_unsqueezed, x_reshape
x_unsqueezed.shape


# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))
x_original.ndim
# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")
x_permuted

#############################################################
x = torch.arange(1, 10).reshape(1, 3, 3)
x
x[0][1][2]
x[:,:,2]

# NumPy array to tensor
array = np.arange(1.0, 8.0)
tensor_array = torch.from_numpy(array).type(torch.float32)
tensor_array
# Tensor to array
t = torch.ones(7)
np_t = t.numpy()
np_t

#############################################################
torch.manual_seed(seed = 42)
x = torch.rand(3, 4)
torch.manual_seed(seed = 42)
y = torch.rand(3, 4)
y
x
x == y
######################
# GPU
print(torch.__version__)
torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
# Count number of devices
torch.cuda.device_count()

tensor_cpu = torch.tensor([1, 2, 3])
print(tensor_cpu, tensor_cpu.device)
# Move tensor to GPU (if available)
tensor_gpu = tensor_cpu.to(device)
print(tensor_gpu, tensor_gpu.device)
# Move tensor to CPU (if available)
## !tensor_gpu.numpy()
tensor_back_to_cpu = tensor_gpu.cpu().numpy()


###############################################################
#Exercises
#2. Create a random tensor with shape (7, 7).
tensor1 = torch.rand(7, 7)
tensor1
#3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)
tensor2 = torch.rand(1, 7)
tensor2_T = tensor2.T
tensor1 @ tensor2_T
#4. Set the random seed to 0 and do 2 & 3 over again.
torch.manual_seed(seed = 0)
tensor3 = torch.rand(2, 3)
tensor3
torch.manual_seed(seed = 0)
tensor3_1 = torch.rand(2, 3)
tensor3_1
#5. Set random seed on the GPU
torch.cuda.manual_seed(1234)

#6. Create tensor_A and tensor_B on gpu
torch.manual_seed(seed = 1234)
tensor5_1 = torch.rand(2, 3)
torch.manual_seed(seed = 1234)
tensor5_2 = torch.rand(2, 3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tengpu5_1 = tensor5_1.to(device)
tengpu5_2 = tensor5_2.to(device)
#7. Perform matmul on tensor_A and tensor_B
ten6 = tengpu5_1 @ tengpu5_2.T
#8. Find the maximum and minimum values of the output of 7.
ten6.max()
ten6.min()
#9. Find the maximum and minimum index values of the output of 7.
ten6.argmax()
ten6.argmin()
'''10. Make a random tensor with shape (1, 1, 1, 10) and then 
       create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10).
       Set the seed to 7 when you create it and print out the first tensor
       and it's shape as well as the second tensor and it's shape.'''
torch.manual_seed(7)
tensor7 = torch.rand(1, 1, 1, 10)
tensor7_sq = tensor7.squeeze()
tensor7_sq.shape
tensor7.shape