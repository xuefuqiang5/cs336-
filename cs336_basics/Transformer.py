import torch 
from einops import einsum, rearrange
import einx
image = torch.randn(32, 64, 128, 3)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

print("image's shape is ", image.shape)
print("dim_by's shape is ", dim_by.shape)

dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
image_rearr = rearrange(image, "batch high width channel -> batch 1 high width channel") 
res = dim_value * image_rearr
print(res.shape)


res = einsum(image, dim_by, "batch high width channel, dim_value -> batch dim_value high width channel") 
print(res.shape)

channels_last = torch.randn(64, 32, 32, 3)
# (batch, height, width, channel)
B = torch.randn(32*32, 32*32)
width = 32
height = 32

res = einx.dot(
    "batch row_in col_in channels, (row_out col_out) (row_in col_in) ->\
        batch row_out col_out channels",
        channels_last, B,
        row_out = width, col_out = height,
    )

print(res.shape)