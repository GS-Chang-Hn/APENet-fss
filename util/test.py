import torch

# 假设输入张量为 input_tensor
input_tensor = torch.randn(1, 512)

# 使用 unsqueeze 方法添加额外的维度
unsqueezed_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

# 使用 expand 方法扩展张量的大小
expanded_tensor = unsqueezed_tensor.expand(1, 56, 56, 512)
print(expanded_tensor.shape)  # 输出 torch.Size([1, 56, 56, 512])