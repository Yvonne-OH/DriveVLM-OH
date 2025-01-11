import torch
from torch.utils import benchmark

typ = torch.float16  # 数据精度
n = 2048 * 16
device = "cuda:1"  # 指定使用的 GPU 设备
a = torch.randn(n, n, device=device, dtype=typ)
b = torch.randn(n, n, device=device, dtype=typ)

t = benchmark.Timer(
    stmt='a @ b',
    globals={'a': a, 'b': b}
)

x = t.timeit(50)
print(f"Performance on {device}: {2 * n**3 / x.median / 1e12:.3f} TFLOPS")

