import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    if isinstance(t, tuple):
        return t
    else:
        return (t, t)


class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__() #确保父类被正确初始化
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self , dim, heads = 8, dim_head = 64, dropout =0. ):
        super().__init__( )
        inner_dim = dim_head * heads
        """
        如果有多个头（heads > 1）或每个头的维度与输入维度不同（dim_head != dim），
        则可能需要一个额外的投影（projection）步骤来整合或调整不同头的输出。
        """
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attention = nn.Softmax(dim = -1)
        """
        用于生成 query, key, 和 value 的表示。dim 是输入维度，
        而 inner_dim * 3 是输出维度，乘以3是因为需要同时为 query, key, 和 value 生成表示。
        """
        self .to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        def forward(self,x):
            b, n, _, h = *x.shape, self.heads # batch size, number of tokens, dimensions, heads
            qkv = self.to_qkv(x).chunk(3,dim = -1)
            """
            b n (h d)：b 代表批次大小（batch_size），n 代表序列长度（sequence_length），(h d) 表示将头数和每头的维度合并为一个维度进行处理。
            -> b h n d：表示重新排列后的形状，b 仍是批次大小，h 是头数（heads），n 是序列长度，d 是每个头的维度（dim_head）
            将隐藏维度 (h * d) 分解成多个“头” (h)，每个头处理一部分维度 (d)。这样，每个头都可以独立学习序列中不同的特征。
            """
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
            """
            表达式: 'b h i d, b h j d -> b h i j'
            输入:
            q 和 k 都是 query 和 key 矩阵，形状均为 (batch_size, heads, seq_length, dim_per_head)。
            b 代表批次大小，h 是头的数量，i 和 j 分别是序列的位置索引，d 是每个头的维度。
            功能描述：
            操作: 该表达式计算了 q 和 k 的每个头的点积。具体来说，对于每个批次、每个头、每个序列位置 i，计算 q 在该位置与 k 在所有位置 j 的点积。这个点积是通过对维度 d（每个头的维度）进行求和实现的。
            输出: 输出的结果是一个四维张量，形状为 (batch_size, heads, seq_length, seq_length)。这个张量的每个元素代表了序列中位置 i 对于位置 j 的注意力得分。
            """

            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            attn = self.attention(dots)

            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 65, dim)
        x += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)
        print(x.shape)

        return self.mlp_head(x)                                                 #  (b, num_classes)

model_vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

img = torch.randn(16, 3, 256, 256)

preds = model_vit(img)

print(preds.shape)  # (16, 1000)

