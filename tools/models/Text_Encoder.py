import torch
import torch.nn as nn

from .Encoder_utils import LayerNorm, Transformer


"""Text Encoder"""

class CLIPTextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408, # 即 token ID 的数量
                #  vocab_size=49408+1,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        # 定义了一个transformer层
        self.transformer = Transformer(
            width=transformer_width, # Transformer 的隐藏层维度（hidden size），决定了每个token的embedding 向量的维度。
            layers=transformer_layers, #
            heads=transformer_heads,
            attn_mask=self.build_attention_mask() # 注意力掩码（attention mask），用于控制哪些 token 之间可以相互关注。
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) #  用于将离散的 token ID 映射为连续的向量表示(embedding), transformer_width:每个 token 的 embedding 向量的维度。
        # nn.Parameter()  是 PyTorch 中用于定义可训练参数的一种方式。 功能：用于定义模型的权重或偏置等可训练参数。
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width) # 通过归一化输入，减少梯度爆炸或梯度消失问题 and 归一化可以加速模型的收敛速度。
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.text_projection = nn.Linear(transformer_width, embed_dim)
    
    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            # checkpoint = torch.load(pretrained)['model']

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                # if k.startswith('module.encode_text.transformer.'):
                #     new_k = k.replace('module.encode_text.', '')
                    # state_dict[new_k] = checkpoint[k].float()
                    state_dict[k] = checkpoint[k].float()
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                # if k == 'module.encode_text.positional_embedding' or k.startswith('module.encode_text.text_projection') or k.startswith('module.encode_text.token_embedding') or k.startswith('module.encode_text.ln_final'):
                #     new_k = k.replace('module.encode_text.', '')
                    # if new_k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    # state_dict[new_k] = checkpoint[k]
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            if u != [] or w != [] :
                print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # 惰性地创建因果注意掩码，在视觉标记之间充分注意
        # pytorch uses additive attention mask; fill with -inf
        # pytorch使用附加注意掩码；用-inf填充
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    # 深度提示嵌入(deep prompt embeddings)的前向传播过程，主要用于在Transformer模型的多层中插入提示嵌入和文本嵌入
    def forward(self, text):
        # 将 token ID 映射为 embedding向量：nn.Embedding(vocab_size, transformer_width)
        x = self.token_embedding(text) # torch.Size([11, 77, 512])
        # 将位置编码加入到embedding的中的positional_embedding
        x = x + self.positional_embedding # positial_embedding：torch.Size([77, 512])
        x = x.permute(1, 0, 2)  # torch.Size (77,11, 512)
        # x输入到transformer得到输出结果
        x = self.transformer(x) # torch.Size([77, 11, 512])
        x = x.permute(1, 0, 2)
        x = self.ln_final(x) # torch.Size([11, 77, 512])
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # torch.Size([11, 512])
        # 从输入张量x中提取特定位置的向量，并将其与 self.text_projection 矩阵相乘。
        # x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x