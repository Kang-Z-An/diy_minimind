from transformers import PretrainedConfig


# 主要是和huggingface有关的一些
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


# 编写RMSNorm
# 这是一个层，需要继承自nn.Module(且强制要求实现前向传播方法)
# self的知识: 只要在类里面其他类里面用变量，就要加self（存在self上的，是成员变量；不加self的，是局部变量）
class RMSNorm(nn.Module):
    #初始化
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    #RMSNorm 计算逻辑
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim = True)+self.eps) #在最后一个维度求均值。[batch_size,seq_len,hidden_size]
    
    #前向传播
    def forward(self,x):
        return (self.weight*self._norm(x.float())).type_as(x) #float() 和 type_as(x) 都是对x数据类型的操作，可能x本身是float16，那么就是中间先转换为float32，再转换为float16
    

    
#YaRN方法（rope_scaling:RoPE缩放配置）
def precompute_freqs_cis(dim:int,end:int=int(32*1024),rope_base:float=1e6,rope_scaling:Optional[dict]=None):
    # RoPE公式: freq = 1 / (base ^ (2i/d))  其中 i = 0, 1, ..., dim/2-1
    # torch.arange(0,dim,2)     → 偶数维索引: [0,2,4,...,dim-2]
    # [:(dim//2)]              → 取前 dim//2 个
    # .float()/dim             → 算出指数: 2i/d
    # rope_base ** (...)       → base^(2i/d)
    # 1.0 / (...)              → 取倒数得到频率
    # 结果: freqs 长度 dim/2, 每个元素是第 i 对维度的频率
    # attn_factor: 对应YaRN论文中的t，在softmax里面一个额外的缩放因子，理论上应该根据上下文长度自动变化
    freqs,attn_factor=(
        1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)),
        1.0)
    
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow,attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), 
            rope_scaling.get("factor", 16),# 上下文扩展倍数
            rope_scaling.get("beta_fast", 32.0), 
            rope_scaling.get("beta_slow", 1.0), 
            rope_scaling.get("attention_factor", 1.0) #注意力缩放因子，等效于调整softmax温度
        )

        # 推断长度大于训练长度的时候，使用YaRN
        if end > orig_max:
            #先求划分高低维的i
            inv_dim = lambda b: (dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))
            # 划分高低维度
            # low低维：不需要缩放的高频部分
            # high高维: 需要缩放的低频部分
            low,high = (
                max(math.floor(inv_dim(beta_fast)),0), #向下取整 
                min(math.ceil(inv_dim(beta_slow)),dim//2-1) #向上取整
            )

            # 5. 计算混合因子 γ (Ramp)--对应平滑过度这一段
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp((torch.arange(dim//2,device=freqs.device).float()-low)/max(high-low,0.001),0,1)
            freqs = freqs*(1-ramp+ramp/factor) #[dim/2]
        
    t = torch.arange(0,end,device=freqs.device) #[end]
    freqs = torch.outer(t,freqs).float() #[end,dim/2 ]
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)*attn_factor
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)*attn_factor
    return freqs_cos,freqs_sin


def apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),dim=-1)
    q_embed = ((q*cos.unsqueeze(unsqueeze_dim))+(rotate_half(q)*sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k*cos.unsqueeze(unsqueeze_dim))+(rotate_half(k)*sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed,k_embed



            
            












    






