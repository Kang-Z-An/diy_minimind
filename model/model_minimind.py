from logging import config
from turtle import st

from transformers import PretrainedConfig

import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# 主要是和huggingface有关的一些
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4) #一共有四个专家 
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1) #每个token使用一个专家，就是前1个专家
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

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



# KV Cache 复制
# KV Cache -- Q:每一步只计算当前token的Query，，计算完成后Q就不再需要了，可以立即释放；但是为了计算每个token与历史token的注意力，需要存起来，因此kv少一点，需要复制
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor: #n_repeat
    bs,slen,num_key_value_heads,head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:,:,:,None,:].expand(bs,slen,num_key_value_heads,n_rep,head_dim).reshape(bs,slen,num_key_value_heads*n_rep,head_dim))


# 注意力部分
class Attention(nn.Module):
    def __init__(self,args:MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads #如果没有配置KV头数，那么就变成一个标准多头
        assert args.num_attention_heads % self.num_key_value_heads == 0 # 不满足条件的话就抛出错误
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads//self.n_local_kv_heads
        self.head_dim = args.head_dim
        self.is_causal = True #因果；是否带因果掩码，t位置只能看到前t-1位置的数据
        self.q_proj = nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias = False) #Q = Wq*X 所以没有线性层,下面这几个也是same
        self.k_proj = nn.Linear(args.hidden_size,args.num_key_value_heads*self.head_dim,bias = False) 
        self.v_proj = nn.Linear(args.hidden_size,args.num_key_value_heads*self.head_dim,bias = False)
        self.o_proj = nn.Linear(args.num_attention_heads*self.head_dim,args.hidden_size,bias = False) #在简单拼接注意力头之后，使用Wo(Weight Out)再投影一次
        self.q_norm = RMSNorm(self.head_dim,eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim,eps=args.rms_norm_eps)
        self.attn_dropout = nn.Dropout(args.dropout) #切断部分token之间的强关联 在QK和V相乘之前操作
        self.resid_dropout = nn.Dropout(args.dropout) #切断部分维度的贡献，在concat不同注意力头*Wo之后，Layer_Norm之前进行dropout
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn #PyTorch>=2.0条件下，可以计算注意力更快一些

    def forward(self,
                x:torch.Tensor,
                position_embeddings:Tuple[torch.Tensor,torch.Tensor], #修改为接受cos和sin
                past_key_value :Optional[Tuple[torch.Tensor,torch.Tensor]]= None, #一个是K的缓存，一个是V的缓存
                use_cache = False, #是否要用KV cache 是否启用KV缓存
                attention_mask :Optional[torch.Tensor]= None): #掩码
        bsz, seq_len, _ =x.shape
        xq,xk,xv = self.q_proj(x),self.k_proj(x),self.v_proj(x)
        xq = xq.view(bsz,seq_len,self.n_local_heads,self.head_dim) # [bsz, seq_len, num_heads * head_dim] → [bsz,seq_len,self.n_local_heads,self.head_dim]
        xk = xk.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim) #[bsz,seq_len,num_key_value_heads*head_dim]→ [bsz,seq_len,self.n_key_value_heads,self.head_dim]
        xv = xv.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        xq,xk = self.q_norm(xq),self.k_norm(xk)
        cos,sin = position_embeddings
        xq,xk = apply_rotary_pos_emb(xq,xk,cos,sin)

        #KV cache 实现，这里是拼接历史的k和v
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0],xk],dim=1) #[batch_size,seq_len,...] 
            xv = torch.cat([past_key_value[1],xv],dim=1)
        past_kv = (xk,xv) if use_cache else None

        xq,xk,xv = (
            xq.transpose(1,2),         #转置 [batch_size,seq_len,self.n_local_heads,self.head_dim] -> [batch_size,self.n_local_heads,seq_len,self.head_dim]
            repeat_kv(xk,self.n_rep).transpose(1,2),       #[batch_size,seq_len,num_key_value_heads*n_rep,self.head_dim] -> [batch_size,num_key_value_heads*n_rep,seq_lem,self_head_dim]
            repeat_kv(xv,self.n_rep).transpose(1,2)        
        )

        #是否使用快速注意力--两种计算方式
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal) #只有在训练的时候开启dropout，推理的时候不使用dropout，充分利用上下文信息
        else:
            scores = (xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim) #计算注意力分数
            if self.is_causal: #哪些不能偷看
                scores[:,:,:,-seq_len:] += torch.full((seq_len,seq_len),float("-inf"),device=scores.device).triu(1) 
                #torch.full(size,fill_value,device,requires_grad=False) 创建指定形状的张量
                #triu：triangle+up 上三角矩阵，创建的是一个上三角全是-♾️，下三角全是0的张量，这样子加到scores上面的话，就形成了上三角全是-♾️的这么一个矩阵，自己只能看到自己和自己前面位置的信息
                #scores[:,:,:,-seq_len:] 这里的-seq_len是因为，使用了KV cache，xk = torch.cat([past_key_value[0],xk],dim=1) #[batch_size,seq_len,...] 
                #那么实际上score存储的东西是[:,:,:,past_seq+seq_len]，只对新加入的这部分做掩码
            if attention_mask is not None: #attention_mask=1 表示该位置是一个有效的token，不是占位符等
                # attention_mask:[bsz,seq_len]->[bsz,1,1,seq_len]；要先unsqueeze，扩展维度，才能使用广播机制
                # attention_mask 是一个attention_mask == 1) 全是1的矩阵，下面对应的意思是，无效的token会加上一个-1e9 （+负♾️）
                # attention_mask =1: 非pading；=0:padding
                scores += (1.0-attention_mask.unsqueeze(1).unsqueeze(2))*-1e9
            output = self.attn_dropout(F.softmax(scores.float(),dim=-1).type_as(xq))@xv
        output = output.transpose(1,2).reshape(bsz,seq_len,-1) # -1：[2,3,8,64]->[]2,3,512] 就是把其他的东西放在最后一个维度
        output = self.resid_dropout(self.o_proj(output))
        return output,past_kv 
        
        
class FeedForward(nn.Module):
    def __init__(self,args:MiniMindConfig,intermediate_size:int = None):
        # intermediate：up_proj 升维到这个中间维度
        super().__init__()
        intermediate_size = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size,intermediate_size,bias = False)
        self.up_proj = nn.Linear(args.hidden_size,intermediate_size,bias = False)
        self.down_proj = nn.Linear(intermediate_size,args.hidden_size,bias = False)        
        self.act_fn = ACT2FN[args.hidden_act] #hidden_act: SiLU；ACT2FN：一个激活函数的包
    def forward(self,x):
        return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)) #这里其实还应该有dropout和残差连接部分



class MOEFeedForward(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size,config.num_experts,bias = False) #[1,hidden_size]@[hidden_size,num_experts]->[1,num_experts]
        self.experts = nn.ModuleList([FeedForward(config,intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self,x):
        batch_size,seq_len,hidden_dim = x.shape #shape是属性，没有括号
        x_flat = x.view(-1,hidden_dim) #展平操作；比如：[2,128,512] -> [256,512]
        scores = F.softmax(self.gate(x_flat),dim=-1)
        #topk_idx选择的是哪个专家
        #topk_weight选择的专家的权重是多少
        topk_weight,topk_idx = torch.topk(scores,k=self.config.num_experts_per_tok,dim=-1,sorted = False)
        if self.config.norm_topk_prob:
            topk_weight = topk_weight/(topk_weight.sum(dim=-1,keepdim = True)+1e-20) #挑选的专家中重新归一化(重归一化权重)
        y = torch.zeros_like(x_flat)
        for i,expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim = -1).nonzero().flatten()
                weight = topk_weight[mask].view(-1,1)
                y.index_add_(0,token_idx,(expert(x_flat[token_idx])*weight).to(y.dtype)) #这里add后面的下划线 是原位修改
            elif self.training:
                y[0,0]+=0*sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef>0:
            load = F.one_hot(topk_idx,self.config.num_experts).float().mean(0)
            self.aux_loss = (load*scores.mean(0)).sum()*self.config.num_experts*self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size,seq_len,hidden_dim)


        

class MiniMindBlock(nn.Module):
    #__init__部分用的参数就对应调用其他class的init的参数，forward的参数就对应调用的其他class的forward参数
    def __init__(self,layer_id:int,config:MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        #两个LayerNorm分别是 Attention部分开始的归一化和 Attention之后，FeedForward部分的归一化
        self.input_layernorm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
    
    def forward(self,hidden_states,position_embeddings,past_key_value = None,use_cache = False,attention_mask = None):
        residual = hidden_states #这里是原始的输入，就是x
        hidden_states,present_key_value = self.self_attn(self.input_layernorm(hidden_states),position_embeddings,
                                                         past_key_value,use_cache,attention_mask)
        hidden_states += residual 
        hidden_states = hidden_states +self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states,present_key_value

        

class MiniMindModel(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size,self.num_hidden_layers = config.vocab_size,config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size) #词表一共多少个词汇？embed到多少维？
        self.dropout = nn.Dropout(config.dropout) #把Embedding的比如768维的向量随机丢弃一些维度
        self.layers = nn.ModuleList([MiniMindBlock(l,config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        freqs_cos,freqs_sin = precompute_freqs_cis(dim=config.head_dim,end=config.max_position_embeddings,rope_base=config.rope_theta,rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos",freqs_cos,persistent = False)
        self.register_buffer("freqs_sin",freqs_sin,persistent = False)

    def forward(self,input_ids #就是输入
                ,attention_mask = None,past_key_values = None,use_cache=False,**kwargs):
        batch_size,seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids)) #防止某一个特征纬度对后续有过大影响，在这里随机丢弃一些
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos+seq_length],self.freqs_sin[start_pos:start_pos+seq_length])
        presents = []
        for layer,past_key_value in zip(self.layers,past_key_values):
            hidden_states,present = layer(
                hidden_states,
                position_embeddings,
                past_key_value = past_key_value,
                use_cache = use_cache,
                attention_mask = attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp,MOEFeedForward)],hidden_states.new_zeros(1).squeeze())
        return hidden_states,presents,aux_loss
        


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids