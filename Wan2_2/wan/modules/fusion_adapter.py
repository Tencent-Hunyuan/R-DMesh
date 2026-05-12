import torch
import torch.nn as nn


class FusionAdapter(nn.Module):
    """
    适配器：将DyMeshVCDiT_joint适配到WanModel的融合流程
    DDP兼容版本
    """
    
    def __init__(self, fusion_model):
        """
        Args:
            fusion_model: DyMeshVCDiT_joint实例（可能已被DDP包装）
        """
        super().__init__()
        
        # ✅ 保存原始引用（用于梯度传播）
        self.fusion_model = fusion_model
        
        # ✅ 安全地获取实际模型（处理DDP包装）
        actual_model = self._unwrap_model(fusion_model)
        
        # ✅ 从unwrap后的模型获取组件（引用，不是复制）
        self.blocks = actual_model.backbone.resblocks
        self.input_proj = actual_model.input_proj
        self.ln_pre = actual_model.ln_pre
        self.ln_post = actual_model.ln_post
        self.output_proj = actual_model.output_proj
        
        # ✅ 用于存储中间特征（每个进程独立）
        self._intermediate_features = []
    
    @staticmethod
    def _unwrap_model(model):
        """
        安全地unwrap DDP/DP/FSDP等包装的模型
        
        Args:
            model: 可能被包装的模型
            
        Returns:
            实际的模型（未包装的）
        """
        # 处理 DistributedDataParallel
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.module
        
        # 处理 DataParallel
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        
        # 处理 FSDP (如果使用)
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if isinstance(model, FSDP):
                return model.module
        except ImportError:
            pass
        
        # 处理通用的 .module 属性
        if hasattr(model, 'module'):
            return model.module
        
        # 如果没有包装，直接返回
        return model
    
    def clear_intermediates(self):
        """清空中间特征"""
        self._intermediate_features = []
    
    def get_intermediate_features(self):
        """获取中间特征"""
        return self._intermediate_features
    
    def preprocess_4d_input(self, x_4d):
        """
        预处理4D输入（只执行一次）
        
        Args:
            x_4d: 原始4D输入 (B, seq_len, input_channels)
            
        Returns:
            h_4d: 预处理后的4D特征 (B, seq_len, width)
        """
        h_4d = self.input_proj(x_4d)
        h_4d = self.ln_pre(h_4d)
        return h_4d
    
    def prepare_time_embedding(self, t):
        """
        从原始时间步生成时间嵌入
        
        Args:
            t: 原始时间步 (B,) 或 (B, seq_len)
            
        Returns:
            t_emb_4d: 4D分支时间嵌入 (B, width)
            t_emb_wan: WAN分支时间嵌入 (B, width)
        """
        # 如果 t 是 (B, seq_len)，取第一个时间步（它们都相同）
        if t.dim() == 2:
            t = t[:, 0]
        
        # ✅ 使用unwrap后的模型调用prepare_t_emb
        actual_model = self._unwrap_model(self.fusion_model)
        t_emb_4d, t_emb_wan = actual_model.prepare_t_emb(t)
        
        return t_emb_4d, t_emb_wan
    
    def postprocess_4d_output(self, h_4d):
        """
        后处理4D输出（只执行一次）
        
        Args:
            h_4d: 处理后的4D特征 (B, seq_len, width)
            
        Returns:
            output_4d: 最终4D输出 (B, seq_len, output_channels)
        """
        h_4d = self.ln_post(h_4d)
        output_4d = self.output_proj(h_4d)
        return output_4d
    
    def forward_single_block(
        self, 
        h_4d, 
        wan_latent, 
        t_emb_4d, 
        t_emb_wan, 
        block_idx, 
        store_intermediate=False
    ):
        """
        执行单个fusion block
        
        Args:
            h_4d: 预处理后的4D特征 (B, seq_len, width)
            wan_latent: WAN条件特征 (B, seq_len, 3072)
            t_emb_4d: 4D分支时间嵌入 (B, width)
            t_emb_wan: WAN分支时间嵌入 (B, width)
            block_idx: block索引
            store_intermediate: 是否存储中间特征
            
        Returns:
            h_4d_out: 更新后的4D特征
            wan_latent_out: 更新后的WAN特征
        """
        # ✅ 检查block_idx是否有效
        if block_idx >= len(self.blocks):
            raise IndexError(
                f"block_idx {block_idx} out of range, "
                f"total blocks: {len(self.blocks)}"
            )
        
        block = self.blocks[block_idx]
        
        # 通过fusion block
        h_4d_out, wan_latent_out = block(
            x=h_4d,
            t_emb=t_emb_4d,
            vid_emb=wan_latent,
            t_emb_wan=t_emb_wan
        )
        
        if store_intermediate:
            # ✅ 使用detach().clone()避免保存整个计算图
            self._intermediate_features.append({
                'block_idx': block_idx,
                'h_4d': h_4d_out.detach().clone(),
                'wan_latent': wan_latent_out.detach().clone()
            })
        
        return h_4d_out, wan_latent_out
    
    def forward_all_blocks(
        self, 
        h_4d, 
        wan_latent, 
        t_emb_4d, 
        t_emb_wan, 
        store_intermediates=False
    ):
        """
        执行所有fusion blocks
        
        Args:
            h_4d: 预处理后的4D特征 (B, seq_len, width)
            wan_latent: WAN条件特征 (B, seq_len, 3072)
            t_emb_4d: 4D分支时间嵌入 (B, width)
            t_emb_wan: WAN分支时间嵌入 (B, width)
            store_intermediates: 是否存储中间特征
            
        Returns:
            h_4d: 处理后的4D特征
            wan_latent: 更新后的WAN特征
        """
        if store_intermediates:
            self.clear_intermediates()
        
        for idx, block in enumerate(self.blocks):
            h_4d, wan_latent = block(
                x=h_4d,
                t_emb=t_emb_4d,
                vid_emb=wan_latent,
                t_emb_wan=t_emb_wan
            )
            
            if store_intermediates:
                # ✅ 使用detach().clone()避免保存整个计算图
                self._intermediate_features.append({
                    'block_idx': idx,
                    'h_4d': h_4d.detach().clone(),
                    'wan_latent': wan_latent.detach().clone()
                })
        
        return h_4d, wan_latent


class WanFusionMixin:
    """为WanModel提供融合功能的Mixin类"""
    
    def _get_or_create_fusion_adapter(self, fusion_net):
        """
        获取或创建FusionAdapter（延迟初始化，DDP安全）
        
        Args:
            fusion_net: DyMeshVCDiT_joint模型（可能被DDP包装）
            
        Returns:
            FusionAdapter实例
        """
        # ✅ 检查是否需要创建新的adapter
        # 如果fusion_net改变了，需要重新创建
        if not hasattr(self, '_fusion_adapter') or self._fusion_adapter is None:
            self._fusion_adapter = FusionAdapter(fusion_net)
            self._fusion_net_id = id(fusion_net)
        elif id(fusion_net) != self._fusion_net_id:
            # fusion_net改变了，重新创建
            self._fusion_adapter = FusionAdapter(fusion_net)
            self._fusion_net_id = id(fusion_net)
        
        return self._fusion_adapter
    
    def _apply_fusion_single(
        self, 
        wan_x, 
        h_4d, 
        t_emb_4d, 
        t_emb_wan, 
        fusion_adapter, 
        store_intermediate=False
    ):
        """
        模式1: 一次性融合所有fusion blocks
        
        Args:
            wan_x: WAN当前特征 (B, seq_len, dim_wan)
            h_4d: 预处理后的4D特征 (B, seq_len, width)
            t_emb_4d: 4D分支时间嵌入 (B, width)
            t_emb_wan: WAN分支时间嵌入 (B, width)
            fusion_adapter: FusionAdapter实例
            store_intermediate: 是否存储中间特征
            
        Returns:
            wan_x_out: 更新后的WAN特征
            h_4d_out: 更新后的4D特征
        """
        h_4d_out, wan_x_out = fusion_adapter.forward_all_blocks(
            h_4d=h_4d,
            wan_latent=wan_x,
            t_emb_4d=t_emb_4d,
            t_emb_wan=t_emb_wan,
            store_intermediates=store_intermediate
        )
        
        return wan_x_out, h_4d_out
    
    def _apply_fusion_interleaved(
        self, 
        wan_x, 
        h_4d, 
        t_emb_4d, 
        t_emb_wan, 
        fusion_adapter, 
        block_idx, 
        store_intermediate=False
    ):
        """
        模式2: 单次交替融合
        
        Args:
            wan_x: WAN当前特征 (B, seq_len, dim_wan)
            h_4d: 当前4D特征 (B, seq_len, width)
            t_emb_4d: 4D分支时间嵌入 (B, width)
            t_emb_wan: WAN分支时间嵌入 (B, width)
            fusion_adapter: FusionAdapter实例
            block_idx: 当前block索引
            store_intermediate: 是否存储中间特征
            
        Returns:
            wan_x_out: 更新后的WAN特征
            h_4d_out: 更新后的4D特征
        """
        h_4d_out, wan_x_out = fusion_adapter.forward_single_block(
            h_4d=h_4d,
            wan_latent=wan_x,
            t_emb_4d=t_emb_4d,
            t_emb_wan=t_emb_wan,
            block_idx=block_idx,
            store_intermediate=store_intermediate
        )
        
        return wan_x_out, h_4d_out