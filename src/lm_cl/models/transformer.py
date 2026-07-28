from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lm_cl.config import ModelConfig


@dataclass
class LMOutput:
    loss_sum: torch.Tensor | None
    target_count: torch.Tensor
    mean_loss: torch.Tensor | None
    logits: torch.Tensor | None
    final_write_memory: torch.Tensor | None = None
    segment_write_memories: tuple[torch.Tensor, ...] | None = None
    text_position_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class ParameterBreakdown:
    token_embeddings: int
    positional_embeddings: int
    non_embedding: int
    total: int


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.qkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.use_bias,
        )
        self.out_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.use_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )
        return self.out_proj(attended)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gelu_approximation = config.gelu_approximation
        self.fc_in = nn.Linear(
            config.hidden_size,
            config.mlp_hidden_size,
            bias=config.use_bias,
        )
        self.fc_out = nn.Linear(
            config.mlp_hidden_size,
            config.hidden_size,
            bias=config.use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(
            F.gelu(self.fc_in(x), approximate=self.gelu_approximation)
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            elementwise_affine=True,
            bias=True,
        )
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            elementwise_affine=True,
            bias=True,
        )
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.ln_1(x),
            attention_mask=attention_mask,
        )
        return x + self.mlp(self.ln_2(x))


class TiedOutputProjection(nn.Module):
    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        self.embeddings = embeddings

    @property
    def weight(self) -> nn.Parameter:
        return self.embeddings.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight)


class ZyphraTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.lm_head = TiedOutputProjection(self.token_embeddings)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.layers)]
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            elementwise_affine=True,
            bias=True,
        )
        self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def lm_head_weight(self) -> nn.Parameter:
        return self.lm_head.weight

    def parameter_breakdown(self) -> ParameterBreakdown:
        token = self.token_embeddings.weight.numel()
        positional = self.position_embeddings.weight.numel()
        total = sum(parameter.numel() for parameter in self.parameters())
        return ParameterBreakdown(
            token_embeddings=token,
            positional_embeddings=positional,
            non_embedding=total - token - positional,
            total=total,
        )

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[int, int]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long dtype")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Input length {sequence_length} exceeds maximum "
                f"{self.config.max_position_embeddings}"
            )
        if sequence_length == 0:
            raise ValueError("Input sequence must not be empty")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        return batch_size, sequence_length

    def embed_text(
        self,
        input_ids: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = input_ids.shape[1]
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=input_ids.device,
        )
        if sequence_length and int(positions[-1]) >= (
            self.config.max_position_embeddings
        ):
            raise ValueError("Text position exceeds maximum position embedding")
        hidden = self.token_embeddings(input_ids)
        hidden = hidden + self.position_embeddings(positions).unsqueeze(0)
        return hidden, positions

    def transform_hidden(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.final_layer_norm(hidden)

    def _loss_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        *,
        ignore_index: int,
        compute_loss: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        loss_sum = None
        mean_loss = None
        target_count = torch.zeros((), device=input_ids.device, dtype=torch.long)
        if compute_loss:
            target_labels = input_ids if labels is None else labels
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_targets = target_labels[:, 1:].contiguous()
            target_count = shifted_targets.ne(ignore_index).sum()
            loss_sum = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_targets.view(-1),
                ignore_index=ignore_index,
                reduction="sum",
            )
            mean_loss = loss_sum / target_count.clamp_min(1)
        return loss_sum, target_count, mean_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        ignore_index: int = -100,
        return_logits: bool = False,
        compute_loss: bool = True,
    ) -> LMOutput:
        _, sequence_length = self._validate_inputs(input_ids, labels)
        hidden, positions = self.embed_text(input_ids)
        hidden = self.transform_hidden(hidden)
        logits = self.lm_head(hidden)
        loss_sum, target_count, mean_loss = self._loss_from_logits(
            logits,
            input_ids,
            labels,
            ignore_index=ignore_index,
            compute_loss=compute_loss,
        )

        return LMOutput(
            loss_sum=loss_sum,
            target_count=target_count,
            mean_loss=mean_loss,
            logits=logits if return_logits else None,
            text_position_ids=positions,
        )


class RMTZyphraTransformer(ZyphraTransformer):
    """The exact Zyphra backbone with recurrent read/write hidden-state memory."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        memory_tokens: int,
        segment_length: int,
    ):
        if memory_tokens <= 0:
            raise ValueError("memory_tokens must be positive")
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        super().__init__(config)
        self.memory_token_count = memory_tokens
        self.segment_length = segment_length
        self._attention_mask_cache: dict[
            tuple[int, str, int | None, torch.dtype], torch.Tensor
        ] = {}
        self.initial_memory = nn.Parameter(
            torch.empty(memory_tokens, config.hidden_size)
        )
        nn.init.normal_(
            self.initial_memory,
            mean=0.0,
            std=config.initializer_std,
        )

    @property
    def backbone_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()) - (
            self.initial_memory.numel()
        )

    @property
    def initial_memory_parameter_count(self) -> int:
        return self.initial_memory.numel()

    @staticmethod
    def build_attention_mask(
        memory_tokens: int,
        text_tokens: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if memory_tokens <= 0 or text_tokens <= 0:
            raise ValueError("Memory and text token counts must be positive")
        total = 2 * memory_tokens + text_tokens
        text_start = memory_tokens
        write_start = memory_tokens + text_tokens
        mask = torch.full(
            (total, total),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        # Read memory can read only incoming read memory.
        mask[:text_start, :text_start] = 0
        # Text can read incoming memory and causal current-segment text.
        mask[text_start:write_start, :text_start] = 0
        text_mask = mask[
            text_start:write_start,
            text_start:write_start,
        ]
        text_mask.copy_(torch.triu(text_mask, diagonal=1))
        # Write rows are diagnostic/next-segment state and can summarize all
        # incoming read memory, current text, and write positions.
        mask[write_start:, :] = 0
        return mask

    def _attention_mask(
        self,
        text_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (
            text_tokens,
            device.type,
            device.index,
            dtype,
        )
        mask = self._attention_mask_cache.get(key)
        if mask is None:
            mask = self.build_attention_mask(
                self.memory_token_count,
                text_tokens,
                device=device,
                dtype=dtype,
            )
            self._attention_mask_cache[key] = mask
        return mask

    def _expand_memory(
        self,
        memory: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        expected = (self.memory_token_count, self.config.hidden_size)
        if memory.ndim == 2:
            if tuple(memory.shape) != expected:
                raise ValueError(
                    f"Root memory shape must be {expected}, got {tuple(memory.shape)}"
                )
            return memory.unsqueeze(0).expand(batch_size, -1, -1)
        if memory.ndim == 3:
            expected_batched = (batch_size, *expected)
            if tuple(memory.shape) != expected_batched:
                raise ValueError(
                    "Batched root memory shape must be "
                    f"{expected_batched}, got {tuple(memory.shape)}"
                )
            return memory
        raise ValueError("Root memory must have rank two or three")

    def forward_segment(
        self,
        read_memory: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        position_offset: int,
        write_memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, text_tokens = input_ids.shape
        read = self._expand_memory(read_memory, batch_size=batch_size)
        write = (
            read
            if write_memory is None
            else self._expand_memory(write_memory, batch_size=batch_size)
        )
        text, positions = self.embed_text(
            input_ids,
            position_offset=position_offset,
        )
        hidden = torch.cat((read, text, write), dim=1)
        mask = self._attention_mask(
            text_tokens,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        hidden = self.transform_hidden(hidden, attention_mask=mask)
        text_start = self.memory_token_count
        write_start = text_start + text_tokens
        return (
            hidden[:, text_start:write_start],
            hidden[:, write_start:],
            positions,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        root_memory: torch.Tensor | None = None,
        ignore_index: int = -100,
        return_logits: bool = False,
        compute_loss: bool = True,
    ) -> LMOutput:
        _, sequence_length = self._validate_inputs(input_ids, labels)
        memory: torch.Tensor = (
            self.initial_memory if root_memory is None else root_memory
        )
        text_hidden: list[torch.Tensor] = []
        writes: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for start in range(0, sequence_length, self.segment_length):
            end = min(start + self.segment_length, sequence_length)
            segment_hidden, memory, segment_positions = self.forward_segment(
                memory,
                input_ids[:, start:end],
                position_offset=start,
            )
            text_hidden.append(segment_hidden)
            writes.append(memory)
            positions.append(segment_positions)
        hidden = torch.cat(text_hidden, dim=1)
        logits = self.lm_head(hidden)
        loss_sum, target_count, mean_loss = self._loss_from_logits(
            logits,
            input_ids,
            labels,
            ignore_index=ignore_index,
            compute_loss=compute_loss,
        )
        return LMOutput(
            loss_sum=loss_sum,
            target_count=target_count,
            mean_loss=mean_loss,
            logits=logits if return_logits else None,
            final_write_memory=writes[-1],
            segment_write_memories=tuple(writes),
            text_position_ids=torch.cat(positions),
        )
