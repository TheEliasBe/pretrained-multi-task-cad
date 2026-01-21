from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class AlignmentLoss(nn.Module):
    """
    Encourage learned prefix embeddings to live in the LM token-embedding space.

    Modes:
      - 'centroid': align prefix centroid to centroid of sampled code token embeddings.
      - 'best_match': for each prefix token, maximize its best cosine similarity
                      against sampled code token embeddings.

    Typical use:
      loss, stats = loss_fn(prefix_embeds, input_ids, embed_fn, pad_id, eos_id)
    """

    def __init__(
        self,
        mode: str = "centroid",  # 'centroid' | 'best_match'
        weight: float = 0.1,  # scales the auxiliary loss
        sample_tokens: int = 128,  # how many real tokens to sample from the batch
        detach_code: bool = True,  # stop grad on code token embeddings
        normalize: bool = True,  # L2-normalize before cosine sims
        random_sample: bool = True,  # random token sampling (vs. first-N)
    ):
        super().__init__()
        assert mode in {"centroid", "best_match"}
        self.mode = mode
        self.weight = weight
        self.sample_tokens = sample_tokens
        self.detach_code = detach_code
        self.normalize = normalize
        self.random_sample = random_sample

    @torch.no_grad()
    def _sample_valid_tokens(
        self,
        input_ids: torch.Tensor,
        pad_id: int,
        eos_id: int,
        ignore_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Returns a 1D tensor of up to `sample_tokens` valid token IDs from input_ids.
        Valid = not {pad, eos, ignore_id}.
        """
        valid = (input_ids != pad_id) & (input_ids != eos_id) & (input_ids != ignore_id)
        flat = input_ids[valid]
        if flat.numel() == 0:
            return None
        if self.random_sample and flat.numel() > self.sample_tokens:
            idx = torch.randperm(flat.numel(), device=flat.device)[: self.sample_tokens]
            return flat[idx]
        return flat[: self.sample_tokens]

    def forward(
        self,
        prefix_embeds: torch.Tensor,  # [B, T_pref, D]
        input_ids: torch.Tensor,  # [B, T_seq]
        embed_fn,  # callable: token_ids -> [N, D] (e.g., model.get_input_embeddings())
        pad_id: int,
        eos_id: int,
        ignore_id: int = 0,
    ) -> (torch.Tensor, Dict[str, Any]):
        # 1) sample real code tokens
        sampled = self._sample_valid_tokens(input_ids, pad_id, eos_id, ignore_id)

        if sampled is None or sampled.numel() == 0:
            # No valid tokens → return zero loss (on the right device/dtype)
            zero = prefix_embeds.new_tensor(0.0)
            return zero, {"alignment_similarity": float("nan"), "alignment_tokens": 0}

        # 2) get code embeddings
        #    optionally detach to prevent updating the token embedding table
        code_embeds = embed_fn(sampled)  # [N, D]
        if self.detach_code:
            code_embeds = code_embeds.detach()

        # 3) normalize if requested
        if self.normalize:
            code_embeds = torch.nn.functional.normalize(code_embeds, dim=-1)

        if self.mode == "centroid":
            # ---- centroid-to-centroid alignment ----
            # code centroid (1, D)
            code_centroid = code_embeds.mean(dim=0, keepdim=True)
            if self.normalize:
                code_centroid = torch.nn.functional.normalize(code_centroid, dim=-1)

            # prefix centroid per-sample (B, D)
            prefix_centroid = prefix_embeds.mean(dim=1)
            if self.normalize:
                prefix_centroid = torch.nn.functional.normalize(prefix_centroid, dim=-1)

            # cosine similarity per sample → mean over batch
            sim = (prefix_centroid * code_centroid).sum(dim=-1).mean()

        else:
            # ---- best-match alignment ----
            # flatten prefix tokens across batch (B*T_pref, D)
            pref = prefix_embeds.reshape(-1, prefix_embeds.size(-1))
            if self.normalize:
                pref = torch.nn.functional.normalize(pref, dim=-1)

            # all-pairs cosine sims (B*T_pref, N)
            sims = pref @ code_embeds.t()
            per_prefix_best = sims.max(dim=1).values  # (B*T_pref,)
            sim = per_prefix_best.mean()

        # 4) loss is negative similarity (maximize similarity)
        loss = (1.0 - sim) * self.weight

        return loss, sim
