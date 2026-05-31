import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from streaminlip.v2.fm_head import DiTBlock, FMHead


class FMHeadTemporalConditionTest(unittest.TestCase):
    def test_fm_head_uses_temporal_condition_order(self):
        torch.manual_seed(0)
        model = FMHead(n_layers=1).eval()
        with torch.no_grad():
            model.final_proj.weight.copy_(torch.eye(512))
            model.final_proj.bias.zero_()
        x = torch.randn(1, 4, 512)
        cond = torch.randn(1, 4, 512)
        cond_reversed = cond.flip(1)
        t = torch.tensor([0.3])

        with torch.no_grad():
            out_a = model._forward_dit(x, cond, t)
            out_b = model._forward_dit(x, cond_reversed, t)

        self.assertFalse(
            torch.allclose(out_a, out_b),
            "FMHead ignored temporal condition order",
        )

    def test_reconstruct_from_cond_uses_temporal_condition(self):
        torch.manual_seed(1)
        model = FMHead(n_layers=1).eval()
        with torch.no_grad():
            model.final_proj.weight.copy_(torch.eye(512))
            model.final_proj.bias.zero_()

        vis = torch.randn(1, 4, 960)
        h_lm = torch.randn(1, 4, 960)
        spk = torch.randn(1, 256)
        h_lm_reversed = h_lm.flip(1)

        with torch.no_grad():
            out_a = model.reconstruct_from_cond(vis, h_lm, spk)
            out_b = model.reconstruct_from_cond(vis, h_lm_reversed, spk)

        self.assertEqual(out_a.shape, (1, 4, 512))
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "reconstruct_from_cond ignored temporal condition order",
        )

    def test_dit_block_cross_attention_uses_condition_tokens(self):
        torch.manual_seed(2)
        block = DiTBlock(512, num_heads=8, use_cross_attn=True).eval()
        with torch.no_grad():
            block.cross_gate.fill_(1.0)
        x = torch.randn(1, 4, 512)
        cond_vec = torch.randn(1, 512)
        cond = torch.randn(1, 4, 512)
        cond_shifted = cond.roll(shifts=1, dims=1)

        with torch.no_grad():
            out_a = block(x, cond_vec, cond)
            out_b = block(x, cond_vec, cond_shifted)

        self.assertEqual(out_a.shape, x.shape)
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "DiT cross-attention ignored condition token order",
        )

    def test_sample_can_backpropagate_for_sample_recon_loss(self):
        torch.manual_seed(3)
        model = FMHead(n_layers=1).train()
        vis = torch.randn(1, 4, 960)
        h_lm = torch.randn(1, 4, 960)
        spk = torch.randn(1, 256)

        pred = model.sample(vis, h_lm, spk, nfe=2)
        loss = pred.float().pow(2).mean()
        loss.backward()

        self.assertIsNotNone(model.final_proj.weight.grad)
        self.assertGreater(
            model.final_proj.weight.grad.abs().sum().item(),
            0.0,
            "sample reconstruction loss cannot backpropagate through inference",
        )


if __name__ == "__main__":
    unittest.main()
