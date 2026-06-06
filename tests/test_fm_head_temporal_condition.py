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

    def test_denoise_from_noise_predicts_latent_shaped_endpoint(self):
        torch.manual_seed(4)
        model = FMHead(n_layers=1).train()
        vis = torch.randn(2, 5, 960)
        h_lm = torch.randn(2, 5, 960)
        spk = torch.randn(2, 256)
        noise = torch.randn(2, 5, 512)
        t = torch.tensor([0.0, 0.5])

        pred = model.denoise_from_noise(vis, h_lm, spk, noise, t)

        self.assertEqual(pred.shape, noise.shape)

    def test_text_token_cross_attention_uses_raw_token_order(self):
        torch.manual_seed(5)
        model = FMHead(
            n_layers=1,
            use_cross_attn=True,
            use_text_token_cross_attn=True,
        ).eval()
        with torch.no_grad():
            model.final_proj.weight.copy_(torch.eye(512))
            model.final_proj.bias.zero_()
            for block in model.blocks:
                block.cross_gate.fill_(1.0)

        vis = torch.randn(1, 4, 960)
        h_lm = torch.zeros(1, 4, 960)
        spk = torch.randn(1, 256)
        text_tokens = torch.randn(1, 3, 960)
        reversed_tokens = text_tokens.flip(1)
        x = torch.randn(1, 4, 512)
        t = torch.tensor([0.5])

        with torch.no_grad():
            cond, raw_tokens = model._build_cond(
                vis, h_lm, spk, text_tokens=text_tokens
            )
            cond_reversed, raw_tokens_reversed = model._build_cond(
                vis, h_lm, spk, text_tokens=reversed_tokens
            )
            out_a = model._forward_dit(x, cond, t, raw_tokens)
            out_b = model._forward_dit(x, cond_reversed, t, raw_tokens_reversed)

        self.assertEqual(raw_tokens.shape, (1, 3, 512))
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "raw text token cross-attention ignored token order",
        )

    def test_text_token_cross_attention_masks_padding_tokens(self):
        torch.manual_seed(6)
        model = FMHead(
            n_layers=1,
            use_cross_attn=True,
            use_text_token_cross_attn=True,
        ).eval()
        with torch.no_grad():
            model.final_proj.weight.copy_(torch.eye(512))
            model.final_proj.bias.zero_()
            for block in model.blocks:
                block.cross_gate.fill_(1.0)

        vis = torch.randn(1, 4, 960)
        h_lm = torch.zeros(1, 4, 960)
        spk = torch.randn(1, 256)
        valid = torch.randn(1, 1, 960)
        padding_a = torch.zeros(1, 2, 960)
        padding_b = torch.randn(1, 2, 960) * 10
        mask = torch.tensor([[True, False, False]])
        x = torch.randn(1, 4, 512)
        t = torch.tensor([0.5])

        with torch.no_grad():
            cond_a, tokens_a = model._build_cond(
                vis,
                h_lm,
                spk,
                text_tokens=torch.cat([valid, padding_a], dim=1),
            )
            cond_b, tokens_b = model._build_cond(
                vis,
                h_lm,
                spk,
                text_tokens=torch.cat([valid, padding_b], dim=1),
            )
            out_a = model._forward_dit(x, cond_a, t, tokens_a, cond_token_mask=mask)
            out_b = model._forward_dit(x, cond_b, t, tokens_b, cond_token_mask=mask)

        self.assertTrue(
            torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5),
            "raw text token cross-attention attended to masked padding tokens",
        )

    def test_frame_cross_attention_ignores_raw_text_token_mask_shape(self):
        torch.manual_seed(11)
        model = FMHead(n_layers=1, use_cross_attn=True).eval()
        vis = torch.randn(2, 5, 960)
        h_lm = torch.randn(2, 5, 960)
        spk = torch.randn(2, 256)
        text_tokens = torch.randn(2, 3, 960)
        text_mask = torch.ones(2, 3, dtype=torch.bool)

        out = model.reconstruct_from_cond(
            vis,
            h_lm,
            spk,
            text_tokens=text_tokens,
            text_token_mask=text_mask,
        )

        self.assertEqual(out.shape, (2, 5, 512))

    def test_extra_condition_changes_temporal_condition(self):
        torch.manual_seed(7)
        model = FMHead(n_layers=1, extra_cond_dim=4).eval()
        with torch.no_grad():
            model.final_proj.weight.copy_(torch.eye(512))
            model.final_proj.bias.zero_()

        vis = torch.randn(1, 4, 960)
        h_lm = torch.randn(1, 4, 960)
        spk = torch.randn(1, 256)
        extra_a = torch.zeros(1, 4, 4)
        extra_b = torch.zeros(1, 4, 4)
        extra_b[:, 2] = 1.0

        with torch.no_grad():
            out_a = model.reconstruct_from_cond(vis, h_lm, spk, extra_cond=extra_a)
            out_b = model.reconstruct_from_cond(vis, h_lm, spk, extra_cond=extra_b)

        self.assertEqual(out_a.shape, (1, 4, 512))
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "extra condition was ignored by FMHead",
        )

    def test_forward_inference_accepts_extra_condition(self):
        torch.manual_seed(8)
        model = FMHead(n_layers=1, extra_cond_dim=4).eval()
        vis = torch.randn(1, 3, 960)
        h_lm = torch.zeros(1, 3, 960)
        spk = torch.randn(1, 256)
        extra = torch.randn(1, 3, 4)

        out = model.forward_inference(vis, h_lm, spk, nfe=1, extra_cond=extra)

        self.assertEqual(out.shape, (1, 3, 512))

    def test_predict_energy_from_condition_returns_frame_level_scalar(self):
        torch.manual_seed(10)
        model = FMHead(n_layers=1, extra_cond_dim=1).eval()
        vis = torch.randn(2, 3, 960)
        h_lm = torch.zeros(2, 3, 960)
        spk = torch.randn(2, 256)

        pred = model.predict_extra_condition(vis, h_lm, spk)

        self.assertEqual(pred.shape, (2, 3, 1))

    def test_ctc_topk_tokens_use_learned_token_embeddings(self):
        torch.manual_seed(9)
        model = FMHead(
            n_layers=1,
            ctc_vocab_size=8,
            ctc_topk=2,
            ctc_token_emb_dim=4,
        ).eval()
        with torch.no_grad():
            model.final_proj.weight.copy_(torch.eye(512))
            model.final_proj.bias.zero_()

        vis = torch.randn(1, 3, 960)
        h_lm = torch.zeros(1, 3, 960)
        spk = torch.randn(1, 256)
        ctc_topk_ids_a = torch.tensor([[[1, 2], [1, 2], [1, 2]]])
        ctc_topk_ids_b = torch.tensor([[[3, 4], [3, 4], [3, 4]]])
        ctc_topk_probs = torch.full((1, 3, 2), 0.5)

        with torch.no_grad():
            out_a = model.reconstruct_from_cond(
                vis,
                h_lm,
                spk,
                ctc_topk_ids=ctc_topk_ids_a,
                ctc_topk_probs=ctc_topk_probs,
            )
            out_b = model.reconstruct_from_cond(
                vis,
                h_lm,
                spk,
                ctc_topk_ids=ctc_topk_ids_b,
                ctc_topk_probs=ctc_topk_probs,
            )

        self.assertEqual(out_a.shape, (1, 3, 512))
        self.assertFalse(
            torch.allclose(out_a, out_b),
            "CTC token ids were not represented by learned embeddings",
        )

    def test_audio_prompt_learned_pool_condition_can_use_prompt_tokens(self):
        torch.manual_seed(16)
        model = FMHead(
            n_layers=1,
            audio_prompt_dim=512,
            audio_prompt_learned_pool_cond=True,
        ).eval()
        vis = torch.randn(1, 4, 960)
        h_lm = torch.randn(1, 4, 960)
        spk = torch.randn(1, 256)
        prompt_a = torch.zeros(1, 3, 512)
        prompt_b = prompt_a.clone()
        prompt_b[:, 1] = 1.0

        with torch.no_grad():
            model.audio_prompt_learned_pool_proj.weight.copy_(torch.eye(512))
            model.audio_prompt_learned_pool_proj.bias.zero_()
            cond_a, _ = model._build_cond(vis, h_lm, spk, audio_prompt=prompt_a)
            cond_b, _ = model._build_cond(vis, h_lm, spk, audio_prompt=prompt_b)

        self.assertFalse(
            torch.allclose(cond_a, cond_b),
            "learned audio prompt pool condition ignored prompt token values",
        )

    def test_audio_prompt_learned_pool_condition_starts_as_noop(self):
        torch.manual_seed(17)
        base = FMHead(n_layers=1, audio_prompt_dim=512).eval()
        learned_pool = FMHead(
            n_layers=1,
            audio_prompt_dim=512,
            audio_prompt_learned_pool_cond=True,
        ).eval()
        learned_pool.load_state_dict(base.state_dict(), strict=False)
        vis = torch.randn(1, 4, 960)
        h_lm = torch.randn(1, 4, 960)
        spk = torch.randn(1, 256)
        prompt = torch.randn(1, 3, 512)

        with torch.no_grad():
            cond_base, _ = base._build_cond(vis, h_lm, spk, audio_prompt=prompt)
            cond_learned, _ = learned_pool._build_cond(vis, h_lm, spk, audio_prompt=prompt)

        torch.testing.assert_close(cond_base, cond_learned)
        self.assertIsNotNone(learned_pool.audio_prompt_learned_pool_proj)


if __name__ == "__main__":
    unittest.main()
