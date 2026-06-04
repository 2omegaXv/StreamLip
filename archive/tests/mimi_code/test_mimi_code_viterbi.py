import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_mimi_code_viterbi import viterbi_topk_decode


class MimiCodeViterbiTest(unittest.TestCase):
    def test_viterbi_topk_decode_uses_transition_scores(self):
        top_ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
        top_logits = torch.tensor([
            [3.0, 2.8],
            [2.0, 2.1],
            [3.0, 2.8],
        ])
        logp = torch.full((7, 7), -10.0)
        bos = 6
        logp[bos, 2] = 0.0
        logp[2, 4] = 0.0
        logp[4, 6] = 0.0

        pred = viterbi_topk_decode(top_ids, top_logits, logp, bos_id=bos, alpha=1.0)

        self.assertEqual(pred.tolist(), [2, 4, 6])


if __name__ == "__main__":
    unittest.main()
