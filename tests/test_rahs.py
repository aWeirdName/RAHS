"""CPU regression tests; no dataset, network, or pretrained weights required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rahs.data import EEGDataset, PooledEEGDataset, load_features, split_training_conditions
from rahs.engine import evaluate, make_dataset, make_model, train
from rahs.extract_features import latent_importance
from rahs.losses import alignment_loss
from rahs.metrics import retrieval_metrics
from rahs.model import RAHSModel, TemporalSpectralEmbedding
from rahs.resampler import Resampler


torch.set_num_threads(1)


def fixture(root, split="train", subject="sub-01", n=8, repeats=2):
    path = root / "eeg" / subject
    path.mkdir(parents=True, exist_ok=True)
    features = root / "features"
    features.mkdir(exist_ok=True)
    ids = [f"category-{i // 2}/image-{i}.jpg" for i in range(n)]
    # Reverse feature order to catch silent positional alignment bugs.
    tokens = torch.randn(n, 3, 4)
    torch.save({"Latents": tokens, "token_importance": torch.rand(n, 3) + 0.1,
                "image_ids": ids[::-1]}, features / f"image_latent_{split}.pt")
    torch.save({"eeg": np.random.default_rng(1).normal(size=(n, repeats, 2, 8)).astype(np.float32),
                "ch_names": ["O1", "Oz"], "img": np.tile(np.array(ids)[:, None], (1, repeats)),
                "label": np.tile((np.arange(n) // 2)[:, None], (1, repeats))}, path / f"{split}.pt")
    return path / f"{split}.pt", features / f"image_latent_{split}.pt"


def tiny_config(root):
    return {"seed": 7, "setting": "intra-subject",
            "model": {"num_tokens": 3, "dim": 4, "global_depth": 1, "fine_depth": 1,
                      "dropout": 0.0},
            "alignment": {"mode": "topk", "weight": 0.1, "topk": 2, "local_reduction": "mean"},
            "data": {"eeg_dir": str(root / "eeg"), "feature_dir": str(root / "features"),
                     "subject": "sub-01", "all_subjects": ["sub-01", "sub-02"],
                     "channels": ["Oz", "O1"], "time_window": [0, 8], "num_workers": 0},
            "train": {"epochs": 2, "batch_size": 3, "lr": 0.001, "weight_decay": 0.0001,
                      "patience": 2, "min_delta": 0.001, "multi_positive": True},
            "evaluation": {"protocol": "heldout", "validation_fraction": 0.25, "batch_size": 2}}


class ModelAndLossTests(unittest.TestCase):
    def test_hierarchical_model_and_gradient(self):
        model = RAHSModel(channels=2, time_samples=8, num_tokens=3, dim=4,
                          global_depth=1, fine_depth=1, dropout=0)
        eeg = torch.randn(4, 2, 8)
        global_features, fine = model(eeg)
        self.assertEqual(global_features.shape, (4, 12))
        self.assertEqual(fine.shape, (4, 3, 4))
        for mode in ("none", "topk", "token_contrastive"):
            model.zero_grad()
            g, f = model(eeg)
            loss, _ = alignment_loss(model, g, f, torch.randn_like(f), torch.rand(4, 3), mode=mode, topk=2)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(model.fc.weight.grad).all())
        with self.assertRaises(ValueError):
            model(torch.randn(4, 3, 8))

    def test_topk_matches_paper_equation(self):
        model = RAHSModel(channels=1, time_samples=8, num_tokens=3, dim=4)
        image = torch.eye(3)[None].repeat(2, 1, 1)
        importance = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 2.0]])
        _, terms = alignment_loss(model, image.flatten(1), image, image, importance, topk=2)
        # Matched unit vectors have cosine 1; averages are -(3+2)/2 and -(4+2)/2.
        self.assertAlmostEqual(terms["fine_loss"].item(), -2.75)
        _, legacy = alignment_loss(model, image.flatten(1), image, image, importance, topk=2, local_reduction="sum")
        self.assertAlmostEqual(legacy["fine_loss"].item(), -5.5)

    def test_spectrum_conventions_and_residual(self):
        magnitude = TemporalSpectralEmbedding(8, 0, "magnitude").eval()
        power = TemporalSpectralEmbedding(8, 0, "power").eval()
        power.load_state_dict(magnitude.state_dict())
        eeg = torch.randn(2, 1, 8)
        spectrum = torch.fft.rfft(eeg, norm="forward").abs()
        expected = power.value_embedding(eeg) + power.spectral_proj(spectrum.square()) + eeg
        torch.testing.assert_close(power(eeg), expected)
        self.assertFalse(torch.allclose(power(eeg), magnitude(eeg)))
        model = RAHSModel(channels=1, time_samples=8, num_tokens=2, dim=4, dropout=0)
        model.eval()
        with torch.no_grad():
            for parameter in model.prj.parameters():
                parameter.zero_()
        actual, _ = model(eeg)
        torch.testing.assert_close(actual, model.fc(model.emb(eeg).flatten(1)))

    def test_multi_positive_reduces_to_paired_objective(self):
        model = RAHSModel(channels=1, time_samples=8, num_tokens=2, dim=4)
        image = torch.randn(3, 2, 4)
        g = torch.randn(3, 8)
        a, _ = alignment_loss(model, g, image, image, torch.ones(3, 2), mode="none")
        b, _ = alignment_loss(model, g, image, image, torch.ones(3, 2), mode="none", image_indices=torch.arange(3))
        torch.testing.assert_close(a, b)

    def test_resampler_returns_latents_and_importance(self):
        model = Resampler(dim=8, depth=2, dim_head=4, heads=2, num_queries=3,
                          embedding_dim=6, output_dim=10)
        output, latents, attention = model(torch.randn(2, 5, 6))
        self.assertEqual(output.shape, (2, 3, 10))
        self.assertEqual(latents.shape, (2, 3, 8))
        self.assertEqual(attention.shape, (2, 2, 3, 8))
        actual = latent_importance(attention, 3)
        expected = torch.stack([attention[b, :, :, -3:].sum((0, 1)) / 6 for b in range(2)])
        torch.testing.assert_close(actual, expected)


class DataAndEvaluationTests(unittest.TestCase):
    def test_full_bank_ranking_is_batch_invariant(self):
        candidates = torch.eye(7)
        queries = candidates[[6, 0, 3]]
        targets = torch.tensor([6, 0, 3])
        a = retrieval_metrics(queries, candidates, targets, 1)
        b = retrieval_metrics(queries, candidates, targets, 3)
        self.assertEqual(a, b)
        self.assertEqual(a["top1"], 1.0)
        self.assertEqual(a["mrr"], 1.0)
        self.assertEqual(a["num_candidates"], 7)
        wrong = retrieval_metrics(queries, candidates, torch.tensor([5, 1, 2]), 2)
        self.assertEqual(wrong["top1"], 0)

    def test_repetition_and_manifest_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            eeg, feature = fixture(Path(temp))
            dataset = EEGDataset(eeg, feature, channels=["Oz"], time_window=(1, 7), average_repetitions=False)
            self.assertEqual(len(dataset), 16)
            self.assertEqual(dataset[0]["eeg"].shape, (1, 6))
            self.assertEqual(dataset.feature_index[:4].tolist(), [7, 7, 6, 6])
            torch.testing.assert_close(dataset[0]["image_tokens"], dataset[1]["image_tokens"])
            training, validation = split_training_conditions(dataset, 0.25, 7)
            labels = dataset.condition_labels[dataset.condition_index]
            self.assertFalse(set(labels[training].tolist()) & set(labels[validation].tolist()))
            self.assertEqual(len(training) + len(validation), len(dataset))
            pooled = PooledEEGDataset([dataset, dataset])
            self.assertEqual(len(pooled), 32)
            self.assertIs(pooled.tokens, dataset.tokens)

    def test_bad_cache_and_channel_metadata_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            eeg, feature = fixture(Path(temp))
            with self.assertRaises(ValueError):
                EEGDataset(eeg, feature, channels=["missing"], time_window=(0, 8))
            cache = torch.load(feature, weights_only=True)
            cache.pop("image_ids")
            torch.save(cache, feature)
            with self.assertRaisesRegex(ValueError, "lacks image_ids"):
                EEGDataset(eeg, feature, channels=["Oz"], time_window=(0, 8))
            cache["token_importance"][0, 0] = float("nan")
            torch.save(cache, feature)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_features(feature)

    def test_train_checkpoint_and_evaluate_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            fixture(root, "test", n=6)
            config = tiny_config(root)
            metrics = train(config, root / "run", "cpu")
            checkpoint = torch.load(root / "run/best.pt", weights_only=True)
            dataset = make_dataset(config, "test")
            model = make_model(config, dataset)
            model.load_state_dict(checkpoint["model"])
            result = evaluate(model, dataset, config, "cpu")
            self.assertEqual(result, metrics["test"])
            self.assertEqual(result["num_candidates"], 6)
            self.assertTrue((root / "run/split.json").exists())
            with self.assertRaisesRegex(ValueError, "not empty"):
                train(config, root / "run", "cpu")

    def test_inter_subject_excludes_target_training_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # The held-out participant deliberately has no training file.
            fixture(root, "train", subject="sub-02")
            fixture(root, "test", subject="sub-01", n=6)
            config = tiny_config(root)
            config["setting"] = "inter-subject"
            config["train"]["epochs"] = 1
            result = train(config, root / "inter", "cpu")
            self.assertEqual(result["test"]["num_queries"], 6)


if __name__ == "__main__":
    unittest.main()
