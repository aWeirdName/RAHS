# Release validation

Validated with Python 3.10.19, PyTorch 2.4.1, NumPy 1.24.4 on CPU.

- `python -m unittest discover -s tests -v`: 10 tests passed, including intra-subject and held-out-subject training/evaluation on synthetic data.
- The cleaned model's global and fine outputs were compared against `base/eeg_model_new.py` from experimental commit `b9c8eb4cffe0d95440496b1c55ed3efba345b71a`, with active weights copied, evaluation mode enabled, and magnitude/no-adapter-residual compatibility settings. Both outputs passed `torch.testing.assert_close`; the source top-k loss also matched with summed reduction.
- The public-API training covariance/whitener was compared against the original `sklearn.discriminant_analysis._cov(..., shrinkage='auto')` calculation on synthetic recordings. `numpy.testing.assert_allclose` passed.
- Training, evaluation, feature extraction, and preprocessing command-line help paths completed successfully.
- Python compilation and wheel build (`pip wheel --no-deps --no-build-isolation .`) completed successfully.

These checks validate interfaces and selected numerical behavior. They do not validate published retrieval accuracy, full raw-recording preprocessing, GPU behavior, or extraction using downloaded pretrained weights. Those require the external datasets and model artifacts described in the README.
