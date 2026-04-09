"""Tests for the k-selection heuristics."""

import numpy as np
import pytest

from array_lrr_gwas.select_k import _mp_upper_edge, select_k_elbow, select_k_mp


class TestMpUpperEdge:
    def test_formula(self):
        # For gamma=1 (square matrix), upper = sigma2 * (1+1)^2 = 4*sigma2
        assert _mp_upper_edge(n_markers=100, n_samples=100, sigma2=2.0) == pytest.approx(8.0)

    def test_thin_matrix(self):
        # n_markers >> n_samples: gamma = n_markers/n_samples is large
        # upper = sigma2 * (1 + sqrt(gamma))^2 grows with gamma
        upper_small = _mp_upper_edge(n_markers=100, n_samples=100, sigma2=1.0)
        upper_large = _mp_upper_edge(n_markers=1000, n_samples=100, sigma2=1.0)
        assert upper_large > upper_small


class TestSelectKMp:
    def test_returns_at_least_one(self):
        # Flat singular values (pure noise) → should still return >= 1
        s = np.ones(10)
        k = select_k_mp(s, n_markers=100, n_samples=50)
        assert k >= 1

    def test_detects_signal(self):
        # One large SV followed by small ones
        s = np.array([50.0, 1.0, 0.9, 0.8, 0.7, 0.6])
        k = select_k_mp(s, n_markers=200, n_samples=100)
        assert k >= 1

    def test_all_signal(self):
        # All very large SVs
        s = np.array([100.0, 90.0, 80.0])
        k = select_k_mp(s, n_markers=200, n_samples=100)
        assert k >= 1

    def test_uses_median_not_minimum(self):
        # With a near-zero smallest SV, the old code (using min) would
        # set noise ≈ 0, causing upper ≈ 0, retaining all components.
        # The fixed code (using median) should give a meaningful threshold.
        s = np.array([100.0, 10.0, 5.0, 4.0, 3.0, 0.001])
        k = select_k_mp(s, n_markers=500, n_samples=100)
        # With median-based noise, the threshold is not near-zero;
        # noise components below the threshold should be excluded.
        assert k < len(s)

    # ------------------------------------------------------------------
    # Precise mathematical validation of the noise normalisation
    # ------------------------------------------------------------------

    def test_sigma2_normalised_by_n_markers(self):
        """The σ² estimate must divide median(s²) by n_markers, not n_samples.

        The data matrix X has shape (n_markers × n_samples).  The eigenvalues
        of the sample covariance X^T X / n_markers each have expectation σ²
        under pure noise, so the correct estimator is median(s²) / n_markers.
        Dividing by n_samples instead over-estimates σ² by n_markers/n_samples
        and raises the threshold so high that real signal is missed.

        This test constructs singular values whose ratio n_markers/n_samples is
        large (mimicking a typical GWAS array) and verifies that exactly
        k_true=3 components are returned.  With the wrong normaliser the
        threshold would be 5× too high and all three signals would be missed.
        """
        n_markers, n_samples = 1000, 200
        # Noise SVs: expected value = sqrt(n_markers * sigma2) = sqrt(1000) ≈ 31.6
        s_noise = np.sqrt(float(n_markers))  # unit-variance noise floor
        # MP upper edge on s² = n_markers * sigma2 * (1+sqrt(n_samples/n_markers))²
        sigma2 = 1.0
        mp_upper_s2 = n_markers * sigma2 * (1 + np.sqrt(n_samples / n_markers)) ** 2
        # Signal SVs: 50% above the MP upper edge (clearly detectable)
        s_signal = np.sqrt(mp_upper_s2 * 1.5)

        k_true = 3
        # 3 clear signal components followed by 17 noise components
        s = np.array([s_signal] * k_true + [s_noise] * 17)

        k = select_k_mp(s, n_markers=n_markers, n_samples=n_samples)
        assert k == k_true, (
            f"Expected k={k_true} signal components, got k={k}. "
            "This likely means the σ² estimator is dividing by n_samples "
            "instead of n_markers, making the threshold too high."
        )

    def test_pure_noise_no_false_positive(self):
        """All singular values at the noise floor → should not detect signal.

        Singular values are set at the expected pure-noise level
        (s ≈ sqrt(n_markers * sigma2)).  With the correct normalisation the
        threshold equals the MP upper edge, so none of these values should
        exceed it and k should fall back to the minimum of 1.
        """
        n_markers, n_samples = 1000, 200
        sigma2 = 1.0
        # Singular values at exactly the expected noise mean
        s_noise = np.sqrt(float(n_markers) * sigma2)
        s = np.full(20, s_noise)

        k = select_k_mp(s, n_markers=n_markers, n_samples=n_samples)
        # Noise-level SVs should not be mistaken for signal
        assert k == 1

    def test_exact_threshold_boundary(self):
        """One SV just above and one just below the analytic MP upper edge.

        Constructs singular values with noise SVs that define the median, then
        places one SV 1% above the MP upper edge and one 1% below.  The
        function should detect exactly 1 signal component.
        """
        n_markers, n_samples = 800, 200
        sigma2 = 1.0
        # Pure noise SVs at the expected value (eigenvalue = sigma2 per entry)
        s_noise = np.sqrt(float(n_markers) * sigma2)
        # MP upper edge on s²: n_markers * sigma2 * (1 + sqrt(n_samples/n_markers))^2
        mp_upper_s2 = n_markers * sigma2 * (1 + np.sqrt(n_samples / n_markers)) ** 2

        s_above = np.sqrt(mp_upper_s2 * 1.01)  # just above threshold
        s_below = np.sqrt(mp_upper_s2 * 0.99)  # just below threshold

        # One value clearly above threshold, rest noise
        s_one_signal = np.array([s_above] + [s_noise] * 19)
        assert select_k_mp(s_one_signal, n_markers, n_samples) == 1

        # One value just below threshold, rest noise → no signal → k=1 (minimum)
        s_no_signal = np.array([s_below] + [s_noise] * 19)
        assert select_k_mp(s_no_signal, n_markers, n_samples) == 1

    def test_gwas_typical_dimensions(self):
        """Smoke test with dimensions representative of a GWAS array.

        n_markers >> n_samples is the typical GWAS regime.  The ratio
        n_markers/n_samples can be 50–500×.  The function should still
        recover the correct number of signal components without inflating
        the threshold due to the large gamma.
        """
        n_markers, n_samples = 50_000, 500
        sigma2 = 1.0
        mp_upper_s2 = n_markers * sigma2 * (1 + np.sqrt(n_samples / n_markers)) ** 2
        s_noise = np.sqrt(float(n_markers) * sigma2)
        s_signal = np.sqrt(mp_upper_s2 * 2.0)  # 2× above threshold

        k_true = 5
        s = np.array([s_signal] * k_true + [s_noise] * 20)
        k = select_k_mp(s, n_markers=n_markers, n_samples=n_samples)
        assert k == k_true, (
            f"Expected k={k_true} in GWAS-scale dimensions, got k={k}."
        )


class TestSelectKElbow:
    def test_clear_elbow(self):
        # Sharp drop after first value → elbow at 1
        s = np.array([100.0, 10.0, 9.0, 8.0, 7.0, 6.0])
        k = select_k_elbow(s)
        assert k >= 1

    def test_returns_at_least_one(self):
        s = np.array([1.0, 1.0, 1.0])
        k = select_k_elbow(s)
        assert k >= 1

    def test_short_array(self):
        s = np.array([5.0, 1.0])
        k = select_k_elbow(s)
        assert k == 1

    def test_max_k_respected(self):
        s = np.array([100.0, 50.0, 10.0, 9.0, 8.0])
        k = select_k_elbow(s, max_k=2)
        assert k <= 2
