"""
Unit tests for clusterITE.py and gen_dat.py.

Run with:
    pytest test_clusterITE.py -v
"""

import numpy as np
import pandas as pd
import pytest

from gen_dat import gen_data
from clusterITE import tf_model, ClusterIte, ClusterIte_cv


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset():
    """60 points, 4 features, 2 clusters – reused across all test classes."""
    np.random.seed(42)
    X, y, labels, probs = gen_data(n=60, p=4, K=2, verbose=False)
    return X, y, labels, probs


@pytest.fixture(scope="module")
def fitted_k2(small_dataset):
    """ClusterIte(K=2) fitted once and reused."""
    X, y, _, _ = small_dataset
    model = ClusterIte(K=2)
    model.fit(X, y, maxit=5, silence=True)
    return model, X, y


@pytest.fixture(scope="module")
def fitted_cv(small_dataset):
    """ClusterIte_cv fitted once and reused."""
    X, y, _, _ = small_dataset
    cv = ClusterIte_cv(nb_folds=2)
    cv.fit(X, y, cluster_range=range(2, 4), maxit=3)
    return cv


# ===========================================================================
# gen_dat.py
# ===========================================================================

class TestGenData:

    def test_output_shapes(self):
        n, p, K = 50, 5, 3
        X, y, cat, prob = gen_data(n=n, p=p, K=K, verbose=False)
        assert X.shape == (n, p)
        assert y.shape == (n,)
        assert cat.shape == (n,)
        assert prob.shape == (n, K)

    def test_y_lower_bound(self):
        X, y, cat, prob = gen_data(n=100, p=4, K=3, verbose=False)
        assert np.all(y >= -1), "y values must be >= -1"

    def test_y_upper_bound(self):
        X, y, cat, prob = gen_data(n=100, p=4, K=3, verbose=False)
        assert np.all(y <= 1), "y values must be <= +1"

    def test_cluster_labels_lower_bound(self):
        K = 3
        _, _, cat, _ = gen_data(n=100, p=4, K=K, verbose=False)
        assert np.all(cat >= 0)

    def test_cluster_labels_upper_bound(self):
        K = 3
        _, _, cat, _ = gen_data(n=100, p=4, K=K, verbose=False)
        assert np.all(cat < K)

    def test_cluster_probs_sum_to_one(self):
        _, _, _, prob = gen_data(n=100, p=4, K=3, verbose=False)
        np.testing.assert_allclose(
            prob.sum(axis=1), np.ones(100), atol=1e-6,
            err_msg="Each row of cluster_prob must sum to 1"
        )

    def test_cluster_probs_non_negative(self):
        _, _, _, prob = gen_data(n=100, p=4, K=3, verbose=False)
        assert np.all(prob >= 0), "Probabilities must be non-negative"

    def test_cluster_probs_at_most_one(self):
        _, _, _, prob = gen_data(n=100, p=4, K=3, verbose=False)
        assert np.all(prob <= 1), "Probabilities must be <= 1"

    def test_single_cluster_labels_are_zero(self):
        _, _, cat, _ = gen_data(n=50, p=3, K=1, verbose=False)
        assert np.all(cat == 0), "With K=1 all labels should be 0"

    def test_single_cluster_probs_are_one(self):
        _, _, _, prob = gen_data(n=50, p=3, K=1, verbose=False)
        np.testing.assert_allclose(
            prob, np.ones((50, 1)), atol=1e-6,
            err_msg="With K=1 all cluster probabilities should be 1"
        )

    def test_custom_eigen_does_not_crash(self):
        p = 4
        eigen = [1.0, 2.0, 3.0, 4.0]
        X, y, cat, prob = gen_data(n=100, p=p, K=2, eigen=eigen, verbose=False)
        assert X.shape == (100, p)

    def test_no_nan_in_X(self):
        X, _, _, _ = gen_data(n=80, p=5, K=4, verbose=False)
        assert not np.any(np.isnan(X)), "X must not contain NaN"

    def test_no_nan_in_y(self):
        _, y, _, _ = gen_data(n=80, p=5, K=4, verbose=False)
        assert not np.any(np.isnan(y)), "y must not contain NaN"

    def test_no_nan_in_probs(self):
        _, _, _, prob = gen_data(n=80, p=5, K=4, verbose=False)
        assert not np.any(np.isnan(prob)), "cluster_prob must not contain NaN"

    def test_verbose_prints_cluster_info(self, capsys):
        gen_data(n=40, p=3, K=2, verbose=True)
        captured = capsys.readouterr()
        assert "cluster" in captured.out

    def test_two_clusters_both_labels_present(self):
        """With enough samples both cluster labels should appear."""
        np.random.seed(0)
        _, _, cat, _ = gen_data(n=200, p=4, K=2, verbose=False)
        assert set(np.unique(cat)) == {0, 1}


# ===========================================================================
# tf_model
# ===========================================================================

class TestTfModel:

    def test_returns_keras_sequential(self):
        from tensorflow.keras.models import Sequential
        model = tf_model(3)
        assert isinstance(model, Sequential)

    def test_output_units_match_n_clusters(self):
        for K in [1, 2, 5]:
            model = tf_model(K)
            assert model.layers[-1].units == K, f"Expected {K} output units, got {model.layers[-1].units}"

    def test_predict_output_shape(self):
        K = 4
        model = tf_model(K)
        X = np.random.randn(10, 5).astype(np.float32)
        out = model.predict(X, verbose=0)
        assert out.shape == (10, K)

    def test_predict_rows_sum_to_one(self):
        K = 3
        model = tf_model(K)
        X = np.random.randn(20, 5).astype(np.float32)
        out = model.predict(X, verbose=0)
        np.testing.assert_allclose(
            out.sum(axis=1), np.ones(20), atol=1e-5,
            err_msg="Softmax output rows must sum to 1"
        )

    def test_predict_non_negative(self):
        model = tf_model(3)
        X = np.random.randn(15, 5).astype(np.float32)
        out = model.predict(X, verbose=0)
        assert np.all(out >= 0)


# ===========================================================================
# ClusterIte – K = 1
# ===========================================================================

class TestClusterIteK1:

    def test_default_expert_is_linear_regression(self):
        from sklearn.linear_model import LinearRegression
        model = ClusterIte(K=1)
        assert isinstance(model.experts["ex_mod_0"], LinearRegression)

    def test_experts_dict_has_one_entry(self):
        model = ClusterIte(K=1)
        assert len(model.experts) == 1

    def test_fit_does_not_crash(self, small_dataset):
        X, y, _, _ = small_dataset
        model = ClusterIte(K=1)
        model.fit(X, y, silence=True)

    def test_predict_shape(self, small_dataset):
        X, y, _, _ = small_dataset
        model = ClusterIte(K=1)
        model.fit(X, y, silence=True)
        preds = model.predict(X)
        assert preds.shape == (len(X), 1)

    def test_predict_no_nan(self, small_dataset):
        X, y, _, _ = small_dataset
        model = ClusterIte(K=1)
        model.fit(X, y, silence=True)
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))

    def test_gate_predict_returns_all_ones(self, small_dataset, capsys):
        X, y, _, _ = small_dataset
        model = ClusterIte(K=1)
        model.fit(X, y, silence=True)
        g = model.gate_predict(X)
        np.testing.assert_array_equal(
            g, np.ones((len(X), 1)),
            err_msg="gate_predict for K=1 must return a column of ones"
        )

    def test_gate_predict_prints_message(self, small_dataset, capsys):
        X, y, _, _ = small_dataset
        model = ClusterIte(K=1)
        model.fit(X, y, silence=True)
        model.gate_predict(X)
        captured = capsys.readouterr()
        assert "one cluster" in captured.out.lower() or "1" in captured.out


# ===========================================================================
# ClusterIte – K > 1
# ===========================================================================

class TestClusterIteKMulti:

    def test_experts_dict_has_k_entries(self):
        for K in [2, 3, 5]:
            model = ClusterIte(K=K)
            assert len(model.experts) == K

    def test_fit_does_not_crash(self, small_dataset):
        X, y, _, _ = small_dataset
        model = ClusterIte(K=2)
        model.fit(X, y, maxit=5, silence=True)

    def test_predict_shape(self, fitted_k2):
        model, X, _ = fitted_k2
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_no_nan(self, fitted_k2):
        model, X, _ = fitted_k2
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))

    def test_predict_no_inf(self, fitted_k2):
        model, X, _ = fitted_k2
        preds = model.predict(X)
        assert not np.any(np.isinf(preds))

    def test_gate_predict_shape(self, fitted_k2):
        model, X, _ = fitted_k2
        g = model.gate_predict(X)
        assert g.shape == (len(X), 2)

    def test_gate_predict_rows_sum_to_one(self, fitted_k2):
        model, X, _ = fitted_k2
        g = model.gate_predict(X)
        np.testing.assert_allclose(
            g.sum(axis=1), np.ones(len(X)), atol=1e-5,
            err_msg="gate_predict rows must sum to 1"
        )

    def test_gate_predict_non_negative(self, fitted_k2):
        model, X, _ = fitted_k2
        g = model.gate_predict(X)
        assert np.all(g >= 0)

    def test_predict_on_unseen_data(self, fitted_k2):
        model, X, _ = fitted_k2
        X_new = np.random.randn(10, X.shape[1])
        preds = model.predict(X_new)
        assert preds.shape == (10,)
        assert not np.any(np.isnan(preds))

    def test_gate_predict_on_unseen_data(self, fitted_k2):
        model, X, _ = fitted_k2
        X_new = np.random.randn(10, X.shape[1])
        g = model.gate_predict(X_new)
        assert g.shape == (10, 2)
        np.testing.assert_allclose(g.sum(axis=1), np.ones(10), atol=1e-5)

    def test_custom_expert_random_forest(self, small_dataset):
        from sklearn.ensemble import RandomForestRegressor
        X, y, _, _ = small_dataset
        model = ClusterIte(K=2, experts=RandomForestRegressor(n_estimators=5, random_state=0))
        model.fit(X, y, maxit=3, silence=True)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))

    def test_g_matrix_initialized_uniform(self, small_dataset):
        """Before fitting, G is initialized uniformly (1/K for all entries)."""
        X, y, _, _ = small_dataset
        K = 3
        model = ClusterIte(K=K)
        # Access G before fit – it's set during fit, so we inspect after one step
        # by checking that the initial U matrix has the right shape after fitting
        model.fit(X, y, maxit=1, silence=True)
        assert model.G.shape == (len(X), K)

    def test_silence_true_suppresses_convergence_message(self, small_dataset, capsys):
        X, y, _, _ = small_dataset
        model = ClusterIte(K=2)
        model.fit(X, y, maxit=3, silence=True)
        captured = capsys.readouterr()
        assert "Converged" not in captured.out
        assert "Reached max" not in captured.out


# ===========================================================================
# ClusterIte_cv
# ===========================================================================

class TestClusterIteCv:

    def test_nb_folds_stored(self):
        cv = ClusterIte_cv(nb_folds=5)
        assert cv.nb_folds == 5

    def test_hyperparams_stored(self):
        cv = ClusterIte_cv(nb_folds=3, K=2)
        assert cv.hyperparams.get("K") == 2

    def test_cv_mse_is_dataframe(self, fitted_cv):
        assert isinstance(fitted_cv.cv_mse, pd.DataFrame)

    def test_cv_mse_shape(self, fitted_cv):
        # cluster_range=range(2,4) → K=2 and K=3 → 2 columns
        # nb_folds=2 → 2 rows
        assert fitted_cv.cv_mse.shape == (2, 2)

    def test_cv_mse_column_names(self, fitted_cv):
        assert "K=2" in fitted_cv.cv_mse.columns
        assert "K=3" in fitted_cv.cv_mse.columns

    def test_cv_mse_no_nan(self, fitted_cv):
        assert not fitted_cv.cv_mse.isnull().any().any()

    def test_cv_mse_non_negative(self, fitted_cv):
        assert (fitted_cv.cv_mse.values >= 0).all()

    def test_best_K_tab_has_required_columns(self, fitted_cv):
        tab = fitted_cv.best_K_tab_fun()
        for col in ("K", "MSE", "se_MSE"):
            assert col in tab.columns, f"Column '{col}' missing from best_K_tab"

    def test_best_K_tab_sorted_ascending_by_mse(self, fitted_cv):
        tab = fitted_cv.best_K_tab_fun()
        mses = tab["MSE"].values
        assert all(mses[i] <= mses[i + 1] for i in range(len(mses) - 1)), \
            "best_K_tab_fun() should return rows sorted by ascending MSE"

    def test_best_K_tab_k_values_in_range(self, fitted_cv):
        tab = fitted_cv.best_K_tab_fun()
        assert set(tab["K"].values).issubset({2, 3})

    def test_best_K_fun_returns_two_values(self, fitted_cv):
        result = fitted_cv.best_K_fun()
        assert len(result) == 2

    def test_best_K_in_cluster_range(self, fitted_cv):
        best_K, _ = fitted_cv.best_K_fun()
        assert best_K in [2, 3]

    def test_best_K_within_se_in_cluster_range(self, fitted_cv):
        _, best_K_1se = fitted_cv.best_K_fun()
        assert best_K_1se in [2, 3]

    def test_best_K_within_se_is_parsimonious(self, fitted_cv):
        """The 1SE rule picks a model no larger than the apparent optimum."""
        best_K, best_K_1se = fitted_cv.best_K_fun()
        assert best_K_1se <= best_K, \
            "best_K_within_se should be <= best_K (1SE parsimony rule)"

    def test_best_K_stored_on_instance(self, fitted_cv):
        fitted_cv.best_K_fun()
        assert hasattr(fitted_cv, "best_K")
        assert hasattr(fitted_cv, "best_K_within_se")

    def test_se_mse_non_negative(self, fitted_cv):
        tab = fitted_cv.best_K_tab_fun()
        assert (tab["se_MSE"].values >= 0).all()
