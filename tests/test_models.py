# tests/test_models.py
import numpy as np
from cbp.models.baseline import ZeroChange, MeanModel, SimpleOLS

def test_zero_change_predicts_zero():
    m = ZeroChange(); m.fit(np.array([[1.0],[2.0]]), np.array([5.0, 6.0]))
    assert (m.predict(np.array([[9.0]])) == 0.0).all()

def test_mean_model_predicts_train_mean():
    m = MeanModel(); m.fit(np.zeros((3, 1)), np.array([2.0, 4.0, 6.0]))
    assert m.predict(np.zeros((1, 1)))[0] == 4.0

def test_ols_recovers_linear():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    y = 3.0 * x[:, 0] + 0.5
    m = SimpleOLS(); m.fit(x, y)
    pred = m.predict(np.array([[1.0]]))[0]
    assert abs(pred - 3.5) < 1e-6
