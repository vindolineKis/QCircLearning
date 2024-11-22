import pytest
from scipy.optimize import OptimizeResult
from QCircLearning.optimizer import Optimizer

# FILE: QCircLearning/test_optimizer.py


def quadratic_function(x):
    return (x - 3) ** 2

def test_optimize_with_record_path():
    optimizer = Optimizer(method="BFGS")
    x0 = [0]
    result = optimizer.optimize(quadratic_function, x0, record_path=True)
    
    assert isinstance(result, OptimizeResult)
    assert result.success
    assert pytest.approx(result.x, abs=1e-3) == [3]
    assert optimizer.get_path_x is not None
    assert optimizer.get_path_y is not None
    assert len(optimizer.get_path_x) > 0
    assert len(optimizer.get_path_y) > 0

def test_optimize_without_record_path():
    optimizer = Optimizer(method="BFGS")
    x0 = [0]
    result = optimizer.optimize(quadratic_function, x0, record_path=False)
    
    assert isinstance(result, OptimizeResult)
    assert result.success
    assert pytest.approx(result.x, abs=1e-3) == [3]
    assert optimizer.get_path_x is None
    assert optimizer.get_path_y is None

if __name__ == "__main__":
    pytest.main()