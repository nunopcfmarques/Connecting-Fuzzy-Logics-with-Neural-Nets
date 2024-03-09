from src.ReLUNetwork import *

def test(x):
    return x + 1

connective_to_relu = {
    "⊙": ReLUNetwork(
        [np.matrix([1., 1.], dtype=np.float32), np.matrix([1.], dtype=np.float32)],
        [np.array([-1.], dtype=np.float32), np.array([0.], dtype=np.float32)]
    ),
    "¬": ReLUNetwork(
        [np.matrix([-1.], dtype=np.float32)],
        [np.array([1.], dtype=np.float32)]
    ),
    "⊕": ReLUNetwork(
        [np.matrix([-1., -1.], dtype=np.float32), np.matrix([-1.], dtype=np.float32)],
        [np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32)]
    ),
}