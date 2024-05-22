import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.CReluToTLogic import *

CReLUNetwork = CReLUNetwork(
    [torch.tensor([[2.0, -0.5],[2, -0.5]], dtype=torch.float64), torch.tensor([[2., 2.], [2., 2.], [2., 2.], [2., 2.]], dtype=torch.float64), torch.tensor([[-1., -1., -1., -1.]], dtype=torch.float64)],
    [torch.tensor([0.0, -1], dtype=torch.float64), torch.tensor([0.0, -1, -2, -3], dtype=torch.float64),  torch.tensor([1], dtype=torch.float64)]
)

construct_MV_terms(CReLUNetwork)