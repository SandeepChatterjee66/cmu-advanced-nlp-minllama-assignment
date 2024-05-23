import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.optimizer import AdamW

seed = 0


def test_optimizer(opt_class) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        correct_bias=True,
    )
    for _ in range(1000):
        opt.zero_grad()
        x = torch.FloatTensor(rng.uniform(size=[model.in_features]))
        y_hat = model(x)
        y = torch.Tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


ref = torch.tensor(np.load("tests/optimizer_test.npy"))
actual = test_optimizer(AdamW)

# Test type
assert isinstance(
    actual, torch.Tensor
), f"Expected: torch.Tensor but got {type(actual)}"
assert ref.dtype == actual.dtype, f"Expected: {ref.dtype}, but got {actual.dtype}"

# Test shape
assert ref.shape == actual.shape, f"Expected: {ref.shape}, but got {actual.shape}"

# Test values; change absolute and relative tolerance from default since the test only involves 4 decimal places (see weight decay above)
assert torch.allclose(
    ref, actual, atol=1e-4, rtol=1e-4
), f"Expected: \n {ref} \n, but got \n {actual}"
print("Optimizer test passed!")
