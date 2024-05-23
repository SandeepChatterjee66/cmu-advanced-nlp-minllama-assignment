from typing import Callable, Dict, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        """
        Initialize the AdamW optimizer. AdamW is a variant of the Adam optimizer that decouples weight decay from this paper: https://arxiv.org/abs/1711.05101

        Args:
            params (Iterable[torch.nn.parameter.Parameter]): Iterable of model parameters.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Defaults to (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-6.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.0.
            correct_bias (bool, optional): Flag to apply bias correction. Defaults to True.

        Raises:

            ValueError: If the learning rate is less than 0.0.
            ValueError: If the beta parameter is not in the range [0.0, 1.0]
            ValueError: If the epsilon value is less than 0.0.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if 0.0 > eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Define a single optimization step for Adam

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss. Defaults to None.

        Raises:
            RuntimeError: when the gradient is sparse.

        Returns:
            torch.Tensor: The loss value evaluated by the closure function.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                self._update_parameters(p, grad, group)

        return loss

    def _update_parameters(
        self, p: torch.Tensor, grad: torch.Tensor, group: Dict[str, float]
    ) -> None:
        """Helper function to update model parameters.
        References: Adam https://arxiv.org/abs/1412.6980 page 2

        Args:
            p (torch.Tensor): Model parameter, with gradient data.
            grad (torch.Tensor): Gradient data for the model parameter.
            group (Dict[str, float]): Hyperparameters for the optimizer.
        """
        # Get the hyperparameters, step size and exponential decay rates
        alpha = group["lr"]
        beta1, beta2 = group["betas"]

        # State initialization
        if p not in self.state:
            self.state[p] = {}
            self.state[p]["step"] = 0.0
            self.state[p]["exp_avg"] = torch.zeros_like(p.data)
            self.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)
        state = self.state[p]

        # Get the first and second moment vectors
        m, v = state["exp_avg"], state["exp_avg_sq"]

        # Update time step
        state["step"] += 1.0
        t = state["step"]

        # Update the first and second moment vectors
        # m = beta1 * m + (1 - beta1) * grad
        # v = beta2 * v + (1 - beta2) * grad^2
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction for the second moment
        bias_correction1 = 1 - beta1**t
        bias_correction2 = 1 - beta2**t

        v_hat = v / bias_correction2

        # Update the model parameters. Note that we use m instead of m_hat like the paper
        # p = p - stepsize * m / (sqrt(v_hat) + eps)
        denom = v_hat.sqrt().add_(group["eps"])
        step_size = alpha / bias_correction1 if group["correct_bias"] else alpha
        p.data.addcdiv_(m, denom, value=-step_size)

        # Apply weight decay https://arxiv.org/pdf/1711.05101
        if abs(group["weight_decay"]) > 1e-9:
            p.data.add_(p.data, alpha=-step_size * group["weight_decay"])
