"""MLP regressor trained with PyTorch full-batch L-BFGS (float64), numpy inference.

Same model as sklearn's MLPRegressor used in Part A2 (tanh hidden layers, linear output,
Glorot-uniform initialisation, loss = 0.5*mean squared error + 0.5*alpha*||W||^2 / n),
but trained far faster, so the L-BFGS iteration budget can be raised from 3,000 (where
every A2/A2b fit stopped unconverged) to tens of thousands.

* Training needs torch; prediction uses only numpy (weights stored as arrays), so saved
  models load and predict without torch.
* Reproducible for a fixed seed and thread count (both stored in the object).
"""
from __future__ import annotations

import time

import numpy as np


class TorchMLPRegressor:
    def __init__(self, hidden=(128, 128, 128), alpha=1e-5, max_iter=20000, seed=0, threads=4,
                 history_size=50, tol_grad=1e-12, tol_change=1e-16, log_every=0):
        self.hidden = tuple(hidden)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.threads = int(threads)
        self.history_size = int(history_size)
        self.tol_grad, self.tol_change = tol_grad, tol_change
        self.log_every = log_every
        self.coefs_: list[np.ndarray] = []
        self.intercepts_: list[np.ndarray] = []
        self.n_iter_ = 0
        self.loss_ = float("nan")
        self.loss_curve_: list[tuple[int, float]] = []
        self.fit_seconds_ = 0.0

    # ------------------------------------------------------------------ training
    def fit(self, X, Y):
        import torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float)
        Y = np.asarray(Y, float).reshape(len(X), -1)
        sizes = [X.shape[1], *self.hidden, Y.shape[1]]
        params = []
        for fi, fo in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6.0 / (fi + fo))                      # Glorot uniform (sklearn, tanh)
            W = torch.tensor(rng.uniform(-bound, bound, (fi, fo)), dtype=torch.float64, requires_grad=True)
            b = torch.tensor(rng.uniform(-bound, bound, fo), dtype=torch.float64, requires_grad=True)
            params += [W, b]
        Xt = torch.tensor(X, dtype=torch.float64)
        Yt = torch.tensor(Y, dtype=torch.float64)
        n = len(X)
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=self.max_iter, max_eval=int(self.max_iter * 1.25),
                                tolerance_grad=self.tol_grad, tolerance_change=self.tol_change,
                                history_size=self.history_size, line_search_fn="strong_wolfe")
        state = {"evals": 0, "loss": None}
        t0 = time.perf_counter()

        def forward(inp):
            h = inp
            for k in range(0, len(params) - 2, 2):
                h = torch.tanh(h @ params[k] + params[k + 1])
            return h @ params[-2] + params[-1]

        def closure():
            opt.zero_grad()
            out = forward(Xt)
            reg = sum((params[k] ** 2).sum() for k in range(0, len(params), 2))
            # sklearn MLPRegressor: squared_loss = mean over all entries / 2, + 0.5*alpha*||W||^2 / n
            loss = 0.5 * ((out - Yt) ** 2).mean() + 0.5 * self.alpha * reg / n
            loss.backward()
            state["evals"] += 1
            state["loss"] = loss.item()
            if self.log_every and state["evals"] % self.log_every == 0:
                self.loss_curve_.append((state["evals"], state["loss"]))
            return loss

        opt.step(closure)
        self.fit_seconds_ = time.perf_counter() - t0
        self.n_iter_ = int(opt.state[opt._params[0]].get("n_iter", 0))
        self.loss_ = state["loss"]
        self.coefs_ = [params[k].detach().numpy().copy() for k in range(0, len(params), 2)]
        self.intercepts_ = [params[k].detach().numpy().copy() for k in range(1, len(params), 2)]
        return self

    # ------------------------------------------------------------------ inference (numpy only)
    def predict(self, X):
        h = np.asarray(X, float)
        for W, b in zip(self.coefs_[:-1], self.intercepts_[:-1]):
            h = np.tanh(h @ W + b)
        return h @ self.coefs_[-1] + self.intercepts_[-1]
