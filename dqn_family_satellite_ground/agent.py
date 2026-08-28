"""Independent pure-DQN implementation of standard DQN and DQN-family implementation.

The variants share the same network, uniform replay, one-step transitions,
epsilon-greedy exploration, optimizer, and standard DQN target. centered full-action adds the
training-only centered counterfactual advantage objective.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn


VARIANTS = (
    "standard_dqn",
    "centered_full_action_dqn",
    "full_action_q_dqn",
    "immediate_advantage_dqn",
    "double_dqn",
    "double_centered_full_action_dqn",
    "contextual_bandit",
)


class QNetwork(nn.Module):
    """The fixed 24-128-128-64-3 network used by every learned policy."""

    def __init__(self, state_dim: int = 24, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class AgentConfig:
    gamma: float = 0.97
    lr: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 100_000
    target_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 13_440
    auxiliary_lambda: float = 0.3


class ReplayBuffer:
    """Uniform one-step replay; no priority, sequences, or multi-step returns."""

    def __init__(self, capacity: int):
        self.data: deque = deque(maxlen=capacity)

    def add(self, transition: tuple) -> None:
        self.data.append(transition)

    def sample(self, size: int, rng: random.Random):
        batch = rng.sample(self.data, size)
        return tuple(np.asarray(values) for values in zip(*batch))

    def __len__(self) -> int:
        return len(self.data)


class DQNAgent:
    """Pure DQN with an optional centered full-action auxiliary objective."""

    def __init__(
        self,
        variant: str,
        config: AgentConfig,
        device: torch.device,
        seed: int,
        state_dim: int = 24,
        action_dim: int = 3,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        if action_dim != 3:
            raise ValueError("the scheduling task requires exactly three actions")
        if self._variant_uses_counterfactuals(variant) and config.auxiliary_lambda <= 0:
            raise ValueError("auxiliary_lambda must be positive for model supervision")

        self.variant = variant
        self.config = config
        self.device = device
        self.action_dim = action_dim
        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.online = QNetwork(state_dim, action_dim).to(device)
        self.target = copy.deepcopy(self.online).to(device).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.steps = 0
        self.epsilon = config.epsilon_start

    @property
    def uses_counterfactuals(self) -> bool:
        return self._variant_uses_counterfactuals(self.variant)

    @staticmethod
    def _variant_uses_counterfactuals(variant: str) -> bool:
        return variant in {
            "centered_full_action_dqn",
            "full_action_q_dqn",
            "immediate_advantage_dqn",
            "double_centered_full_action_dqn",
        }

    @property
    def uses_double_target(self) -> bool:
        return self.variant in {"double_dqn", "double_centered_full_action_dqn"}

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_dim)
        with torch.no_grad():
            value = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return int(self.online(value).argmax(dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        preview_costs: np.ndarray | None = None,
        preview_resources: np.ndarray | None = None,
    ) -> None:
        if preview_costs is None:
            preview_costs = np.zeros(3, dtype=np.float32)
        if preview_resources is None:
            preview_resources = np.zeros((3, 6), dtype=np.float32)
        self.buffer.add(
            (
                state,
                action,
                reward,
                next_state,
                float(done),
                preview_costs,
                preview_resources,
            )
        )
        self.steps += 1
        fraction = min(1.0, self.steps / self.config.epsilon_decay_steps)
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start
            + fraction * (self.config.epsilon_end - self.config.epsilon_start),
        )

    def counterfactual_targets(
        self,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        preview_costs: torch.Tensor,
        preview_resources: torch.Tensor,
    ) -> torch.Tensor:
        """Standard-DQN Bellman targets for every modeled action consequence."""
        batch_size, state_dim = next_states.shape
        counterfactual_next = next_states[:, None, :].expand(
            batch_size, self.action_dim, state_dim
        ).clone()
        # The next task is exogenous and action independent. Only the six
        # endogenous resource coordinates are replaced by the action preview.
        counterfactual_next[:, :, -6:] = preview_resources
        with torch.no_grad():
            # Standard variants use target-network maximization. Double
            # variants use online selection and target-network evaluation.
            flat_next = counterfactual_next.reshape(-1, state_dim)
            if self.uses_double_target:
                selected = self.online(flat_next).argmax(dim=1, keepdim=True)
                next_value = self.target(flat_next).gather(1, selected).squeeze(1)
            else:
                next_value = self.target(flat_next).max(dim=1).values
            next_value = next_value.reshape(batch_size, self.action_dim)
            return -preview_costs + self.config.gamma * (
                1.0 - dones[:, None]
            ) * next_value

    @staticmethod
    def centered(values: torch.Tensor) -> torch.Tensor:
        return values - values.mean(dim=1, keepdim=True)

    def compute_losses(self, batch: tuple[np.ndarray, ...]) -> dict[str, torch.Tensor]:
        states, actions, rewards, next_states, dones, costs, resources = batch
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online(states_t)
        chosen_q = q_values.gather(1, actions_t[:, None]).squeeze(1)
        with torch.no_grad():
            if self.uses_double_target:
                selected = self.online(next_states_t).argmax(dim=1, keepdim=True)
                next_value = self.target(next_states_t).gather(1, selected).squeeze(1)
            else:
                next_value = self.target(next_states_t).max(dim=1).values
            factual_target = rewards_t + self.config.gamma * (1.0 - dones_t) * next_value
        td_loss = nn.functional.smooth_l1_loss(chosen_q, factual_target)
        auxiliary_loss = torch.zeros((), dtype=torch.float32, device=self.device)

        if self.uses_counterfactuals:
            costs_t = torch.as_tensor(costs, dtype=torch.float32, device=self.device)
            resources_t = torch.as_tensor(
                resources, dtype=torch.float32, device=self.device
            )
            if self.variant == "immediate_advantage_dqn":
                targets = -costs_t
            else:
                targets = self.counterfactual_targets(
                    next_states_t, dones_t, costs_t, resources_t
                )
            if self.variant == "full_action_q_dqn":
                auxiliary_loss = nn.functional.smooth_l1_loss(q_values, targets)
            else:
                auxiliary_loss = nn.functional.smooth_l1_loss(
                    self.centered(q_values), self.centered(targets)
                )

        total = td_loss
        if self.uses_counterfactuals:
            total = total + self.config.auxiliary_lambda * auxiliary_loss
        return {"loss": total, "td_loss": td_loss, "auxiliary_loss": auxiliary_loss}

    def update(self) -> dict[str, float] | None:
        if len(self.buffer) < self.config.batch_size:
            return None
        losses = self.compute_losses(
            self.buffer.sample(self.config.batch_size, self.rng)
        )
        self.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        if self.steps % self.config.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        result = {name: float(value.item()) for name, value in losses.items()}
        if not np.isfinite(list(result.values())).all():
            raise FloatingPointError(f"non-finite update: {result}")
        return result

    def checkpoint_config(self) -> dict:
        return asdict(self.config)
