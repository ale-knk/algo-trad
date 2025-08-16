from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


class SumTree:
    """
    A sum tree data structure used for prioritized experience replay.
    Stores priorities and allows O(log n) sampling based on priorities.
    """

    def __init__(self, capacity: int):
        """
        Initialize the SumTree with a given capacity.

        Args:
            capacity: The maximum number of elements the tree can hold
        """
        # Number of leaf nodes (transitions)
        self.capacity = capacity
        # Tree structure to calculate the sum: 2*capacity - 1 nodes in total
        self.tree = np.zeros(2 * capacity - 1)
        # Data stored in leaves
        self.data = [None] * capacity
        # Current write pointer
        self.write_pointer = 0
        # Current number of elements
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """
        Propagate a change in a leaf node priority up the tree.

        Args:
            idx: The index of the node that was updated
            change: The amount the node was changed by
        """
        # Get parent index
        parent = (idx - 1) // 2
        # Add change to parent node
        self.tree[parent] += change
        # If not root node, propagate up
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Find the leaf index containing the value s within the sum prefix.

        Args:
            idx: The index of the current node
            s: The value to find

        Returns:
            The index of the leaf node containing s
        """
        # If leaf node, return index
        left = 2 * idx + 1
        right = left + 1

        # If leaf node, return index
        if left >= len(self.tree):
            return idx

        # If value is less than left child, search left subtree
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        # Otherwise, search right subtree with value reduced by left sum
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Get the total sum of priorities."""
        return self.tree[0]

    def add(self, priority: float, data) -> None:
        """
        Add a new sample with given priority.

        Args:
            priority: Priority of the sample
            data: The sample data to store
        """
        # Get index in data array
        idx = self.write_pointer + self.capacity - 1
        # Store data
        self.data[self.write_pointer] = data
        # Update tree
        self.update(idx, priority)
        # Move pointer
        self.write_pointer = (self.write_pointer + 1) % self.capacity
        # Update number of entries
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """
        Update the priority of an existing sample.

        Args:
            idx: The tree index of the sample to update
            priority: The new priority
        """
        # Calculate change in priority
        change = priority - self.tree[idx]
        # Update tree value
        self.tree[idx] = priority
        # Propagate change upward
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        """
        Get a sample using a value s in range [0, total).

        Args:
            s: The value in range [0, total)

        Returns:
            Tuple of (idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class ReplayBuffer:
    """
    A replay buffer for training reinforcement learning agents.
    Supports both uniform and prioritized experience replay.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = 100000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
        epsilon: float = 1e-6,
    ):
        """
        Initialize replay buffer.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_size: Maximum number of experiences to store
            device: Device to store tensors on
            use_prioritized: Whether to use prioritized experience replay
            alpha: Determines how much prioritization to use (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction factor (0 to 1)
            beta_increment: How much to increment beta at each sampling
            epsilon: Small value to add to priorities to ensure non-zero
        """
        self.max_size = max_size
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Prioritized Experience Replay parameters
        self.use_prioritized = use_prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initial max priority for new transitions

        if use_prioritized:
            # Initialize SumTree for prioritized experience replay
            self.sum_tree = SumTree(max_size)
        else:
            # Use deque with maxlen for automatic FIFO behavior in uniform sampling
            self.states = deque(maxlen=max_size)
            self.actions = deque(maxlen=max_size)
            self.rewards = deque(maxlen=max_size)
            self.next_states = deque(maxlen=max_size)
            self.dones = deque(maxlen=max_size)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        error: Optional[float] = None,
    ):
        """
        Add a new experience to memory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
            error: TD error (for prioritized replay)
        """
        if self.use_prioritized:
            # Calculate priority with max priority for new samples if error is not provided
            if error is None:
                priority = self.max_priority
            else:
                # Priority = (|error| + epsilon)^alpha
                priority = (abs(error) + self.epsilon) ** self.alpha
                # Update max priority
                self.max_priority = max(self.max_priority, priority)

            # Add experience to sum tree with priority
            self.sum_tree.add(priority, (state, action, reward, next_state, done))
        else:
            # Add to regular deques for uniform sampling
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)

    def update_priorities(
        self, indices: List[int], errors: Union[List[float], np.ndarray]
    ):
        """
        Update priorities for prioritized experience replay.

        Args:
            indices: Tree indices of the transitions to update
            errors: TD errors for each transition
        """
        if not self.use_prioritized:
            return

        for idx, error in zip(indices, errors):
            # Calculate new priority
            priority = (abs(error) + self.epsilon) ** self.alpha
            # Update in sum tree
            self.sum_tree.update(idx, priority)
            # Update max priority
            self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences from memory.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            If not prioritized: Tuple of (states, actions, rewards, next_states, dones) as tensors
            If prioritized: Tuple of (states, actions, rewards, next_states, dones, indices, weights) as tensors
        """
        if self.use_prioritized:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)

    def _sample_uniform(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample uniformly from the buffer."""
        # Get total number of experiences stored
        total_size = len(self.states)

        # Sample random indices
        indices = np.random.choice(total_size, batch_size, replace=False)

        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array([self.states[i] for i in indices])).to(
            self.device
        )

        actions = torch.FloatTensor(np.array([self.actions[i] for i in indices])).to(
            self.device
        )

        rewards = (
            torch.FloatTensor(np.array([self.rewards[i] for i in indices]))
            .unsqueeze(1)
            .to(self.device)
        )

        next_states = torch.FloatTensor(
            np.array([self.next_states[i] for i in indices])
        ).to(self.device)

        dones = (
            torch.FloatTensor(np.array([self.dones[i] for i in indices]))
            .unsqueeze(1)
            .to(self.device)
        )

        return states, actions, rewards, next_states, dones

    def _sample_prioritized(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample from the buffer using priorities."""
        # Increase beta value (annealing towards 1)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Initialize arrays for collection
        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        # Calculate the segment size for prioritized sampling
        segment = self.sum_tree.total() / batch_size

        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            # Get sample from SumTree
            idx, priority, data = self.sum_tree.get(s)

            indices[i] = idx
            priorities[i] = priority
            batch.append(data)

        # Calculate importance sampling weights
        # P(i) = priority_i / sum(priorities)
        # weights = (N * P(i))^(-beta)
        # Normalize weights to scale updates
        sampling_probabilities = priorities / self.sum_tree.total()
        weights = (self.sum_tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Extract batch components
        states = np.vstack([b[0] for b in batch])
        actions = np.vstack([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        # Convert to PyTorch tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        indices_tensor = torch.LongTensor(indices).to(self.device)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            indices_tensor,
            weights_tensor,
        )

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        if self.use_prioritized:
            return self.sum_tree.n_entries
        else:
            return len(self.states)
