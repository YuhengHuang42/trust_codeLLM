"""Activation Store Base Class."""
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import final

from jaxtyping import Float
import torch
from torch import Tensor
from torch.utils.data import Dataset

from method.extract.tensor_types import Axis


class ActivationStore(
    Dataset[Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]], ABC
):
    """Activation Store Abstract Class.

    Extends the `torch.utils.data.Dataset` class to provide an activation store, with additional
    :meth:`append` and :meth:`extend` methods (the latter of which should typically be
    non-blocking). The resulting activation store can be used with a `torch.utils.data.DataLoader`
    to iterate over the dataset.

    Extend this class if you want to create a new activation store (noting you also need to create
    `__getitem__` and `__len__` methods from the underlying `torch.utils.data.Dataset` class).

    Example:
    >>> import torch
    >>> class MyActivationStore(ActivationStore):
    ...
    ...     @property
    ...     def current_activations_stored_per_component(self):
    ...        raise NotImplementedError
    ...
    ...     @property
    ...     def num_components(self):
    ...         raise NotImplementedError
    ...
    ...     def __init__(self):
    ...         super().__init__()
    ...         self._data = [] # In this example, we just store in a list
    ...
    ...     def append(self, item) -> None:
    ...         self._data.append(item)
    ...
    ...     def extend(self, batch):
    ...         self._data.extend(batch)
    ...
    ...     def empty(self):
    ...         self._data = []
    ...
    ...     def __getitem__(self, index: int):
    ...         return self._data[index]
    ...
    ...     def __len__(self) -> int:
    ...         return len(self._data)
    ...
    >>> store = MyActivationStore()
    >>> store.append(torch.randn(100))
    >>> print(len(store))
    1
    """

    @abstractmethod
    def append(
        self,
        item: Float[Tensor, Axis.names(Axis.INPUT_OUTPUT_FEATURE)],
        component_idx: int = 0,
    ) -> Future | None:
        """Add a Single Item to the Store."""

    @abstractmethod
    def extend(
        self,
        batch: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)],
        component_idx: int = 0,
    ) -> Future | None:
        """Add a Batch to the Store."""

    @abstractmethod
    def empty(self) -> None:
        """Empty the Store."""

    @property
    @abstractmethod
    def num_components(self) -> int:
        """Number of components."""

    @property
    @abstractmethod
    def current_activations_stored_per_component(self) -> list[int]:
        """Current activations stored per component."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the Length of the Store."""

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]:
        """Get an Item from the Store."""

    def shuffle(self) -> None:
        """Optional shuffle method."""

    @final
    def fill_with_test_data(
        self,
        num_batches: int = 16,
        batch_size: int = 16,
        num_components: int = 1,
        input_features: int = 256,
    ) -> None:
        """Fill the store with test data.

        For use when testing your code, to ensure it works with a real activation store.

        Warning:
            You may want to use `torch.seed(0)` to make the random data deterministic, if your test
            requires inspecting the data itself.

        Example:
            >>> from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
            >>> store = TensorActivationStore(max_items=16*16, num_neurons=256)
            >>> store.fill_with_test_data()
            >>> len(store)
            256
            >>> store[0].shape
            torch.Size([256])

        Args:
            num_batches: Number of batches to fill the store with.
            batch_size: Number of items per batch.
            num_components: Number of source model components the SAE is trained on.
            input_features: Number of input features per item.
        """
        for _ in range(num_batches):
            for component_idx in range(num_components):
                sample = torch.rand(batch_size, input_features)
                self.extend(sample, component_idx)


class StoreFullError(IndexError):
    """Exception raised when the activation store is full."""

    def __init__(self, message: str = "Activation store is full"):
        """Initialise the exception.

        Args:
            message: Override the default message.
        """
        super().__init__(message)