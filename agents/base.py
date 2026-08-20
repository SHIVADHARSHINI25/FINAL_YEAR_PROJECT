from abc import ABC

class BaseAgent(ABC):
    """Abstract base class for all agents in the framework."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
