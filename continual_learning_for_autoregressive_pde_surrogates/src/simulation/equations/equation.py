from abc import ABC, abstractmethod


class Equation(ABC):
    @abstractmethod
    def solve(self):
        ...
