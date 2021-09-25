from abc import ABC, abstractmethod

class BaseTemplate(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def represent(self, data):
        pass