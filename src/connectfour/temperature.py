from dataclasses import dataclass, field

@dataclass
class TemperatureSchedule:
    """defines the evolution of temperature parameter tau at each round of the game
    as a step function.
    The temperature helps to control the level of exploration-exploitation tradeoff 
    by either smoothing the move distribution obtained after MCTS (tau > 1) or increasing 
    on the contrary thechances to select the best candidate identified for the next 
    step (tau << 1).
    
    Note: tau = 1 means that the distribution will be proporational to the number of 
    node visits.
    
    
    Attributes:
        tau_start (float, optional): the value for tau for the first steps
        threshold (int, optional): the number of steps until tau switches from 
            tau_start to tau_end.
        tau_end (float, optional): the value for tau after step `threshold`
        counter (int): 
        
    Example:
    >> tau = TemperatureSchedule(1, 2, .1)
    >> print([next(tau) for _ in range(5)])
        # [1, 1, 0.1, 0.1, 0.1]
    """
    tau_start: float = 1
    threshold: int = 10
    tau_end: float = 0.1
    
    def __post_init__(self):
        self.counter: int = 0 # field(default=0, init=False, repr=False)
        
    def __next__(self):
        value = self.tau_start if self.counter < self.threshold else self.tau_end
        self.counter += 1
        return value
    
    def __str__(self) -> str:
        return f"{self.tau_start}-{self.threshold}-{self.tau_end}".replace(".","")