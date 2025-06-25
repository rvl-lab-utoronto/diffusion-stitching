import numpy as np
import torch
class ToyEnvInvDyn():
    """ To unify decision-making with the algorithms
    using impainting under a single banner (so to speak)"""
    def __init__(self):
        pass
    def predict(self,current_states, future_states):
        return (future_states - current_states)