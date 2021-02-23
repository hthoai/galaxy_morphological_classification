from torch import Tensor
from torch.nn import Module

"""
Customized maxout activation (function) layer. 
When __init__, pass in step size - how many numbers to take maximum at a time.  
"""

class MaxOut(Module):
    def __init__(self, step_size):
        super(MaxOut, self).__init__()
        self.step_size = step_size

    def forward(self, input: Tensor) -> Tensor: 
        output = []
        if input.size()[1] % self.step_size != 0:
            raise ValueError('Step size does not divide input size')
        for i in range(0, input.size()[1], self.step_size):
            output.append(max(input[:,i + self.step_size-1]))
        return Tensor(output)
        

