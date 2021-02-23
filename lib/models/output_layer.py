import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.output_layer_util import slice_input_by_question

class OutputLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(OutputLayer, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        probabilities = self.calc_output_layer(input)
        processed_probabilities = self.weighted_output_layer(probabilities)
        return processed_probabilities

    @staticmethod
    def calc_output_layer(input: torch.Tensor) -> torch.Tensor:
        #TODO: subcheck here
        questions = slice_input_by_question(input)
        # + eps for softmax?
        probabilities = [F.softmax(question, dim=1) for question in questions]
        return probabilities

    @staticmethod
    def weighted_output_layer(questions: list):
        w1 = 1
        w2 = questions[0][:, 1] # question 1 answer 2
        w3 = questions[1][:, 1] * w2 # question 2 answer 2 * w2
        w4 = w3
        w7 = questions[0][:, 0] # question 1 answer 1
        w9 = questions[1][:, 0] * w2 # question 2 answer 1 * w2
        w10 = questions[3][:, 0] * w4 # question 4 answer 1 * w4
        w11 = w10
        w5 = w4
        w6 = 1
        w8 = questions[5][:, 0] * w6

        # weighted answers
        #TODO: Refactor Dump Code
        questions[1] = questions[1] * torch.transpose(w2.repeat(2, 1), 0, 1)
        questions[2] = questions[2] * torch.transpose(w3.repeat(2, 1), 0, 1)
        questions[3] = questions[3] * torch.transpose(w3.repeat(2, 1), 0, 1)
        questions[4] = questions[4] * torch.transpose(w3.repeat(4, 1), 0, 1)
        questions[5] = questions[5] * torch.transpose(w3.repeat(2, 1), 0, 1)
        questions[6] = questions[6] * torch.transpose(w3.repeat(3, 1), 0, 1)
        questions[7] = questions[7] * torch.transpose(w3.repeat(7, 1), 0, 1)
        questions[8] = questions[8] * torch.transpose(w3.repeat(3, 1), 0, 1)
        questions[9] = questions[9] * torch.transpose(w3.repeat(3, 1), 0, 1)
        questions[10] = questions[10] * torch.transpose(w3.repeat(6, 1), 0, 1)

        return torch.cat(questions, dim=1)



