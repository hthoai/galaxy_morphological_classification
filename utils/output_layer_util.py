import torch 

def slice_input_by_question(input: torch.Tensor):
    #TODO subcheck size 37
    return [
        input[:,0:3],
        input[:,3:5],
        input[:,5:7],
        input[:,7:9],
        input[:,9:13],
        input[:,13:15],
        input[:,15:18],
        input[:,18:25],
        input[:,25:28],
        input[:,28:31],
        input[:,31:37],
    ]