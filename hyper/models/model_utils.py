import torch
import numpy as np

try:
    import psutil
except:
    print("import psutil failed!")

initial_cpu_memory = None
initial_gpu_memory = None
def report_memory():
    global initial_cpu_memory
    global initial_gpu_memory
    if initial_cpu_memory is None:
        initial_cpu_memory = psutil.Process().memory_info().rss / 1024 ** 2
    if initial_gpu_memory is None:
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2
    print("GPU memory_allocated:", round((torch.cuda.memory_allocated()  / 1024 ** 2) - initial_gpu_memory, 2), "[MB]")
    print("CPU rss memory (current - initial):      ", round((psutil.Process().memory_info().rss / 1024 ** 2) - initial_cpu_memory, 2), "[MB]")

    pass

def reporting_hook():
    # called when printing during the printed_forward run ...
    #(opt)report_memory()
    pass

def num_of_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
