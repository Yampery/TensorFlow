"""打印模型内容
"""
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

savedir = 'log/'
print_tensors_in_checkpoint_file(savedir + 'linermodel.cpkt', None, True)

"""打印内容
tensor_name:  bias
[0.00747073]
tensor_name:  weight
[1.9772009]
"""