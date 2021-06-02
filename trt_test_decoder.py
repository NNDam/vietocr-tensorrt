import numpy as np 
from exec_backends.trt_loader import TrtOCRDecoder
import onnxruntime as rt
import torch
from torch.nn.functional import log_softmax, softmax
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from torch import nn

class TorchDecoder(nn.Module):
    def __init__(self, model):
        
        super(TorchDecoder, self).__init__()
        
        self.model = model

    def forward(self, tgt_inp, memory):
        output, _ = self.model.transformer.forward_decoder(tgt_inp, memory)
        output = softmax(output, dim=-1)
        values, indices  = torch.topk(output, 5)
        return values, indices

config = Cfg.load_config_from_name('vgg_transformer')
dataset_params = {
    'name':'hw',
    'data_root':'./my_data/',
    'train_annotation':'train_line_annotation.txt',
    'valid_annotation':'test_line_annotation.txt'
}

params = {
         'print_every':200,
         'valid_every':15*200,
          'iters':20000,
          'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',    
          'export':'./weights/transformerocr.pth',
          'metrics': 10000
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cpu'

trainer = Predictor(config)

# tgt_inp = torch.randint(0, 232, (20, 1))
tgt_inp = np.random.randint(0, 232, (20, 3))
memory = np.random.randn(170, 3, 256)
# memory = torch.randn(170, 1, 256, requires_grad=True)

trt_model = TrtOCRDecoder('transformer_decoder.trt')
torch_model = TorchDecoder(trainer.model)
torch_model.eval()
onnx_model = rt.InferenceSession('transformer_decoder.onnx')

trt_values, trt_indices = trt_model.run(tgt_inp, memory)

with torch.no_grad():
    torch_values, torch_indices = torch_model(torch.LongTensor(tgt_inp.copy()), torch.Tensor(memory.copy()))
    torch_values = torch_values.detach().cpu().numpy()
    torch_indices = torch_indices.detach().cpu().numpy()

onnx_inp = {onnx_model.get_inputs()[0].name: tgt_inp.copy(), onnx_model.get_inputs()[1].name: memory.copy().astype('float32')} 
onnx_values, onnx_indices = onnx_model.run(None, onnx_inp)

print(np.squeeze(torch_values))
print(np.squeeze(onnx_values))
print(np.squeeze(trt_values))

print(np.squeeze(torch_indices))
print(np.squeeze(onnx_indices))
print(np.squeeze(trt_indices))