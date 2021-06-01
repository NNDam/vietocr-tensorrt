import numpy as np 
from exec_backends.trt_loader import TrtOCREncoder
import onnxruntime as rt
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from torch import nn

class TorchEncoder(nn.Module):
    def __init__(self, model):
        
        super(TorchEncoder, self).__init__()
        
        self.model = model

    def forward(self, img):
        """
        Shape:
            - img: (N, C, H, W)
            - output: b t v
        """
        src = self.model.cnn(img)
        # print('CNN out', src.shape)
        memory = self.model.transformer.forward_encoder(src)

        # memories = self.model.transformer.forward_encoder(src)
        return memory

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

sample_inp = np.array(np.random.rand(6, 3, 32, 160), dtype = np.double)

trt_model = TrtOCREncoder('transformer_encoder.trt')
torch_model = TorchEncoder(trainer.model)
torch_model.eval()
onnx_model = rt.InferenceSession('transformer_encoder.onnx')

trt_out = np.squeeze(trt_model.run(sample_inp.copy().astype('float32')))
with torch.no_grad():
    torch_out = np.squeeze(torch_model(torch.Tensor(sample_inp.copy())).detach().cpu().numpy())

onnx_inp = {onnx_model.get_inputs()[0].name: sample_inp.copy().astype('float32')} 
onnx_out = np.squeeze(onnx_model.run(None, onnx_inp))
print(torch_out[10: 20, 0, 0])
print(onnx_out[10: 20, 0, 0])
print(trt_out[10: 20, 0, 0])