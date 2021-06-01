from vietocr.tool.config import Cfg
# from vietocr.model.trainer import Trainer
from vietocr.tool.predictor import Predictor
from vietocr.tool.translate import batch_translate_beam_search
import torch
from torch import nn

class OCREncoder(nn.Module):
    def __init__(self, model):
        
        super(OCREncoder, self).__init__()
        
        self.model = model

    def forward(self, img):
        src = self.model.cnn(img)
        memory = self.model.transformer.forward_encoder(src)
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

x = torch.randn(5, 3, 32, 160, requires_grad=True)

model = OCREncoder(trainer.model)
model.eval()
rs = model(x)
print(rs.shape)
dynamic_axes = {'input': {0: 'batch', 3: 'im_width'}, 'output': {0: 'feat_width', 1: 'batch'}}
torch.onnx.export(model,               # model being run
              x,                         # model input (or a tuple for multiple inputs)
              "transformer_encoder.onnx",   # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=12,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes = dynamic_axes)
