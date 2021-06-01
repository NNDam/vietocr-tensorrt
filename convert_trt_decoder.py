from vietocr.tool.config import Cfg
from vietocr.tool.translate import batch_translate_beam_search
from vietocr.tool.predictor import Predictor
import torch
from torch.nn.functional import log_softmax, softmax
from torch import nn
from PIL import Image


class OCRDecoder(nn.Module):
    def __init__(self, model):
        
        super(OCRDecoder, self).__init__()
        
        self.model = model

    def forward(self, tgt_inp, memory):
        output, memory = self.model.transformer.forward_decoder(tgt_inp, memory)
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

# trainer = Trainer(config, pretrained=True)

predictor = Predictor(config)
img = Image.open('image/test_2.png')
print(predictor.predict(img))
tgt_inp = torch.randint(0, 232, (20, 1))
memory = torch.randn(170, 1, 256, requires_grad=True)

model = OCRDecoder(predictor.model)
model.eval()
rs = model(tgt_inp, memory)
print(rs[0].shape, rs[1].shape, rs[2].shape)
dynamic_axes = {'tgt_inp': {0: 'sequence_length'}, \
                'memory': {0: 'feat_width'}, \
                'values': {1: 'sequence_length'}, \
                'indices': {1: 'sequence_length'}}
# Fix triu operator
torch_triu = torch.triu
def triu_onnx(x, diagonal=0, out=None):
    assert out is None
    assert len(x.shape) == 2 and x.size(0) == x.size(1)
    template = torch_triu(torch.ones((128, 128), dtype=torch.int32), diagonal)   #1024 is max sequence length
    mask = template[:x.size(0),:x.size(1)]
    return torch.where(mask.bool(), x, torch.zeros_like(x))
torch.triu = triu_onnx

# Convert to ONNX
torch.onnx.export(model,               # model being run
              (tgt_inp, memory),                         # model input (or a tuple for multiple inputs)
              "transformer_decoder.onnx",   # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=12,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['tgt_inp', 'memory'],   # the model's input names
              output_names = ['values', 'indices'], # the model's output names
              dynamic_axes = dynamic_axes)
