from PIL import Image
import time
from exec_backends.trt_loader import TrtOCREncoder, TrtOCRDecoder
from vietocr.tool.translate import build_model, translate_trt, translate_beam_search, process_input, predict
from vietocr.model.vocab import Vocab
from vietocr.tool.config import Cfg

import torch
from vietocr.tool.predictor import Predictor

class TrtOCR(object):
    def __init__(self, encoder_model, decoder_model, config):
        self.encoder_model = TrtOCREncoder(encoder_model)
        self.decoder_model = TrtOCRDecoder(decoder_model)
        self.config = config
        self.vocab = Vocab(config['vocab'])

    def predict(self, img, return_prob = True):
        '''
            Input:
                - img: pillow Image
        '''
        tik = time.time()
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        

        s, prob = translate_trt(img, self.encoder_model, self.decoder_model)
        s = s[0].tolist()
        prob = prob[0]

        s = self.vocab.decode(s)
        tok = time.time()
        print(tok - tik)
        if return_prob:
            return s, prob
        else:
            return s

if __name__ == '__main__':
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
    config['device'] = 'cuda:0'

    img = Image.open('test_3.png')
    # Trt
    ocr_model = TrtOCR('transformer_encoder.trt', 'transformer_decoder.trt', config)
    ocr_model.predict(img)
    tik = time.time()
    for i in range(100):
        ocr_model.predict(img)
    tok = time.time()
    print(tok - tik)
    # Base line
    # baseline_model = Predictor(config)
    # baseline_model.model.eval()
    # with torch.no_grad():
    #     baseline_model.predict(img)
    # tik = time.time()
    # for i in range(100):
    #     with torch.no_grad():
    #         print(baseline_model.predict(img))
    # tok = time.time()
    # print(tok - tik)