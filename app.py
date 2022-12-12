# adapted from https://github.com/4ndrewparr/clarius-tha-1/blob/main/api/api-pytorch.py
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image

from flask import Flask, request
from flasgger import Swagger

import torch
from torchvision import models
from torchvision import transforms as T
import segmentation_models_pytorch as smp

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/segmentation', methods=["POST"])
def segmentation():
    """Endpoint to perform semantic segmentation of floodwater. A 'POST' implementation.
	---
	parameters:
		-	name: input_image
			in: formData
			type: file
			required: true
			
	responses:
		200:
			description: "
				__Response:__ JSON array of an image where
					each pixel has been assigned one class.\n
				__Classes:__ {
					0: 'not floodwater', 1: 'floodwater'
				}
			"
	"""
    img = Image.open(request.files.get("input_image")) #../optical data/test.png

    trf = T.Compose([
        T.Resize([512,512]),
        T.ToTensor()])
    inp = trf(img)[:3, :, :].unsqueeze(0)  # we need batch dim
    #inp = inp.unsqueeze(0)  # we need batch dim

    PATH = '../Unet-Mobilenet-overall-miou-0.74.pth'

    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=2, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()

    output = model(inp) #model preds
    
#     def predb_to_mask(predb, idx):
#     p = torch.functional.F.softmax(predb[idx], 0)
#     return p.argmax(0).cpu()
    
    
    out_merged = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    out_json = json.dumps(out_merged.tolist())

    #predb_to_mask(output)
    return out_json


# if __name__ == '__main__':
#     #app.debug = True
#     app.run(host='0.0.0.0', port=8888)
