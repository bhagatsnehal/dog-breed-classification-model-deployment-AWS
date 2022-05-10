'''SageMaker PyTorch inference container overrides to serve ResNet50 model'''


import os 
import smdebug

import json
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    
    print(model_dir)
    
    model = models.resnet50(pretrained=True)
 
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 133))
    
    model = model.to(device)
    
     
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location = device),strict=False)
        print('MODEL-LOADED')

    model.eval()

    return model




def input_fn(request_body, content_type='image/jpeg'):
    
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    logger.debug(f'Request length is: {len(request_body)}')
    logger.debug(f'Request body is: {request_body}')
    if content_type == JPEG_CONTENT_TYPE: 
#         request_body.seek()
        return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# inference
def predict_fn(input_object, model):
    logger.info('In predict fn')
    test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    logger.info("transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction
