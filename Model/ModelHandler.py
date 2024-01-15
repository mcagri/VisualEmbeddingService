import open_clip
from PIL import Image
import requests
from io import BytesIO
import torch.nn.functional as F

model, _, transform = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="laion2b_s13b_b90k"
)


def get_embedding(data):
    img = Image.open(BytesIO(data))
    im = transform(img).unsqueeze(0)
    output = model(im)
    embedding = F.normalize(output['image_features']).tolist()[0]
    return embedding
