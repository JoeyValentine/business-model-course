import io
from google.cloud import vision
from google.cloud.vision import types
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from enum import Enum

DISCOVERY_URL='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'

"""Run a label request on a single image"""

credentials = GoogleCredentials.get_application_default()
service = discovery.build('vision', 'v1', credentials=credentials,
                          discoveryServiceUrl=DISCOVERY_URL)

photo_file = "D:/NaverCloud/Lecture/소프트웨어실습/drum.jpg"
#photo_file = "D:/Tesseract-OCR/licensePlate1.jpg"

def detect_labels(path):
    """Detects labels in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels:')

    for label in labels:
        print(label.description)

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


detect_labels(photo_file)

