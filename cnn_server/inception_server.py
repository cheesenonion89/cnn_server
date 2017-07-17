import json

import base64
from flask import request, Flask
from flask_cors import CORS
from flask_restful import Resource, Api

from cnn_server import imagenet_classifier as inception

app = Flask(__name__)
CORS(app)
api = Api(app)


def persist_image(file_name, base64_image):
    f = open(file_name, 'wb')
    f.write(base64.b64decode(base64_image))
    f.close()


def classify_image(file_name):
    result = inception.inference(file_name)
    image = Image(file_name)
    image.results = result
    return image


class Image:
    def __init__(self, title):
        self.title = title
        self.labels = None
        self.probabilities = None
        self.results = None

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class Images(Resource):
    def post(self):
        response = request.get_json(force=True)
        base64_image = response['base64Image']
        file_name = response['title'].split('.', 1)[0].replace(':', '_').replace(' ', '_')
        file_name = ("image_%s.jpg" % file_name)
        persist_image(file_name, base64_image)
        image = classify_image(file_name)
        return image.to_json(), 201


api.add_resource(Images, '/image')

if __name__ == "__main__":
    app.run('0.0.0.0')
