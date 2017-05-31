import os

from slim import inference_image_classifier as classifier

DATA_DIR = 'test_images/'
vw_01 = os.path.join(DATA_DIR, 'aston_test_01.jpg')

vw_cam = os.path.join(DATA_DIR, 'vw_cam_01.jpg')
bmw_cam = os.path.join(DATA_DIR, 'bmw_cam_01.jpg')
audi_cam = os.path.join(DATA_DIR, 'audi_cam_01.jpg')


def main():
    print(classifier.inference_on_image('root', audi_cam, return_labels=5))
    print('#############################################################')


if __name__ == '__main__':
    main()
