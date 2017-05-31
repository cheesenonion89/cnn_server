import os

from slim import inference_image_classifier as classifier

DATA_DIR = 'test_images/'
aston_01 = os.path.join(DATA_DIR, 'aston_test_01.jpg')

vw_cam = os.path.join(DATA_DIR, 'vw_cam_01.jpg')
bmw_cam = os.path.join(DATA_DIR, 'bmw_cam_01.jpg')
audi_cam = os.path.join(DATA_DIR, 'audi_cam_01.jpg')

vw_side = os.path.join(DATA_DIR, 'vw_side_view.jpg')
bmw_side = os.path.join(DATA_DIR, 'bmw_side_view.jpg')
audi_side = os.path.join(DATA_DIR, 'audi_side_view.jpg')

bmw_logo = os.path.join(DATA_DIR, 'bmw_logo.jpg')
bmw_nologo = os.path.join(DATA_DIR, 'bmw_nologo.jpg')

ninja01 = os.path.join(DATA_DIR, 'bmw_ninja_01.jpg')
ninja02 = os.path.join(DATA_DIR, 'bmw_ninja_02.jpg')
ninja03 = os.path.join(DATA_DIR, 'bmw_ninja_03.jpg')
sneaky_ninja03 = os.path.join(DATA_DIR, 'bmw_ninja_03_blurred.jpg')

def main():
    print(classifier.inference_on_image('root', sneaky_ninja03, return_labels=5))
    print('#############################################################')


if __name__ == '__main__':
    main()
