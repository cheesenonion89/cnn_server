import os

from slim import inference_image_classifier as classifier

DATA_DIR = 'test_images/'
test_images = {
    'lambo_01': os.path.join(DATA_DIR, 'lambo01.jpg')
}


def main():
    for test_image in test_images:
        print(classifier.inference_on_image('root', test_image, return_labels=5))
        print('#############################################################')


if __name__ == '__main__':
    main()
