import os

from slim import inference_image_classifier as classifier

DATA_DIR = 'test_images/'
test_images = ['fone04.jpg']


def main():
    for test_image in test_images:
        img = os.path.join(DATA_DIR, test_image)
        print(classifier.inference_on_image(bot_id='', suffix='', setting_id=2, image_file=img,
                                            return_labels=1))
        print('#############################################################')


if __name__ == '__main__':
    main()
