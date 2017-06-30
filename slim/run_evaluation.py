from slim import eval_image_classifier as evaluator


def main():
    evaluator.eval('cars', 2)


if __name__ == '__main__':
    main()
