import sys
import argparse

from hamster.code_prediction import Predictor


def parse_arguments(argv=None):
    argv = argv or sys.argv
    parser = argparse.ArgumentParser(description='Automatic Medical Coding with Deep Neural Networks.')
    parser.add_argument('manifest', help='Address of manifest.yml function')
    parser.add_argument('text', help='input medical text to predict ICD-9 codes')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='The confidence threshold'
                                                                            ' (default = 0.5)')
    parser.add_argument('-t', '--threshold', type=float, default=0.2, help='The attention threshold'
                                                                           ' (default = 0.2)')
    return parser.parse_args(argv[1:])


def main(argv=None):
    args = parse_arguments(argv=argv)
    predict = Predictor(args.manifest)
    alphas_list, codes, confidence = predict.predict_code(
        args.text, args.confidence, args.threshold
    )

    return alphas_list, codes, confidence
