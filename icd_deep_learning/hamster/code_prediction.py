"""
This is the main function for automated medical coding train
"""

import yaml
from os import path


from hamster.preprocessing.remove_number import RemoveNumber
from hamster.preprocessing.remove_punctuation import RemovePunctuation
from hamster.machine_learning.embedding_layer import convert_data_to_index, \
    create_embedding_layer, create_embedding_matrix, loading_w2v_model
from hamster.machine_learning.utils import zero_pad
from hamster.machine_learning.machine_learning import MachineLearning


class Predictor:

    def __init__(self, manifest_filename):
        manifest_directory = path.dirname(manifest_filename)

        with open(manifest_filename, 'r') as stream:
            manifest = yaml.load(stream)
        self.word2vector = path.join(manifest_directory,
                                     manifest['word2vector_filename'])
        self.deep_learning_directory = path.join(manifest_directory,
                                                 manifest['deep_learning_directory'])

        self.manifest = manifest

    def predict_code(self, text, threshold_label=0.1,
                     threshold_alpha=0.2):
        if not isinstance(text, str):
            return 'Please pass string type for prediction'
        elif not isinstance(threshold_label, float):
            return 'Please pass float type as threshold'
        elif not isinstance(threshold_alpha, float):
            return 'Please pass float type as threshold'
        else:
            data_text = RemoveNumber(text).remove_number()

            data_text = RemovePunctuation(*data_text).remove_punctuation()

            w2v_model = loading_w2v_model(self.word2vector)

            embedding_matrix = create_embedding_matrix(w2v_model)

            embedding_var = create_embedding_layer(embedding_matrix)

            data_index = convert_data_to_index(
                w2v_model, data_text
                )

            data = zero_pad(data_index, self.manifest['machine_learning']['sequence_length'])

            label_orders = ['001', '140', '240', '280', '290', '320', '390', '460', '520',
                            '580', '630', '680', '710', '740', '760', '780', '800', 'e800', 'v01']

            model = MachineLearning(
                number_of_labels=self.manifest['machine_learning']['labels'],
                label_orders=label_orders,
                embedding_matrix=embedding_var,
                sequence_length=self.manifest['machine_learning']['sequence_length'],
                hidden_size=self.manifest['machine_learning']['hidden_size'],
                attention_size=self.manifest['machine_learning']['attention_size'],
                keep_probability=self.manifest['machine_learning']['probability'],
                batch_size=self.manifest['machine_learning']['batch_size'],
                number_of_epochs=self.manifest['machine_learning']['epochs'],
                delta=self.manifest['machine_learning']['delta'],
                save_path=self.deep_learning_directory
            )

            result = model.predict_labels(
                data, text, threshold_label, threshold_alpha)

        return result
