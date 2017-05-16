import pandas as pd

from cnn_server.server import file_service as dirs

# Oversampling to compensate for missing images
SAMPLE_SIZES = [10015, 10015, 10015]


def sample():
    data = pd.DataFrame().from_csv(dirs.get_transfer_learning_file('seasons'), header=0, sep=';', index_col=None);
    sample = data[0:0]
    labels = pd.Series.unique(data.ix[:, 3])
    for label in labels:
        print('LABEL: %s' % label)
        subset = data.loc[data['season'] == label]
        sample = sample.append(subset.sample(10015))
    print(sample)
    sample.to_csv(dirs.get_transfer_learning_sample_file('seasons'), sep=';', index=False)
sample()
