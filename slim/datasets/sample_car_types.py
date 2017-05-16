import pandas as pd

from cnn_server.server import file_service as dirs



def sample():
    data = pd.DataFrame().from_csv(dirs.get_transfer_learning_file('car_types'), header=0, sep=';', index_col=None);
    sample = data[0:0]
    labels = pd.Series.unique(data.ix[:, 3])
    print(labels)
    for label in labels:
        if label == 'concept car' or label == 'custom car':
            continue
        print('LABEL: %s' % label)
        subset = data.loc[data['car_type'] == label]
        if not label == 'world rally car':
            sample = sample.append(subset.sample(1005))
        else:
            sample = sample.append(subset)
    print(sample)
    sample.to_csv(dirs.get_transfer_learning_sample_file('car_types'), sep=';', index=False)
sample()
