import pandas as pd
import numpy as np
import hisel


def main():
    df = pd.read_csv('mydata.csv')
    xdf = df.iloc[:, :-1]
    ydf = df.iloc[:, [-1]]
    discrete_features = [col for col in xdf if col.startswith('d')]
    continuous_features = [col for col in xdf if col.startswith('c')]
    all_features = discrete_features + continuous_features
    # Make sure that the discrete features are listed at the front!
    x = xdf[all_features].values
    y = ydf.values

    selector = hisel.select.HSICSelector(
        x, y,
        xfeattype=hisel.select.FeatureType.BOTH,
        yfeattype=hisel.select.FeatureType.CONT,
        feature_names=all_features,
        catcont_split=len(discrete_features),
    )
    selected_features = selector.select(
        number_of_features=10,
        number_of_epochs=4
    )

    print('\n\n##########################################################')
    print(
        f'The following features are relevant for the prediction of {ydf.columns.tolist()}:')
    print(f'{selected_features}')


if __name__ == '__main__':
    main()
