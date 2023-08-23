import pandas as pd
import hisel


def main():
    # Minimial example of `hisel` usage
    df = pd.read_csv('mydata.csv')
    xdf = df.iloc[:, :-1]
    yser = df.iloc[:, -1]
    results = hisel.feature_selection.select_features(xdf, yser)
    print('\n\n##########################################################')
    print(
        f'The following features are relevant for the prediction of {yser.name}:')
    print(f'{results.selected_features}')


if __name__ == '__main__':
    main()
