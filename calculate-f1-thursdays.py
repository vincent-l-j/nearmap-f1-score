# Calculate the f1 for Thursdays... this is not a trick question! We will search for the first 5 d.p. in your CV
import pandas as pd
from sklearn import metrics


def main():
    # load test data
    df = pd.read_csv(
        'resource/test.psv',
        sep="|",
        comment='#',
        parse_dates=True,
        index_col='dates',
    )
    # filter out thursdays
    df_thursdays = df[df.index.weekday == 3]
    # calculate f1 score
    f1_score = metrics.f1_score(df_thursdays['y'], df_thursdays['yhat'])
    print(f"{f1_score:.5f}")


if __name__ == "__main__":
    main()
