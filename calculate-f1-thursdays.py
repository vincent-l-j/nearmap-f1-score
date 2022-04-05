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
    day = 'Thursday'
    # calculate f1 score for Thursdays
    weekdays = (
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday',
    )
    f1_score = calculate_weekday_f1_score(df, weekdays.index(day))
    print(f"The f1 score for {day}s is: {f1_score:.5f}")


def calculate_weekday_f1_score(df, day):
    """
    Calculate the f1 score for a particular day of the week.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with test and prediction values. It is
        assumed that the test column is named `y` and prediction
        column is named `yhat`.
    day : int
        The day of the week. It is assumed the week starts on Monday,
        which is denoted by 0 and ends on Sunday which is denoted by 6.
    """
    # filter out day of the week
    df_weekday = df[df.index.weekday == day]
    # calculate f1 score
    f1_score = metrics.f1_score(df_weekday['y'], df_weekday['yhat'])

    return f1_score


if __name__ == "__main__":
    main()
