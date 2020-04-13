import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima

cumulCases = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"


def dataReadin(csvlink, country):
    cumulDf = pd.read_csv(csvlink)
    df = pd.DataFrame()
    df = cumulDf.loc[cumulDf["Country/Region"] == country]
    df = df.iloc[:, 4:].transpose()
    dt_index = pd.to_datetime(df.index, format="%m/%d/%y")
    df.index = dt_index
    df.rename(columns={df.columns[0]: "Actual"}, inplace=True)
    df = pd.DataFrame(df)
    plot = plt.plot(df["Actual"], "ob-", label="Actual")
    plt.title("US Daily Cumulative Cases")
    plt.legend(loc="upper left", fontsize=8)
    plt.show()
    print(df)
    return df


data = dataReadin(cumulCases, "US")


# Forecasting US Cases using ARIMA
# Autocorrelation Plot (using Pandas Autocorrelation Plot)
ts = data["Actual"]
pd.plotting.autocorrelation_plot(ts)

# Training and Comparing Actual with Forecasts


def cumul_train_arima(df, target_value, order):
    # Model Fitting
    size = int(len(df.head(-7)))
    train, test = df[0:size], df[size : len(df)]
    target = train[[target_value]]
    model = auto_arima(target, max_order=order)
    fitted = model.fit(target)
    # print(model.summary())
    fc, confint = model.predict(n_periods=len(test), return_conf_int=True)

    # make series for plotting purpose
    index_of_fc = test.index
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(12, 10), dpi=200)
    plt.plot(train[target_value], "ob-", label="training")
    plt.plot(test[target_value], "og-", label="actual")
    plt.plot(fc_series, "sr-", label="forecast")
    plt.fill_between(
        lower_series.index, lower_series, upper_series, color="k", alpha=0.20
    )
    plt.title("Forecast vs Actuals")
    plt.legend(loc="upper left", fontsize=8)
    plt.show()

    forecastdf = pd.DataFrame({"Forecast": fc[:]}).astype(np.int64)
    test = test.reset_index()
    test.rename(columns={"index": "date"}, inplace=True)

    comparison = pd.merge(test, forecastdf, left_index=True, right_index=True)
    comparison["Absolute_Error"] = abs(comparison["Actual"] - comparison["Forecast"])
    # Accuracy
    test_series = np.array(test["Actual"])
    MAPE = np.mean(np.abs(fc - test_series) / np.abs(test_series))  # MAPE
    accuracy = 1 - MAPE
    return print("Model Accuracy: " + "{:.0%}".format(accuracy)), print(comparison)


cumul_train_arima(data, "Actual", 7)

# Generating Future Forecasts


def cumul_forecast_arima(df, target_value, order, periods):
    # Model Fitting
    target = df[[target_value]]
    model = auto_arima(target, max_order=order)
    fitted = model.fit(target)
    print(model.summary())
    fc, confint = model.predict(n_periods=periods, return_conf_int=True)

    # make series for plotting purpose
    index_of_fc = np.arange(len(target), len(target) + periods)
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    asofDate = datetime.datetime.strftime(df.index[-1], "%B %d %Y")
    df = df.reset_index()
    df = df.rename(columns={"index": "Date"})
    target = df[[target_value]]
    plt.figure(figsize=(12, 10), dpi=200)
    plt.plot(target, "ob-", label="actual")
    plt.plot(fc_series, "sr-", label="forecast")
    plt.fill_between(
        lower_series.index, lower_series, upper_series, color="k", alpha=0.20
    )
    plt.title("US Seven Day Cumulative Cases Forecast as of {}".format(asofDate))
    plt.legend(loc="upper left", fontsize=10)
    plt.show()

    # Create dataframe of future forecast
    date_today = datetime.datetime.today()
    days = pd.date_range(
        date_today, date_today + datetime.timedelta(periods - 1), freq="D"
    )
    future = pd.DataFrame({"date": days, "forecast": fc})
    future["date"] = pd.to_datetime(future["date"])
    future["forecast"] = future["forecast"].astype(np.int64)

    return print("Seven Day Forecast:"), print(future)


cumul_forecast_arima(data, "Actual", 5, 7)
