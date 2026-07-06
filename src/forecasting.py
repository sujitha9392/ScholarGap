import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def forecast_keyword_trends(trend_df):
    """
    Forecast next-year keyword/topic count using Linear Regression.

    Input columns:
    - year
    - keyword
    - count
    """

    if trend_df.empty:
        return pd.DataFrame()

    forecast_rows = []

    latest_year = int(trend_df["year"].max())
    next_year = latest_year + 1

    for keyword in trend_df["keyword"].unique():
        temp = trend_df[trend_df["keyword"] == keyword].sort_values("year")

        if len(temp) < 2:
            continue

        X = temp[["year"]].values
        y = temp["count"].values

        model = LinearRegression()
        model.fit(X, y)

        predicted_count = model.predict(np.array([[next_year]]))[0]
        predicted_count = max(0, round(predicted_count, 2))

        current_count = int(temp.iloc[-1]["count"])
        predicted_growth = round(predicted_count - current_count, 2)

        if predicted_growth > 1:
            forecast_status = "Expected to Grow"
        elif predicted_growth < -1:
            forecast_status = "Expected to Decline"
        else:
            forecast_status = "Expected to Stay Stable"

        forecast_rows.append({
            "keyword": keyword,
            "latest_year": latest_year,
            "latest_year_count": current_count,
            "forecast_year": next_year,
            "predicted_count": predicted_count,
            "predicted_growth": predicted_growth,
            "forecast_status": forecast_status
        })

    forecast_df = pd.DataFrame(forecast_rows)

    if not forecast_df.empty:
        forecast_df = forecast_df.sort_values(
            by="predicted_growth",
            ascending=False
        ).reset_index(drop=True)

    return forecast_df