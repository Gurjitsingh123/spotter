from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.8
EQUIPMENT_TYPES = ["Dry Van", "Reefer", "Flatbed"]
SHORT_HAUL_MILES = 150


def straight_line_miles(pickup_lat, pickup_lon, delivery_lat, delivery_lon):
    pickup_lat, pickup_lon, delivery_lat, delivery_lon = map(
        np.radians, [pickup_lat, pickup_lon, delivery_lat, delivery_lon]
    )
    half_chord = (
        np.sin((delivery_lat - pickup_lat) / 2) ** 2
        + np.cos(pickup_lat) * np.cos(delivery_lat) * np.sin((delivery_lon - pickup_lon) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(half_chord))


def find_bad_prices(loads: pd.DataFrame, too_low: float = 0.5, too_high: float = 2.0) -> pd.Series:
    rate_per_mile = loads["posted_rate"] / loads["distance"]
    usual_rate_per_mile = rate_per_mile.groupby(loads["equipment"]).transform("median")
    ratio = rate_per_mile / usual_rate_per_mile
    return (ratio < too_low) | (ratio > too_high)


def daily_market_levels(*load_tables: pd.DataFrame) -> pd.DataFrame:
    all_loads = pd.concat(
        [table[["date", "market_index", "quote_signal"]] for table in load_tables], ignore_index=True
    )
    all_loads["date"] = pd.to_datetime(all_loads["date"])
    return all_loads.groupby("date")[["market_index", "quote_signal"]].mean()


class FeatureBuilder:
    def fit(self, training_loads: pd.DataFrame) -> "FeatureBuilder":
        weights = training_loads["weight"].abs()
        self.median_weight_by_equipment = weights.groupby(training_loads["equipment"]).median().to_dict()

        pickups = training_loads[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        deliveries = training_loads[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        self.city_locations = pd.concat([pickups, deliveries]).groupby("city")[["lat", "lon"]].first()
        return self

    def add_city_coordinates(self, loads: pd.DataFrame) -> pd.DataFrame:
        filled = loads.copy()
        for side in ("pickup", "delivery"):
            for axis in ("lat", "lon"):
                column = f"{side}_{axis}"
                known = filled[column] if column in filled else pd.Series(np.nan, index=filled.index)
                filled[column] = known.fillna(filled[side].map(self.city_locations[axis]))
        return filled

    def transform(self, loads: pd.DataFrame, market_by_day: pd.DataFrame) -> pd.DataFrame:
        loads = loads.copy()
        loads["date"] = pd.to_datetime(loads["date"])

        loads["weight"] = loads["weight"].abs()
        loads["weight"] = loads["weight"].fillna(loads["equipment"].map(self.median_weight_by_equipment))

        days_market_index = loads["date"].map(market_by_day["market_index"])
        days_quote_signal = loads["date"].map(market_by_day["quote_signal"])
        if "market_index" in loads:
            loads["market_index"] = loads["market_index"].fillna(days_market_index)
        else:
            loads["market_index"] = days_market_index
        if "quote_signal" in loads:
            loads["quote_signal"] = loads["quote_signal"].fillna(days_quote_signal)
        else:
            loads["quote_signal"] = days_quote_signal

        features = pd.DataFrame(index=loads.index)
        features["log_distance"] = np.log1p(loads["distance"])
        features["distance"] = loads["distance"]
        features["straight_miles"] = straight_line_miles(
            loads["pickup_lat"], loads["pickup_lon"], loads["delivery_lat"], loads["delivery_lon"]
        )
        features["short_haul"] = (loads["distance"] < SHORT_HAUL_MILES).astype(int)
        features["weight"] = loads["weight"]
        for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
            features[column] = loads[column]
        features["lat_delta"] = loads["delivery_lat"] - loads["pickup_lat"]
        features["lon_delta"] = loads["delivery_lon"] - loads["pickup_lon"]
        features["market_index"] = loads["market_index"]
        features["market_index_day"] = days_market_index.fillna(loads["market_index"])
        features["quote_signal"] = loads["quote_signal"]
        features["quote_signal_day"] = days_quote_signal.fillna(loads["quote_signal"])
        features["day_of_week"] = loads["date"].dt.dayofweek
        for equipment in EQUIPMENT_TYPES:
            features[f"eq_{equipment.replace(' ', '_').lower()}"] = (loads["equipment"] == equipment).astype(int)
        return features
