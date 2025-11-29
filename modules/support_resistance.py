"""Support and Resistance Detection Module"""
import numpy as np
from sklearn.cluster import DBSCAN


def find_support_resistance(df, lookback=2, eps=2.0, min_samples=2, count=5):
    """Find support and resistance levels using clustering."""
    highs = df["High"].values
    lows = df["Low"].values
    close = df["Close"].iloc[-1]

    # Find swing highs & lows
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df)-lookback):
        high_window = highs[i-lookback: i+lookback+1]
        low_window = lows[i-lookback: i+lookback+1]

        if highs[i] == max(high_window):
            swing_highs.append(highs[i])

        if lows[i] == min(low_window):
            swing_lows.append(lows[i])

    swing_highs = np.array(swing_highs).reshape(-1, 1) if swing_highs else np.array([]).reshape(-1, 1)
    swing_lows = np.array(swing_lows).reshape(-1, 1) if swing_lows else np.array([]).reshape(-1, 1)

    # Clustering
    def cluster(values):
        if len(values) == 0:
            return [], []
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(values)
        labels = clustering.labels_
        unique = [lbl for lbl in set(labels) if lbl != -1]

        zones = []
        for lbl in unique:
            pts = values[labels == lbl]
            zones.append((float(pts.min()), float(pts.max())))

        zones = sorted(zones, key=lambda x: x[0])
        levels = [round((a+b)/2, 2) for (a, b) in zones]
        return zones, levels

    res_zones, res_levels = cluster(swing_highs)
    sup_zones, sup_levels = cluster(swing_lows)

    # Filter resistances above close
    res_levels_above = sorted([lvl for lvl in res_levels if lvl > close])[:count]
    res_zones_above = sorted([z for z in res_zones if z[0] > close or z[1] > close],
                              key=lambda x: x[0])[:count]

    # Filter supports below close
    sup_levels_below = sorted([lvl for lvl in sup_levels if lvl < close],
                               reverse=True)[:count]

    sup_zones_below = sorted([z for z in sup_zones if z[1] < close],
                              key=lambda x: x[0], reverse=True)[:count]

    return {
        'nearest_resistances': res_levels_above,
        'nearest_supports': sup_levels_below,
        'resistance_zones': res_zones_above,
        'support_zones': sup_zones_below
    }