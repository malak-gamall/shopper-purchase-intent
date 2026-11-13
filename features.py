import numpy as np

def add_features(d):
    d = d.copy()
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDuration"] = d["Administrative_Duration"] + d["Informational_Duration"] + d["ProductRelated_Duration"]
    d["AvgSecPerPage"] = d["TotalDuration"] / d["TotalPages"].replace(0, np.nan)
    d["HasPageValue"] = (d["PageValues"] > 0).astype(int)
    return d
