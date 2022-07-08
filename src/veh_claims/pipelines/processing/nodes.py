import pandas as pd

from typing import Dict, Any

from category_encoders import BinaryEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def clean_dataset(dataset: pd.DataFrame) -> Dict[str, Any]:
    # Since both DayOfWeekClaimed and MonthClaimed are 0 for the same entry, I will drop
    dataset.drop("PolicyNumber", axis=1, inplace=True)
    df: pd.DataFrame = dataset.loc[dataset["DayOfWeekClaimed"] != "0"]
    df.reset_index(drop=True, inplace=True)

    # Need to check the policy holders age and then reassign a value.
    df2_age0_idx = df["Age"] == 0
    # Assigns an age of 16 to all rows with missing Age values
    df.loc[list(df2_age0_idx), "Age"] = 16.5
    df.drop_duplicates(inplace=True)
    return dict(clean=df)


def encode_features(dataset: pd.DataFrame):
    """
    Encode features of data file.
    """
    features = dataset.copy()

    col_ordering = [
        {
            "col": "Month",
            "mapping": {
                "Jan": 1,
                "Feb": 2,
                "Mar": 3,
                "Apr": 4,
                "May": 5,
                "Jun": 6,
                "Jul": 7,
                "Aug": 8,
                "Sep": 9,
                "Oct": 10,
                "Nov": 11,
                "Dec": 12,
            },
        },
        {
            "col": "DayOfWeek",
            "mapping": {
                "Monday": 1,
                "Tuesday": 2,
                "Wednesday": 3,
                "Thursday": 4,
                "Friday": 5,
                "Saturday": 6,
                "Sunday": 7,
            },
        },
        {
            "col": "DayOfWeekClaimed",
            "mapping": {
                "Monday": 1,
                "Tuesday": 2,
                "Wednesday": 3,
                "Thursday": 4,
                "Friday": 5,
                "Saturday": 6,
                "Sunday": 7,
            },
        },
        {
            "col": "MonthClaimed",
            "mapping": {
                "Jan": 1,
                "Feb": 2,
                "Mar": 3,
                "Apr": 4,
                "May": 5,
                "Jun": 6,
                "Jul": 7,
                "Aug": 8,
                "Sep": 9,
                "Oct": 10,
                "Nov": 11,
                "Dec": 12,
            },
        },
        {
            "col": "PastNumberOfClaims",
            "mapping": {"none": 0, "1": 1, "2 to 4": 2, "more than 4": 5},
        },
        {
            "col": "NumberOfSuppliments",
            "mapping": {"none": 0, "1 to 2": 1, "3 to 5": 3, "more than 5": 6},
        },
        {
            "col": "VehiclePrice",
            "mapping": {
                "more than 69000": 69001,
                "20000 to 29000": 24500,
                "30000 to 39000": 34500,
                "less than 20000": 19999,
                "40000 to 59000": 49500,
                "60000 to 69000": 64500,
            },
        },
        {
            "col": "AgeOfVehicle",
            "mapping": {
                "3 years": 3,
                "6 years": 6,
                "7 years": 7,
                "more than 7": 8,
                "5 years": 5,
                "new": 0,
                "4 years": 4,
                "2 years": 2,
            },
        },
        {"col": "AccidentArea", "mapping": {"Rural": 0, "Urban": 1}},
        {"col": "Sex", "mapping": {"Male": 0, "Female": 1}},
        {"col": "Fault", "mapping": {"Third Party": 0, "Policy Holder": 1}},
        {"col": "PoliceReportFiled", "mapping": {"No": 0, "Yes": 1}},
        {"col": "WitnessPresent", "mapping": {"No": 0, "Yes": 1}},
        {"col": "AgentType", "mapping": {"Internal": 0, "External": 1}},
        {
            "col": "Days_Policy_Accident",
            "mapping": {
                "more than 30": 31,
                "15 to 30": 22.5,
                "none": 0,
                "1 to 7": 4,
                "8 to 15": 11.5,
            },
        },
        {
            "col": "Days_Policy_Claim",
            "mapping": {
                "more than 30": 31,
                "15 to 30": 22.5,
                "8 to 15": 11.5,
                "none": 0,
            },
        },
        {
            "col": "AgeOfPolicyHolder",
            "mapping": {
                "16 to 17": 16.5,
                "18 to 20": 19,
                "21 to 25": 23,
                "26 to 30": 28,
                "31 to 35": 33,
                "36 to 40": 38,
                "41 to 50": 45.5,
                "51 to 65": 58,
                "over 65": 66,
            },
        },
        {
            "col": "AddressChange_Claim",
            "mapping": {
                "no change": 0,
                "under 6 months": 0.5,
                "1 year": 1,
                "2 to 3 years": 2.5,
                "4 to 8 years": 6,
            },
        },
        {
            "col": "NumberOfCars",
            "mapping": {
                "1 vehicle": 1,
                "2 vehicles": 2,
                "3 to 4": 3.5,
                "5 to 8": 6.5,
                "more than 8": 9,
            },
        },
    ]
    cols_ordinal = [x["col"] for x in col_ordering]
    cols_ordinal_float = [
        "Days_Policy_Accident",
        "Days_Policy_Claim",
        "AgeOfPolicyHolder",
        "AddressChange_Claim",
        "NumberOfCars",
    ]
    cols_ordinal_int = list(set(cols_ordinal) - set(cols_ordinal_float))
    ord_encoder = OrdinalEncoder(mapping=col_ordering, return_df=True)
    features_ordered = ord_encoder.fit_transform(features)
    for col in cols_ordinal_int:
        features_ordered[col] = features_ordered[col].astype("int")

    cols_onehot = [
        "Make",
        "MaritalStatus",
        "PolicyType",
        "VehicleCategory",
        "BasePolicy",
    ]
    one_hot_encoder = OneHotEncoder(
        cols=cols_onehot, use_cat_names=True, return_df=True
    )
    features_onehot_ordered = one_hot_encoder.fit_transform(features_ordered)

    encoders = [(cols_ordinal, ord_encoder), (cols_onehot, one_hot_encoder)]
    return dict(features=features_onehot_ordered, transform_pipeline=encoders)


def split_dataset(dataset: pd.DataFrame, test_ratio: float) -> Dict[str, Any]:
    """
    Splits dataset into a training set and a test set.
    """
    X = dataset.drop("FraudFound_P", axis=1)
    y = dataset["FraudFound_P"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=40
    )

    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
