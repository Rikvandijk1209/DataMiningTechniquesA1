import polars as pl
import pandas as pd
import os
from datetime import datetime, timedelta, date


class DataPreprocessor:
    def __init__(self):
        pass
    
    def load_and_preprocess_data(self, interval:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the data from the specified path, transform it to a pivot table, and fill in missing dates.
        """
        # Load the data
        df = self.load_data()
        
        # Transform the data to a pivot table
        df_pivot = self.transform_data(df, interval)
        
        # Fill in missing dates, this method is skipped for now as rows that are missing for the mood are dropped
        df_filled = self.fill_date_ranges(df_pivot, interval)

        # We will put the mood into buckets, this is done to make the prediction easier
        df_bucketed = self.put_mood_into_buckets(df_filled, "mood", "mood", 0.25)
        
        # Add the features now to be included in interpolation
        df_features = self.add_features(df_bucketed, "id", "date")

        # Handle missing values
        df_interpolated = self.handle_missing_values(df_features)
        
        # Since we will be predicting the mood, we have to shift all the feature values forward by 1 day
        df_shifted = self.shift_feature_values(df_interpolated)

        # We now split the data into training and test sets, where the test set only consists of the last row for each id
        # The training set consists of all the other rows
        df_train, df_test = self.train_test_split(df_shifted)

        # We can now remove the rows for which the mood is missing in the training set
        df_train = df_train.drop_nulls(subset=["mood"])

        # Standardize the features, we do it here to ensure that the features are standardized after nulls have been removed
        # The features of the test set are included in the calculation of the mean and standard deviation as they are known at this point
        df_train, df_test = self.standardize_features(df_train, df_test)

        # We add a time_since_last_obs feature to the data
        df_train, df_test = self.add_days_since_last_obs(df_train, df_test, "id", "date")

        return df_train.to_pandas(), df_test.to_pandas()

    def load_data(self) -> pl.DataFrame:
        """
        Load the data from the specified path.
        """
        data_path = self.get_data_path()
        # Load the data using polars
        df = pl.read_csv(data_path, separator=",", schema_overrides={"value": pl.Float64}, null_values=["NA"])
        # Remove first column as it is an index column
        df = df.drop("")

        df = df.with_columns(pl.col("time").str.to_datetime())
        return df
    
    def get_data_path(self):
        """
        Get the path to the data file.
        """
        # Get the root directory of the project at folder name DataMiningTechniquesA1
        root_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the path to the data file
        data_path = os.path.join(root_dir, "data", "dataset_mood_smartphone.csv")
        return data_path
    
    def transform_data(self, df: pl.DataFrame, truncation_string:str) -> pl.DataFrame:
        """
        Transform the data from long format to a pivot table. Aggregate the data on date level with value aggregation by specified format.
        The interval of dates was used with 1 day as the eventual target is to predict mood per day.
        """
        # Aggregate on date (day) level
        df = df.with_columns(pl.col("time").dt.truncate(truncation_string).cast(pl.Datetime).alias("truncated_time"))
        # Here we apply all necessary transformations to later select per variable which to use
        df_agg = df.group_by(["id", "truncated_time", "variable"]).agg([
            pl.col("value").sum().alias("value_sum"), 
            pl.col("value").mean().alias("value_mean"),
            ]).sort(["id", "truncated_time", "variable"])
        
        df_pivot:pl.DataFrame = df_agg.pivot(
            values=["value_sum", "value_mean"],
            index = ["id", "truncated_time"],
            columns = "variable",
        )
        # Specify for which variables we should use the mean, the others will use the sum
        mean_col = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]
        # Get the columns that are not in mean_col
        unique_variables = df_agg["variable"].unique().sort().to_list()
        sum_col = list(set(unique_variables) - set(mean_col))

        for col in unique_variables:
            mean_col_name = f"value_mean_{col}"
            sum_col_name = f"value_sum_{col}"
            if col in mean_col:
                # If the column is in mean_col, we will use the mean
                df_pivot = df_pivot.with_columns(pl.col(mean_col_name).alias(col))
            elif col in sum_col:
                df_pivot = df_pivot.with_columns(pl.col(sum_col_name).alias(col))

        # Drop the old sum and mean columns after renaming
        drop_columns = [f"value_sum_{col}" for col in unique_variables] + [f"value_mean_{col}" for col in unique_variables]
        df_pivot = df_pivot.drop(drop_columns)
        
        df_pivot = df_pivot.with_columns(pl.col("truncated_time").cast(pl.Date))  
        # Sort the columns
        return df_pivot.select(["id", "truncated_time"] + [var for var in unique_variables])
    
    def standardize_features(self, df_train:pl.DataFrame, df_test:pl.DataFrame) -> tuple[pl.DataFrame:pl.DataFrame]:
        """
        Standardize the features in the DataFrame.
        The features are standardized using the mean and standard deviation of the training set.
        """
        # Get the feature columns, all columns except id, date and mood (this will)
        feature_cols = list(set(df_train.columns) - set(["id", "date", "mood"]))
        # Get the mean and standard deviation of the training set
        mean = df_train[feature_cols].mean()
        std = df_train[feature_cols].std()

        # Standardize the features in the training set
        df_train = df_train.with_columns([
            (pl.col(col) - mean[col]) / std[col] for col in feature_cols
        ])

        # Standardize the features in the test set
        df_test = df_test.with_columns([
            (pl.col(col) - mean[col]) / std[col] for col in feature_cols
        ])

        return df_train, df_test
    
    def fill_date_ranges(self, df: pl.DataFrame, interval: str) -> pl.DataFrame:
        """
        Fill missing dates between first and last observed date per ID.
        Returns a DataFrame with one row per (id, date).
        """
        # Ensure truncated_time is a Date
        df = df.with_columns(pl.col("truncated_time").cast(pl.Date))

        # Step 1: Get first and last date per id
        id_ranges = df.group_by("id").agg([
            pl.col("truncated_time").min().alias("start_date"),
            pl.col("truncated_time").max().alias("end_date")
        ])

        # Step 2: Build full date ranges per id (in Pandas for convenience)
        full_ranges = []
        for row in id_ranges.iter_rows(named=True):
            date_range = pd.date_range(start=row["start_date"], end=row["end_date"], freq=interval)
            full_ranges.extend([(row["id"], d.date()) for d in date_range])  # convert to `datetime.date`

        # Step 3: Convert to Polars DataFrame
        full_df = pl.DataFrame(full_ranges, schema=["id", "date"], orient="row")

        # Step 4: Join with original df
        df = df.rename({"truncated_time": "date"})
        df_filled = full_df.join(df, on=["id", "date"], how="left")

        return df_filled
    
    def handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Handle missing values in the DataFrame. We will fill the numerical columns with 0 and interpolate the other columns in which
        zeros dont make sense.
        The columns that are filled with 0 are the sum columns from before, the columns that are interpolated are the mean columns from before.
        The mood column is the target variable and should not be filled with 0 or interpolated. Therefore we drop the rows with missing values for the mood.
        """
        # Firstly it does not make sense to fill missing values for the mood as this will be the target variable
        # Therefore we will drop the rows with missing values for the mood
        # df = df.drop_nulls(subset=["mood"])

        # We now define the numerical columns for which we should do interpolation 
        # Note that these are the same as the mean columns from before (minus the mood column)
        cols_interpolate = ["circumplex.arousal", "circumplex.valence", "activity", "lagged_mood"]

        # Next we get the remaining columns for which we fill in 0, these are the sum columns from before
        cols = df.columns
        cols_fill_zero = list(set(cols) - set(cols_interpolate) - {"id", "date", "mood"})

        # Ensure the DataFrame is sorted properly for time-based interpolation
        df = df.sort(by=["id", "date"])
        # Fill 0s
        df_filled = df.with_columns([
            pl.col(col).fill_null(0) for col in cols_fill_zero
        ])

        # Interpolate within each panel group
        df_interpolated = (
            df_filled
            .group_by("id", maintain_order=True)
            .agg([
                pl.col("date"),
                *[pl.col(col).interpolate() for col in cols_interpolate],
                *[pl.col(col) for col in df.columns if col not in cols_interpolate and col not in ["date", "id"]],
            ])
            .explode(df.columns[1:])
        )

        # We can not apply interpolation in case there is no first or last value(s) are missing
        # For these cases we take either the first or last non-null value
        for col in cols_interpolate:
            # Fill the first and last values with the first and last non-null values
            df_interpolated = df_interpolated.with_columns([
                pl.col(col).fill_null(strategy="forward").fill_null(strategy="backward")
            ])


        # Sort the DataFrame by id and date
        df_interpolated = df_interpolated.sort(by=["id", "date"])
        return df_interpolated
    
    def shift_feature_values(self, df: pl.DataFrame, id_col:str = "id", target_col:str = "mood") -> pl.DataFrame:
        """
        Shift the feature values in the DataFrame by one day.
        This is done to ensure that the features are from the previous day and not from the same day.
        """
        # Shift the feature values by one day
        df = df.with_columns(
            pl.col(target_col).shift(1).over(id_col).alias(target_col)
        )
        return df
    
    def put_mood_into_buckets(self, df:pl.DataFrame, target_col:str, bucketed_col_name:str, mood_interval:float) -> pl.DataFrame:
        """
        This function takes a DataFrame and a mood interval and puts the mood into buckets.
        The mood is divided into buckets of the specified interval.
        Furthermore these buckets will become integers to allow for ordinal classification
        """ 
        # Define the mood buckets
        mood_buckets = [i * mood_interval for i in range(int(10/mood_interval) + 1)]
        def assign_bucket(val):
            return next((i for i in mood_buckets if val <= i), None)

        df = df.with_columns(
            pl.col(target_col).map_elements(assign_bucket, return_dtype=pl.Float64).alias(bucketed_col_name)
        )
        # Convert the mood buckets to integers
        def map_mood_to_integer(mood_value):
            # Divide mood by 0.25 and round it to the nearest integer
            return int(round(mood_value / 0.25))

        # Apply the mapping function to the mood column (assuming column name is 'mood')
        df = df.with_columns(
            pl.col(bucketed_col_name).map_elements(map_mood_to_integer, return_dtype=pl.Int32).alias("mood")
        )
        return df
    
    def add_features(self, df: pl.DataFrame, id_col:str, time_col:str) -> pl.DataFrame:
        """
        Add features to the DataFrame. The features are:
        - time_since_last_obs: The time since the last observation in hours.
        - lagged_mood: The mood of the last observation.
        """
        # Sort by id and time to ensure proper chronological order for each id
        df = df.sort([id_col, time_col])
        
        # Lagged mood, this is set to the current mood as we will shift the features later
        df = df.with_columns(
            pl.col("mood").alias("lagged_mood")
        )

        # Day of the week
        df = df.with_columns(
            pl.col(time_col).dt.weekday().alias("day_of_week")
        )
        
        return df
    
    def train_test_split(self, df:pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split the DataFrame into training and test sets.
        The training set consists of all the rows except the last row based on date for each id.
        The test set consists of the last row based on date for each id.
        """
        # Get the last row for each id
        last_rows = df.group_by("id").agg(pl.col("date").last().alias("date"))
        # Join the last rows with the original DataFrame to get the last row for each id
        test_df = df.join(last_rows, on=["id", "date"], how="inner")
        # Get the training set by filtering out the last row for each id
        train_df = df.join(last_rows, on=["id", "date"], how="anti")
        # Sort the training and test sets by id and date
        train_df = train_df.sort(["id", "date"])
        test_df = test_df.sort(["id", "date"])
        
        return train_df, test_df

    def add_days_since_last_obs(self, train_df:pl.DataFrame, test_df:pl.DataFrame, id_col:str, time_col:str) -> pl.DataFrame:
        """
        Add a days since last observation feature to the DataFrame. For the training set this can be done by 
        taking the difference between the current date and the previous date for each id observation by a shift over id.
        For the test set we will take the difference between the current date and the last date in the training set 
        for each id observation.
        """
        # Sort the DataFrame by id and date
        train_df = train_df.sort([id_col, time_col])
        test_df = test_df.sort([id_col, time_col])

        # Add the days since last observation feature to the training set
        train_df = train_df.with_columns(
            (pl.col(time_col) - pl.col(time_col).shift(1).over(id_col)).dt.total_days().alias("days_since_last_obs")
        ).fill_null(0)
        # For the test set we will first add the last date from the training set to the test set
        last_dates = train_df.group_by(id_col).agg(pl.col(time_col).last().alias("last_date"))
        # Join the last dates with the test set to get the last date for each id
        test_df = test_df.join(last_dates, on=id_col, how="left")
        # Add the days since last observation feature to the test set
        test_df = test_df.with_columns(
            (pl.col(time_col) - pl.col("last_date")).dt.total_days().alias("days_since_last_obs")
        )

        return train_df, test_df
    



