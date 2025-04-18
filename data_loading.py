import polars as pl
import pandas as pd
import os
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self):
        pass
    
    def load_and_preprocess_data(self, interval:str, bucket_step:float, technique: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the data from the specified path, transform it to a pivot table, and fill in missing dates.
        """
        # Load the data
        df = self.load_data()
        
        # Transform the data to a pivot table
        df_pivot = self.transform_data(df, interval)

        # Fill in missing dates
        df_filled = self.fill_date_ranges(df_pivot, interval)

        # We will put the mood into buckets, this is done to make the prediction easier
        df_bucketed = self.put_mood_into_buckets(df_filled, "mood", "mood", bucket_step)
        
        # Add the features now to be included in interpolation
        df_features = self.add_features(df_bucketed, "id", "date")

        # Handle missing values based on the technique
        if technique == 1:
            df_interpolated = self.handle_missing_values(df_features)
        elif technique == 2:
            df_interpolated = self.handle_missing_values_decay(df_features)
        elif technique == 3:
            df_interpolated = self.handle_missing_values_with_filling(df_features)
        else:
            # Add fallback behavior or raise an error if technique is invalid
            print(f"Warning: Invalid technique {technique}. Defaulting to technique 1 (interpolation).")
            df_interpolated = self.handle_missing_values(df_features)  # Defaulting to technique 1
        
        # Since we will be predicting the mood, we have to shift all the feature values forward by 1 day
        df_shifted = self.shift_feature_values(df_interpolated)

        # We now split the data into training and test sets, where the test set only consists of the last row for each id
        # The training set consists of all the other rows
        df_train, df_pred = self.train_pred_split(df_shifted)

        # We can now remove the rows for which the mood is missing in the training set
        df_train = df_train.drop_nulls(subset=["mood"])

        # We add a time_since_last_obs feature to the data
        df_train, df_pred = self.add_days_since_last_obs(df_train, df_pred, "id", "date")

        # The following methods use pandas rather than polars
        df_train = df_train.to_pandas()
        df_pred = df_pred.to_pandas()

        # We will now split the training set into a training and validation set
        df_train, df_val = self.split_train_val(df_train, fraction=0.2)

        # Standardize the features, we do it here to ensure that the features are standardized after nulls have been removed
        # The features of the test set are included in the calculation of the mean and standard deviation as they are known at this point
        df_train, df_val = self.standardize_per_id(df_train, df_val)
        df_train, df_pred = self.standardize_per_id(df_train, df_pred)

        return df_train, df_val, df_pred

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
      
    def standardize_per_id(self, df_train: pd.DataFrame, df_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        new_train_dfs = []
        new_pred_dfs = []
        
        for id_val in df_train["id"].unique():
            id_train = df_train[df_train["id"] == id_val].copy()
            id_pred = df_pred[df_pred["id"] == id_val].copy()
            
            id_train, id_pred = self.standardize_features(id_train, id_pred)
            
            new_train_dfs.append(id_train)
            new_pred_dfs.append(id_pred)
        
        new_df_train = pd.concat(new_train_dfs, ignore_index=True)
        new_df_pred = pd.concat(new_pred_dfs, ignore_index=True)
        
        return new_df_train, new_df_pred
    
    def standardize_features(self, df_train: pd.DataFrame, df_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize the features in the DataFrame using training stats.
        This is done on a per-ID level (handled in the caller function).
        """
        df_train, df_pred = self.add_days_since_last_obs(df_train, df_pred, id_col="id", date_col="date")

        # Select feature columns
        feature_cols = [col for col in df_train.columns if col not in ["id", "date", "mood"]]

        # Compute mean and std from training set
        mean = df_train[feature_cols].mean()
        std = df_train[feature_cols].std()

        # Avoid division by zero by replacing 0 std with 1
        std_replaced = std.replace(0, 1)

        # Standardize train and test using training stats
        df_train[feature_cols] = (df_train[feature_cols] - mean) / std_replaced
        df_pred[feature_cols] = (df_pred[feature_cols] - mean) / std_replaced

        return df_train, df_pred
    
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
            pl.col(target_col).map_elements(assign_bucket, return_dtype=pl.Float32).alias(bucketed_col_name)
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
    
    def train_pred_split(self, df:pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
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
    
    def split_train_val(train_df: pd.DataFrame, fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the training DataFrame into training and validation sets for each unique `id`.
        The split is done by time, ensuring that the training set contains earlier data than the validation set.
        By doing this per id we ensure that the model is trained on past data and validated on future data for each user.
        """
        # Step 1: Split each `id`'s time series
        train_ids = train_df['id'].unique()
        train_split_per_id = []

        for id in train_ids:
            # Get all data for the current `id`
            id_data = train_df[train_df['id'] == id]
            
            # Sort by time (assuming 'date' is the datetime column)
            id_data_sorted = id_data.sort_values(by='date')
            
            # Calculate the split index (e.g., use 90% for training)
            split_idx = int((1-fraction) * len(id_data_sorted))
            
            # Split the data
            train_data = id_data_sorted[:split_idx]
            val_data = id_data_sorted[split_idx:]
            
            # Append to the list of splits
            train_split_per_id.append((train_data, val_data))

        # Step 2: Combine all the training and validation splits
        train_df_split = pd.concat([train_data for train_data, _ in train_split_per_id])
        val_df_split = pd.concat([val_data for _, val_data in train_split_per_id])

        return train_df_split, val_df_split

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
        ).drop("last_date")

        return train_df, test_df
      
    def handle_missing_values_decay(self, df: pl.DataFrame, alpha: float = 0.1) -> pl.DataFrame:
        """
        Fill missing values using exponential decay toward the column mean.
        """
        cols_decay = [col for col in df.columns if col not in ["id", "date"]]
        df_filled = df.sort(["id", "date"])

        # Group by ID and apply exponential decay row-wise
        result_dfs = []

        for user_id, group_df in df_filled.group_by("id"):
            group_dict = {"id": [], "date": []}
            for col in cols_decay:
                group_dict[col] = []

            group_np = group_df.to_pandas()
            group_dict["id"] = group_np["id"].values
            group_dict["date"] = group_np["date"].values

            for col in cols_decay:
                values = group_np[col].values
                col_mean = np.nanmean(values)
                filled = self._exponential_decay_fill(values, alpha=alpha, toward=col_mean)
                group_dict[col] = filled

            result_dfs.append(pl.DataFrame(group_dict))

        df_result = pl.concat(result_dfs)
        return df_result

    def _exponential_decay_fill(self, values, alpha=0.1, toward=0.0):
        """
        Apply exponential decay to fill missing values.
        Missing values decay toward a target (default: column mean).
        """
        filled = []
        last_valid = None

        for v in values:
            if np.isnan(v):
                if last_valid is None:
                    new_val = toward
                else:
                    new_val = last_valid * (1 - alpha) + toward * alpha
                filled.append(new_val)
                last_valid = new_val
            else:
                filled.append(v)
                last_valid = v

        return filled




    def compare_missing_values_strategies_plot(self, df_interpolated, df_decay, features=["activity", "mood", "circumplex.arousal", "circumplex.valence"]):
        # Make sure 'truncated_time' is datetime
        df_interpolated["truncated_time"] = pd.to_datetime(df_interpolated["date"])
        df_decay["truncated_time"] = pd.to_datetime(df_decay["date"])

        #df_interpolated = df_interpolated.with_columns(pl.col("truncated_time").cast(pl.Datetime))
        #df_decay = df_decay.with_columns(pl.col("truncated_time").cast(pl.Datetime))

        # Get all unique IDs
        ids = df_interpolated["id"].drop_duplicates().sort_values().tolist()

        #ids = df_interpolated.select("id").unique().sort("id")["id"].to_list()

        for i in ids:
            
            df_interp_user = df_interpolated[df_interpolated["id"] == i].sort_values("truncated_time")
            df_decay_user = df_decay[df_decay["id"] == i].sort_values("truncated_time")


            df_interp_pd = df_interp_user.set_index("truncated_time")
            df_decay_pd = df_decay_user.set_index("truncated_time")


            plt.figure(figsize=(12, 6))

            for feature in features:
                plt.plot(df_interp_pd.index, df_interp_pd[feature], label=f"{feature} (Interp)", linestyle="--", marker="o")
                plt.plot(df_decay_pd.index, df_decay_pd[feature], label=f"{feature} (Decay)", linestyle="-", marker="x")
                


            plt.title(f"User {i}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


    def handle_missing_values_with_filling(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fill missing (NaN) values in a dataframe using the following strategy:
        1. Forward fill: If no previous non-null value exists, fill with the next available value.
        2. Backward fill: If no next non-null value exists, fill with the last available value.
        """
        # Ensure the DataFrame is sorted by 'id' and 'date'
        df = df.sort(by=["id", "date"])

        # List all columns to be filled (except for non-numeric ones like "id" and "date")
        cols_to_fill = [col for col in df.columns if col not in ["id", "date"]]

        # Apply forward fill first
        for col in cols_to_fill:
            df = df.with_columns([
                pl.col(col).fill_null(strategy="forward")
            ])

        # Then apply backward fill
        for col in cols_to_fill:
            df = df.with_columns([
                pl.col(col).fill_null(strategy="backward")
            ])

        return df

 


