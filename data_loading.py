import polars as pl
import pandas as pd
import os
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self):
        pass
    
    def load_and_transform_data(self, interval:str) -> pl.DataFrame:
        """
        Load the data from the specified path, transform it to a pivot table, and fill in missing dates.
        """
        # Load the data
        df = self.load_data()
        
        
        # Transform the data to a pivot table
        df_pivot = self.transform_data(df, interval)
        # Fill in missing dates, this method is skipped for now as rows that are missing for the mood are dropped
        # df_filled = self.fill_date_ranges(df_pivot, interval)

        # Handle missing values
        df_interpolated = self.handle_missing_values(df_pivot)
        
        return df_interpolated

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
        df = df.with_columns(pl.col("time").dt.truncate(truncation_string).alias("truncated_time"))
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
           
        # Sort the columns
        return df_pivot.select(["id", "truncated_time"] + [var for var in unique_variables])
    
    def fill_date_ranges(self, df: pl.DataFrame, interval:str) -> pl.DataFrame:
        """
        Fill the date ranges for each id in the DataFrame.

        NOTE: Since we drop the rows with missing values for the mood, this method becomes obsolete.
        I still keep it here as it might be useful in the future or for some data analysis.
        """
        date_range = pd.date_range(
            start=df["truncated_time"].min(),
            end=df["truncated_time"].max(),
            freq=interval,
        )
        # Create a DataFrame with all combinations of id and date
        id_range = df.select("id").unique()
        date_range_df = pl.DataFrame([
            pl.Series("truncated_time", date_range).cast(pl.Datetime("us"))
        ])
        # Cross join the id and date DataFrames
        cross_joined = id_range.join(date_range_df, how="cross")
        # Join the original DataFrame with the cross-joined DataFrame
        filled_df = cross_joined.join(df, on=["id", "truncated_time"], how="left")
        return filled_df
    
    def handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Handle missing values in the DataFrame. We will fill the numerical columns with 0 and interpolate the other columns in which
        zeros dont make sense.
        The columns that are filled with 0 are the sum columns from before, the columns that are interpolated are the mean columns from before.
        The mood column is the target variable and should not be filled with 0 or interpolated. Therefore we drop the rows with missing values for the mood.
        """
        # Firstly it does not make sense to fill missing values for the mood as this will be the target variable
        # Therefore we will drop the rows with missing values for the mood
        df = df.drop_nulls(subset=["mood"])

        # We now define the numerical columns for which we should do interpolation 
        # Note that these are the same as the mean columns from before (minus the mood column)
        cols_interpolate = ["circumplex.arousal", "circumplex.valence", "activity"]

        # Next we get the remaining columns for which we fill in 0, these are the sum columns from before
        cols = df.columns
        cols_fill_zero = list(set(cols) - set(cols_interpolate) - {"id", "truncated_time", "mood"})

        # Ensure the DataFrame is sorted properly for time-based interpolation
        df = df.sort(by=["id", "truncated_time"])
        # Fill 0s
        df_filled = df.with_columns([
            pl.col(col).fill_null(0) for col in cols_fill_zero
        ])

        # Interpolate within each panel group
        df_interpolated = (
            df_filled
            .group_by("id", maintain_order=True)
            .agg([
                pl.col("truncated_time"),
                *[pl.col(col).interpolate() for col in cols_interpolate],
                *[pl.col(col) for col in df.columns if col not in cols_interpolate and col not in ["truncated_time", "id"]],
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


        # Sort the DataFrame by id and truncated_time
        df_interpolated = df_interpolated.sort(by=["id", "truncated_time"])
        return df_interpolated
    
    def add_features(self, df: pl.DataFrame, id_col:str, time_col:str) -> pl.DataFrame:
        """
        Add features to the DataFrame. The features are:
        - time_since_last_obs: The time since the last observation in hours.
        """
        # Sort by id and time to ensure proper chronological order for each id
        df = df.sort([id_col, time_col])
        
        # Compute time_since_last_obs
        df = df.with_columns(
            ((pl.col(time_col) - pl.col(time_col).shift(1))/3600000000) # This division is to convert nanoseconds to hours, kinda ugly I know
           .over(id_col).cast(pl.Int64).fill_null(0).alias("time_since_last_obs")
        )
        
        return df
    
    def put_mood_into_buckets(self, df:pl.DataFrame, mood_interval:float) -> pl.DataFrame:
        """
        This function takes a DataFrame and a mood interval and puts the mood into buckets.
        The mood is divided into buckets of the specified interval.
        """ 
        # Define the mood buckets
        mood_buckets = [i * mood_interval for i in range(int(10/mood_interval) + 1)]
        def assign_bucket(val):
            return next((i for i in mood_buckets if val <= i), None)

        df = df.with_columns(
            pl.col("mood").map_elements(assign_bucket, return_dtype=pl.Float64).alias("mood")
        )
        return df
    



