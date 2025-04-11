import polars as pl
import pandas as pd
import os
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self):
        pass

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
