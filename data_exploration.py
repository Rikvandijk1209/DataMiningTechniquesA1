import polars as pl
import os
from matplotlib import pyplot as plt

class DataExplorer:
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

    def get_aggregated_data_info(self, df: pl.DataFrame):
        """
        Get the information about the data.
        """
        # Get the shape of the data
        observations = df.shape[0]
        print(f"Number of observations: {observations}")
        # Get n_unique for certain columns
        columns = ["id", "time", "variable"]
        n_unique = {col: df[col].n_unique() for col in columns}
        print(f"Number of unique values: {n_unique}")
        # Get the number of null values for each column
        null_values = df.null_count()
        print(f"Number of null values: {null_values}")
        # Get the variables where value is null
        null_values_var = df.filter(pl.col("value").is_null()).select(pl.col("variable")).unique()
        print(f"The variables that contain null values: {null_values_var}")
        null_values_var_id = df.filter(pl.col("value").is_null()).select(pl.col("id"), pl.col("variable")).unique().sort(["id", "variable"])
        print(f"The variable and id combinations that contain null values: {null_values_var_id}")
        # Get the number of observations per id
        n_observations_per_id = df.group_by("id").agg(pl.count()).sort("id")
        print(f"Number of observations per id: {n_observations_per_id}")
        # Get the number of observations per variable
        n_observations_per_variable = df.group_by("variable").agg(pl.count()).sort("variable")
        print(f"Number of observations per variable: {n_observations_per_variable}")
        # Get the mean, min and max value per variable

        avg_value_per_variable = df.group_by("variable").agg([pl.mean("value").alias("mean"), pl.min("value").alias("min"), pl.max("value").alias("max")]).sort("variable")
        print(f"Average value per variable: {avg_value_per_variable}")
        # Get the number of observations per variable per id
        n_observations_per_variable_per_id = df.group_by(["id", "variable"]).agg(pl.count()).sort(["id", "variable"])
        n_observations_per_variable_per_id_pivot = n_observations_per_variable_per_id.pivot(values = "count", index = "id", columns = "variable")
        print(f"Number of observations per variable per id: {n_observations_per_variable_per_id_pivot}")
        return {
            "observations": observations,
            "n_unique": n_unique,
            "null_values": null_values_var,
            "null_values_var": null_values_var,
            "null_values_var_id": null_values_var_id,
            "n_observations_per_id": n_observations_per_id,
            "n_observations_per_variable": n_observations_per_variable,
            "avg_value_per_variable": avg_value_per_variable,
            "n_observations_per_variable_per_id": n_observations_per_variable_per_id_pivot,
        }
    
    def get_aggregated_data_plots(self, df: pl.DataFrame):
        df_pd = df.to_pandas()
        # Plot number of observations per id in histogram
        plt.figure(figsize=(10, 6))
        counts, bins, patches = plt.hist(df_pd["id"], bins=len(df_pd["id"].unique()))
        plt.title("Histogram of ID observation counts")
        plt.xlabel("ID")
        plt.ylabel("Count")
        # Get the positions of the bars (bin centers)
        bin_centers = [patch.get_x() + patch.get_width() / 2 for patch in patches]
        # Set x-ticks at the center of each bin
        plt.xticks(bin_centers, rotation=45)
        plt.show()

        # Plot number of observations per variable in histogram
        plt.figure(figsize=(10, 6))
        counts, bins, patches = plt.hist(df_pd["variable"].sort_values(), bins=len(df_pd["variable"].unique()))
        plt.title("Histogram of variable observation counts")
        plt.xlabel("Variable")
        plt.ylabel("Count")
        # Get the positions of the bars (bin centers)
        bin_centers = [patch.get_x() + patch.get_width() / 2 for patch in patches]
        # Set x-ticks at the center of each bin
        plt.xticks(bin_centers, rotation=90)
        plt.show()

        # Get the distribution plots of all variables
        unique_var = df["variable"].unique().sort().to_list()
        for var in unique_var:
            df_pd_var = df.filter(pl.col("variable") == var).to_pandas()
            # Plot the distribution of the variable
            plt.figure(figsize=(10, 6))
            plt.hist(df_pd_var["value"], bins=20)
            plt.title(f"Histogram of {var} values")
            plt.xlabel(var)
            plt.ylabel("Count")
            plt.show()


    def transform_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform the data from long format to a pivot table. Aggregate the data on date level with value aggregation by specified format.
        The interval of dates was used with 1 day as the eventual target is to predict mood per day.
        """
        # Aggregate on date (day) level
        df = df.with_columns(pl.col("time").dt.date().alias("date"))
        # Here we apply all necessary transformations to later select per variable which to use
        df_agg = df.group_by(["id", "date", "variable"]).agg([
            pl.col("value").sum().alias("value_sum"), 
            pl.col("value").mean().alias("value_mean"),
            ]).sort(["id", "date", "variable"])
        
        df_pivot:pl.DataFrame = df_agg.pivot(
            values=["value_sum", "value_mean"],
            index = ["id", "date"],
            columns = "variable",
        )
        # Specify for which variables we should use the mean, the others will use the sum
        mean_col = ["mood", "cirumplex.arousal", "circumplex.valence", "activity"]
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
        return df_pivot.select(["id", "date"] + [var for var in unique_variables])
                

        