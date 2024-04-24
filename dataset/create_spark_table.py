"""
Follow [this page](https://spark.apache.org/docs/3.5.0/api/python/getting_started/install.html) to
install PySpark on your system.

"""
import os
import os.path as osp
import string
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.types import StructField, TimestampType, StringType, StructType

import const
from utility.utils_data import load_arXiv_data

sys.path.insert(0, os.path.abspath('..'))
from arguments import parse_args
from utility.utils_misc import project_setup


def find_papers_by_tag(sdf, tag):
    return sdf.filter(sdf.tag == tag).collect()


def create_or_load_spark_dataframe(args):
    path = osp.join(args.data_dir, "arXiv_spark_table.parquet")
    # Initialize a Spark session
    spark = SparkSession.builder \
        .appName("arXiv Metadata") \
        .getOrCreate()

    print("spark.executor.memory", spark.conf.get("spark.executor.memory", "Not set"))
    print("spark.driver.memory", spark.conf.get("spark.driver.memory", "Not set"))

    if osp.exists(path):
        sdf = spark.read.parquet(osp.join(args.data_dir, "arXiv_spark_table.parquet"))

    else:

        schema = StructType([
            StructField("id", StringType(), True),
            StructField("title", StringType(), True),
            StructField("summary", StringType(), True),
            StructField("arxiv_comment", StringType(), True),
            StructField("published", TimestampType(), True),  # Specify TimestampType for "published"
            StructField("updated", TimestampType(), True),  # Specify TimestampType for "updated"
            StructField("authors", StringType(), True),
            StructField("tags", StringType(), True)
        ])

        df = load_arXiv_data(args.data_dir)

        """
        Convert the tags field from a string to a list of tags:
        df["tags"] = df["tags"].apply(lambda x: x.strip(string.punctuation).split('\', \'') if isinstance(x, 
        str) else x) 
        """

        # "tags" is read as a string, so we convert it to a list
        df["tags"] = df["tags"].apply(
            lambda x: [tag for tag in x if tag in
                       const.ARXIV_SUBJECTS_LIST])

        sdf = spark.createDataFrame(df, schema=schema)

        # Explode the 'tags' list to create new rows for each tag
        sdf_exploded = sdf.withColumn("tag", explode(sdf.tags))

        # Cache the DataFrame for faster query performance
        sdf_exploded.cache()

        sdf_exploded.write.parquet(path, mode="overwrite")

    return sdf


if __name__ == "__main__":
    project_setup()
    args = parse_args()
    sdf = create_or_load_spark_dataframe(args)

    papers_with_tag = find_papers_by_tag(sdf, "math.NT")
    for paper in papers_with_tag:
        print(paper)

    spark.stop()
