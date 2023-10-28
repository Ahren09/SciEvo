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

import const
from utility.utils_data import load_data

sys.path.insert(0, osp.join(os.getcwd(), "src"))

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

    if osp.exists(path):
        sdf = spark.read.parquet(osp.join(args.data_dir, "arXiv_spark_table.parquet"))

    else:
        df = load_data(args)

        # "tags" is read as a string, so we convert it to a list
        df["tags"] = df["tags"].apply(lambda x: x.strip(string.punctuation).split('\', \'')).apply(
            lambda x: [tag for tag in x if tag in
                       const.ARXIV_SUBJECTS_LIST])

        sdf = spark.createDataFrame(df)

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
