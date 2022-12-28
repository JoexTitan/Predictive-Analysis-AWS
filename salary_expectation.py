slr_expectation_emr
from pyspark.sql.dataframe import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession

def get_data(sc: SparkSession, file: str):
    return sc.read.csv(file, header='true', sep=',', inferSchema=False)

def table_join(features: DataFrame, response: DataFrame, on: str) -> DataFrame:
    full = features.join(response, on)
    return full

def spark_sess(app_name: str) -> SparkSession:
    ss = SparkSession.builder.master('local').appName(app_name).getOrCreate()
    return ss

def get_feature_avg(df: DataFrame, feature: str) -> DataFrame:
    return df.groupby(feature) \
        .agg(round(avg('slr'), 4).alias('find_avg_salary')) \
        .sort('find_avg_salary', ascending=False)


def output_result(DataFrame, output_location, output_folder):
    df.coalesce(1).write.csv(path=output_location + output_folder,
                             mode='append', header=True)

if __name__ == '__main__':
    spark = create_spark_session("find avg salary")
    train_features_df = read_in_data(spark, 's3://destination_path')
    train_salaries_df = read_in_data(spark, 's3://destination_path')
    train_df = join_tables(train_features_df, train_salaries_df, 'jobId')
    avg_sal_by_companyId = avg_by_feature(train_df, 'companyId')
    avg_sal_by_industry = avg_by_feature(train_df, 'industry')
    avg_sal_by_jobType = avg_by_feature(train_df, 'jobType')
    avg_sal_by_degree = avg_by_feature(train_df, 'degree')
    avg_sal_by_major = avg_by_feature(train_df, 'major')
    OUTPUT_LOCATION = 's3://destination_path'
    output_result(avg_sal_by_major, OUTPUT_LOCATION, 'major')
    output_result(avg_sal_by_industry, OUTPUT_LOCATION, 'industry')
    output_result(avg_sal_by_companyId, OUTPUT_LOCATION, 'companyId')
    output_result(avg_sal_by_jobType, OUTPUT_LOCATION, 'jobType')
    output_result(avg_sal_by_degree, OUTPUT_LOCATION, 'degree')
    spark.stop()