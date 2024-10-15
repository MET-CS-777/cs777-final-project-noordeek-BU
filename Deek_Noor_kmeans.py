from __future__ import print_function
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql import functions as F


######################## Pre-data work such as function-building for row correction and k-means clustering #########################


sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

# Checks if the values can be converted to a float
def isfloat(value):
    if value is None or value == '':
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False

def correctRows(row):
    return row['tic'] is not None and \
           row['cusip'] is not None and \
           row['fyearq'] is not None and \
           row['fqtr'] is not None and \
           isfloat(row['atq']) and \
           isfloat(row['dlttq']) and \
           isfloat(row['ltq']) and \
           isfloat(row['niq']) and \
           isfloat(row['revtq']) and \
           isfloat(row['cshoq']) and \
           isfloat(row['prccq']) and \
           float(row['cshoq']) > 0.5 and \
           float(row['prccq']) > 0

# Function to run k-means clustering
def run_kmeans(df, feature_column, k=3):
    kmeans = KMeans(k=k, seed=1, featuresCol=feature_column).setMaxIter(100).setTol(1e-6)
    model = kmeans.fit(df)
    
    df_clustered = model.transform(df)
    cluster_centers = model.clusterCenters()
    print(f"Cluster Centers: {cluster_centers}")
    
    return df_clustered


#################### Begins the main portion of my code ###############################

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: kmeans_clustering.py <input_file>", file=sys.stderr)
        sys.exit(-1)

    # Reads the data via command-line argument
    input_file = sys.argv[1]
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_file)


    ########################## Begin the data cleaning process #################################

    # Apply the cleaning function and filter rows where cshoq and prccq are zero
    # Calculate man_mkt_val as cshoq * prccq and add it as a new column (the dataset's market value column was inaccurate and didn't appropriately display market cap)
    # Remove rows with null values in the feature columns
    df_cleaned = df.rdd.filter(lambda row: correctRows(row)).toDF()
    df_cleaned = df_cleaned.withColumn("man_mkt_val", F.col("cshoq") * F.col("prccq"))


    ############################ Assemble features, Convert RDDs, and other data prep ###################

    feature_columns = ['atq', 'dlttq', 'ltq', 'niq', 'revtq', 'prccq', 'prchq', 'prclq']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features_unscaled", handleInvalid="skip")
    df_features = assembler.transform(df_cleaned)
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_features)
    df_scaled = scaler_model.transform(df_features)

    # Repartitioning the data to avoid partition mismatch errors
    df_repartitioned = df_scaled.repartition(10)

    # Filtered out any null values
    df_filtered = df_repartitioned.na.drop()


    #################### Running k-means clustering ##########################
    print("Running K-means clustering...")
    df_clustered = run_kmeans(df_filtered, "features", k=3)

    # Show only the featured columns after clustering + the ticker (labeled 'tic') + which cluster said row was assigned to
    df_clustered.select("tic", "atq", "dlttq", "ltq", "niq", "revtq", "prccq", "prchq", "prclq", "man_mkt_val", "prediction").show()

    sc.stop()
