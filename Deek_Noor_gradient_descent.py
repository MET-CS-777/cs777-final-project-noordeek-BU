from __future__ import print_function
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F
from pyspark.ml.linalg import DenseVector


######################## Pre-data work such as function-building for row correction and gradient descent #########################

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

# Data cleaning function to filter valid rows based on selected columns
# For 'cshoq' - common shares outstanding (per million), only want companies with over 500k shares outstanding, to ensure I'm getting semi-quality balance sheet data
# for 'prccq' - price close, want companies with greater than $0 price close. Willing to accept any value other than 0, as some great startups started as penny stocks
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

# Function to calculate gradient descent with multiple variables
# implemented a tolerance into the function as minor changes to the learning rate caused wonky bold-driver responses
def gradient_descent_mult(features_rdd, target_rdd, num_iterations=50, learning_rate=0.5, tol=1e-6):
    weights = np.zeros(features_rdd.first().size)
    intercept = 0.0

    bold_driver_increase = 1.05
    bold_driver_decrease = 0.7
    max_learning_rate = 10.0
    min_learning_rate = 0.001
    prev_loss = None

    # Caching the RDDs to avoid recomputation in each iteration
    features_rdd.cache()
    target_rdd.cache()

    # Combining features and target RDDs
    combined_rdd = features_rdd.zip(target_rdd)

    for i in range(num_iterations):
        def compute_gradient(iterator):
            grad_sum = np.zeros(weights.size)
            intercept_grad_sum = 0.0
            loss_sum = 0.0
            count = 0

            for (features, target) in iterator:
                features = np.array(features)
                prediction = np.dot(features, weights) + intercept
                error = prediction - target
                grad_sum += features * error
                intercept_grad_sum += error
                loss_sum += error ** 2
                count += 1

            return [(grad_sum, intercept_grad_sum, loss_sum, count)]

        grad_loss_rdd = combined_rdd.mapPartitions(compute_gradient)
        grad_sum, intercept_grad_sum, loss_sum, count = grad_loss_rdd.reduce(
            lambda x, y: (x[0] + y[0],
                          x[1] + y[1],
                          x[2] + y[2],
                          x[3] + y[3])
        )

        gradient = grad_sum / count
        intercept_grad = intercept_grad_sum / count
        prev_weights = weights.copy()
        prev_intercept = intercept

        weights -= learning_rate * gradient
        intercept -= learning_rate * intercept_grad
        loss = loss_sum / count

        if prev_loss is not None:
            if loss < prev_loss - tol:
                learning_rate = min(learning_rate * bold_driver_increase, max_learning_rate)
                print(f"Iteration {i + 1}: Loss improved, increasing learning rate to {learning_rate}")
            elif loss > prev_loss + tol:
                learning_rate = max(learning_rate * bold_driver_decrease, min_learning_rate)
                weights = prev_weights
                intercept = prev_intercept
                print(f"Iteration {i + 1}: Loss worsened, decreasing learning rate to {learning_rate}")
        else:
            print(f"Iteration {i + 1}: Initial iteration")

        prev_loss = loss

        print(f"Iteration {i + 1}: Loss = {loss}, Learning Rate = {learning_rate}")

    return weights, intercept, prev_loss, learning_rate


# Begins the main portion of my code
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Deek_Noor_gradient_descent.py <input_file>", file=sys.stderr)
        sys.exit(-1)

    # Sets up argument for the CSV input file + read input file
    input_file = sys.argv[1]
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_file)




    ########################## Begin the data cleaning process #################################



    # Count the total number of rows before cleaning (will use to calcuate how many rows removed)
    total_rows_before = df.count()

    # Apply the cleaning function and filter rows where cshoq and prccq are zero
    # Calculate man_mkt_val as cshoq * prccq and add it as a new column (the dataset's market value column was inaccurate and didn't appropriately display market cap)
    # Remove rows with null values in the feature columns

    df_cleaned = df.rdd.filter(lambda row: correctRows(row)).toDF()
    df_cleaned = df_cleaned.withColumn("man_mkt_val", F.col("cshoq") * F.col("prccq"))
    feature_columns = ['atq', 'dlttq', 'ltq', 'niq', 'revtq', 'prccq', 'prchq', 'prclq']
    df_cleaned = df_cleaned.na.drop(subset=feature_columns)


    # Count the number of rows after cleaning
    total_rows_after = df_cleaned.count()
    excluded_rows = total_rows_before - total_rows_after

    # Print how many rows removed
    print(f"Total rows before cleaning: {total_rows_before}")
    print(f"Total rows after cleaning: {total_rows_after}")
    print(f"Number of rows excluded during cleaning: {excluded_rows}")

    # Show the cleaned and filtered data
    df_cleaned.show(10)




    ############################ Assemble features, Convert RDDs, and other data prep ###################




    # Assemble my assessed features into vector + standardize features
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features_unscaled", handleInvalid="skip")
    df_features = assembler.transform(df_cleaned)
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_features)
    df_scaled = scaler_model.transform(df_features)

    # Convert the features and target columns into RDDs
    features_rdd = df_scaled.select(F.col("features")).rdd.map(lambda x: x[0].toArray())
    target_rdd = df_scaled.select(F.col("man_mkt_val")).rdd.map(lambda x: x[0])


    #################### Gradient Descent Iterations ##########################
    print("Starting gradient descent with multiple variables...")
    weights, intercept, final_cost, final_learning_rate = gradient_descent_mult(features_rdd, target_rdd)

    print("Final weights:", weights)
    print("Final intercept:", intercept)
    print(f"Final cost: {final_cost}")
    print(f"Final learning rate: {final_learning_rate}")



    ############################# Data evaluation and interpretation #########################



    # Returning regression-related values for each feature, as well as the intercept, final cost, and learning rate
    results_3 = [
        f"Intercept (b) (Man-Made Market Value): {intercept}",
        f"Slope m1 (Assets - Total (atq)): {weights[0]}",
        f"Slope m2 (Long-Term Debt - Total (dlttq)): {weights[1]}",
        f"Slope m3 (Liabilities - Total (ltq)): {weights[2]}",
        f"Slope m4 (Net Income (niq)): {weights[3]}",
        f"Slope m5 (Revenue - Total (revtq)): {weights[4]}",
        f"Slope m6 (Price Close - Quarter (prccq)): {weights[5]}",
        f"Slope m7 (Price High - Quarter (prchq)): {weights[6]}",
        f"Slope m8 (Price Low - Quarter (prclq)): {weights[7]}",
        f"Final Cost: {final_cost}",
        f"Final Learning Rate: {final_learning_rate}"
    ]

    # Collecting and printing the average market value for all cleaned rows in the dataset, so we have a basis of comparison
    avg_market_value = df_cleaned.select(F.avg(F.col("man_mkt_val"))).collect()[0][0]
    print(f"The dataset's average market value is (in millions): {avg_market_value}")

    # Printing the regression-related values
    for result in results_3:
        print(result)

    sc.stop()
