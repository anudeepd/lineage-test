from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Spline Lineage Test") \
    .config("spark.jars", "./spark-3.5-spline-agent-bundle_2.12-2.2.3.jar") \
    .config("spark.sql.queryExecutionListeners", "za.co.absa.spline.harvester.listener.SplineQueryExecutionListener") \
    .config("spark.spline.lineageDispatcher.http.producer.url", "http://localhost:8080/producer") \
    .config("spark.spline.lineageDispatcher", "console") \
    .getOrCreate()

# Sample sales and customer data
sales_data = [
    (1, "Product_A", 100, 5, "2023-01-01"),
    (2, "Product_B", 200, 3, "2023-01-02"),
    (3, "Product_A", 150, 2, "2023-01-03"),
    (4, "Product_C", 300, 4, "2023-01-04"),
    (5, "Product_B", 250, 6, "2023-01-05")
]

customer_data = [
    (1, "John", "Premium"),
    (2, "Jane", "Standard"),
    (3, "Bob", "Premium"),
    (4, "Alice", "Standard"),
    (5, "Charlie", "Premium")
]

# Create DataFrames (Spline captures this)
sales_df = spark.createDataFrame(sales_data, ["order_id", "product", "amount", "quantity", "order_date"])
customer_df = spark.createDataFrame(customer_data, ["order_id", "customer_name", "tier"])

print("Source DataFrames created - Spline tracking lineage")

from pyspark.sql.functions import col, sum as spark_sum, avg, when

print("🔄 Executing Transformations...")

# Transformation 1: Product Revenue Calculation
product_revenue_df = sales_df.groupBy("product") \
    .agg(
        spark_sum("amount").alias("total_revenue"),
        avg("amount").alias("avg_order_value"),
        spark_sum("quantity").alias("total_quantity")
    )

# Transformation 2: Customer Enrichment with Discounts
enriched_df = sales_df.join(customer_df, "order_id", "inner") \
    .withColumn(
        "discounted_amount",
        when(col("tier") == "Premium", col("amount") * 0.9)
        .otherwise(col("amount"))
    )

# Transformation 3: Final Analytics by Product and Tier
final_analytics_df = enriched_df.groupBy("product", "tier") \
    .agg(
        spark_sum("discounted_amount").alias("tier_revenue"),
        avg("discounted_amount").alias("avg_tier_revenue"),
        spark_sum("quantity").alias("tier_quantity")
    ) \
    .orderBy("product", "tier")

print("Transformations executed - Spline captured lineage")

print("📊 RESULTS (Spline captures lineage on these actions):")

print("\n--- Product Revenue Summary ---")
product_revenue_df.show()

print("--- Customer Enriched Data Sample ---")
enriched_df.select("product", "customer_name", "tier", "amount", "discounted_amount").show(5)

print("--- Final Analytics by Product and Tier ---")
final_analytics_df.show()

# Write to parquet (Spline captures write lineage)
output_path = "./output/spline_analytics"
final_analytics_df.write.mode("overwrite").parquet(output_path)
print(f"📁 Data written to {output_path} - Spline captured write lineage")

spark.stop()
