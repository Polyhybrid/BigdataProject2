from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# 1. สร้าง Spark Session
spark = SparkSession.builder \
    .appName("MovieLens Recommendation System") \
    .getOrCreate()

# 2. โหลดข้อมูล
ratings_df = spark.read.csv("ratings.csv", header=True, inferSchema=True)
movies_df = spark.read.csv("movies.csv", header=True, inferSchema=True)

# 3. แบ่งข้อมูลสำหรับการ Train และ Test (80/20)
(training, test) = ratings_df.randomSplit([0.8, 0.2], seed=42)

# 4. สร้างและ Train โมเดล ALS
als = ALS(maxIter=10, 
          regParam=0.1, 
          userCol="userId", 
          itemCol="movieId", 
          ratingCol="rating",
          coldStartStrategy="drop") # Drop user/movie ที่ไม่มีใน training set ตอน test
model = als.fit(training)

# 5. ประเมินผลโมเดล (Evaluation)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", 
                                labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error (RMSE) = {rmse}")

# 6. แนะนำภาพยนตร์ Top 5 สำหรับผู้ใช้ทุกคน
userRecs = model.recommendForAllUsers(5)

# นำผลลัพธ์มา Join กับ movies_df เพื่อแสดงชื่อหนังให้ดูง่ายขึ้น
userRecs.show(5, truncate=False)

# หยุดการทำงานของ Spark
spark.stop()