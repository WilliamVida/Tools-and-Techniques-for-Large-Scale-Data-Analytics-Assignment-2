package question2;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class Question2b {

	public static void main(String[] args) {
		// Configure Spark.
		SparkConf sparkConf = new SparkConf().setAppName("Question2").setMaster("local[4]").set("spark.executor.memory",
				"2g");

		try (JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf)) {
			JavaRDD<String> imdbJavaRDD = javaSparkContext.textFile("./src/question2/imdb_labelled.txt");
			final HashingTF tf = new HashingTF(100);

			// Convert each review into a LabeledPoint RDD containing its label and a
			// feature vector created from the review text.
			JavaRDD<LabeledPoint> labelledPointData = imdbJavaRDD.map(review -> {
				String feature[] = review.split("\\t");
				double label = Double.parseDouble(feature[1]);

				// Convert the review to lowercase and split it into words.
				// https://stackoverflow.com/a/22433520
				return new LabeledPoint(label, tf.transform(Arrays.asList(feature[0].toLowerCase().split("\\W+"))));
			});

			// Randomly split the dataset into 60% training data and 40% testing data.
			double weights[] = { 0.6, 0.4 };
			JavaRDD<LabeledPoint>[] splitsJavaRDD = labelledPointData.randomSplit(weights);
			JavaRDD<LabeledPoint> trainingData = splitsJavaRDD[0];
			JavaRDD<LabeledPoint> testingData = splitsJavaRDD[1];

			// Cache.
			trainingData.cache();

			// Train the SVM model.
			SVMModel svmModel = SVMWithSGD.train(trainingData.rdd(), 100, 0.01, 1.0);

			// Compare the model's predictions with the actual labels of five randomly
			// sampled test reviews.
			System.out.println("SVM Model predictions");
			List<LabeledPoint> testReviews = testingData.takeSample(false, 5);
			for (LabeledPoint labelledPoint : testReviews) {
				double testLabel = labelledPoint.label();
				double predict = svmModel.predict(labelledPoint.features());

				System.out.println("========================");
				System.out.println("Testing data label   " + testLabel);
				System.out.println("SVM Model prediction " + predict);
				System.out.println("========================");
			}

			// Question 2b.
			JavaPairRDD<Object, Object> predictionAndLabels = testingData
					.mapToPair(p -> new Tuple2<>(svmModel.predict(p.features()), p.label()));
			BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabels.rdd());
			System.out.println("Area under ROC = " + metrics.areaUnderROC());

			javaSparkContext.stop();
		}
	}
}
