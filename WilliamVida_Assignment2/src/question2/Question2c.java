package question2;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;

public class Question2c {

	public static void main(String[] args) throws IOException {
		// Configure Spark.
		SparkConf sparkConf = new SparkConf().setAppName("Question2").setMaster("local[4]").set("spark.executor.memory",
				"2g");

		try (JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf)) {
			// ArrayList of String arrays for the reviews and response.
			List<String[]> reviews = new ArrayList<String[]>();

			// Parse the CSV file using the first row as columns and skipping it.
			// https://medium.com/@zakariafarih142/mastering-csv-parsing-in-java-comprehensive-methods-and-best-practices-a3b8d0514edf
			try (Reader reader = new FileReader("./src/question2/hotel-reviews.csv")) {
				Iterable<CSVRecord> rows = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).get()
						.parse(reader);

				// Add each row's description and response to the ArrayList as an array.
				for (CSVRecord row : rows) {
					String description = row.get("Description");
					String response = row.get("Is_Response");
					reviews.add(new String[] { description, response });
				}
			}

			// Convert the ArrayList to a JavaRDD.
			// https://stackoverflow.com/a/24953895
			JavaRDD<String[]> hotelReviewsJavaRDD = javaSparkContext.parallelize(reviews);
			final HashingTF tf = new HashingTF(100);

			JavaRDD<LabeledPoint> labelledPointData = hotelReviewsJavaRDD.map(review -> {
				String description = review[0];
				String response = review[1];

				// Set the values of the labels depending on the response.
				double label = 0.0;
				if (response.equalsIgnoreCase("happy")) {
					label = 1.0;
				} else if (response.equalsIgnoreCase("not happy")) {
					label = 0.0;
				}

				// Convert the review to lowercase and split it into words.
				// https://stackoverflow.com/a/22433520
				return new LabeledPoint(label, tf.transform(Arrays.asList(description.toLowerCase().split("\\W+"))));
			});

			// Randomly split the dataset into 60% training data and 40% testing data.
			double weights[] = { 0.6, 0.4 };
			JavaRDD<LabeledPoint>[] splitsJavaRDD = labelledPointData.randomSplit(weights);
			JavaRDD<LabeledPoint> trainingData = splitsJavaRDD[0];
			JavaRDD<LabeledPoint> testingData = splitsJavaRDD[1];

			// Cache.
			trainingData.cache();

			// Train the Logistic Regression model.
			LogisticRegressionWithSGD lrLearner = new LogisticRegressionWithSGD(1.0, 100, 0.01, 1.0);
			LogisticRegressionModel lrModel = lrLearner.run(trainingData.rdd());

			// Compare the model's predictions with the actual labels of five randomly
			// sampled test reviews.
			System.out.println("Logistic Regression Model predictions");
			List<LabeledPoint> testReviews = testingData.takeSample(false, 5);
			for (LabeledPoint labelledPoint : testReviews) {
				double testLabel = labelledPoint.label();
				double predict = lrModel.predict(labelledPoint.features());

				System.out.println("========================================");
				System.out.println("Testing data label   " + testLabel);
				System.out.println("Logistic Regression Model prediction " + predict);
				System.out.println("========================================");
			}
		}
	}
}
