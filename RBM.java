import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class RBM {
	private int numVisible;
	private int numHidden;
	private double[][] W; // Weights between visible and hidden
	private double[] b; // Visible biases
	private double[] c; // Hidden biases
	private Random random;

	// Constants for the Iris dataset
	private static final int NUM_VISIBLE_FEATURES = 12;
	private static final int NUM_OUTPUT = 3;
	private static final int NUM_VISIBLE = NUM_VISIBLE_FEATURES + NUM_OUTPUT;
	private static final int NUM_HIDDEN = 10; // You can adjust this value

	private RBM(int numVisible, int numHidden) {
		this.numVisible = numVisible;
		this.numHidden = numHidden;
		this.random = new Random();

		// Initialize weights with small random values (e.g., Gaussian)
		W = new double[numVisible][numHidden];
		for (int i = 0; i < numVisible; i++) {
			for (int j = 0; j < numHidden; j++) {
				W[i][j] = random.nextGaussian() * 0.01;
			}
		}

		// Initialize biases to zero
		b = new double[numVisible];
		c = new double[numHidden];

		// Initialize the weights and biases to small random values
		for (int i = 0; i < numVisible; i++) {
			b[i] = random.nextGaussian() * 0.01; // Visible Biases
		}

		for (int j = 0; j < numHidden; j++) {
			c[j] = random.nextGaussian() * 0.01; // Hidden Biases
		}

	}

	// Sigmoid function
	private double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	private InferenceResult inference(double[] v) {
		// 1. Calculate hidden unit probabilities
		double[] hProb = new double[numHidden];
		for (int j = 0; j < numHidden; j++) {
			double sum = c[j];
			for (int i = 0; i < numVisible; i++) {
				sum += v[i] * W[i][j];
			}
			hProb[j] = sigmoid(sum);
		}
		// 2. Sample hidden unit states
		double[] hSample = new double[numHidden];
		for (int j = 0; j < numHidden; j++) {
			hSample[j] = random.nextDouble() < hProb[j] ? 1.0 : 0.0;
		}
		// 3. Calculate visible unit probabilities (for reconstruction and output)
		double[] vProb = new double[numVisible];
		for (int i = 0; i < numVisible; i++) {
			double sum = b[i];
			for (int j = 0; j < numHidden; j++) {
				sum += hSample[j] * W[i][j];
			}
			vProb[i] = sigmoid(sum);
		}
		// 4. Determine predicted class based on the last 3 visible units
		int yPredicted = -1;
		double maxProb = -1.0;
		for (int k = 0; k < NUM_OUTPUT; k++) {
			if (vProb[NUM_VISIBLE_FEATURES + k] > maxProb) {
				yPredicted = k;
				maxProb = vProb[NUM_VISIBLE_FEATURES + k];
			}
		}
		return new InferenceResult(hProb, hSample, vProb, yPredicted);
	}

	// Inner class to hold the results of inference
	private class InferenceResult {
		private double[] hProb;
		private double[] hSample;
		private double[] vProb;
		private int yPredicted;

		private InferenceResult(double[] hProb, double[] hSample, double[] vProb, int yPredicted) {
			this.hProb = hProb;
			this.hSample = hSample;
			this.vProb = vProb;
			this.yPredicted = yPredicted;
		}
	}

	private static double[] discretizeDataSample(double[] sample, double[][] data) {
		double[] discretizedSample = new double[NUM_VISIBLE_FEATURES];
		int featureIndex = 0;

		for (int i = 0; i < NUM_VISIBLE_FEATURES / 3; i++) {
			double lowerThreshold = percentile(data, i, 33);
			double upperThreshold = percentile(data, i, 67);
			if (sample[i] <= lowerThreshold) {
				discretizedSample[featureIndex] = 1.0;
			} else if (sample[i] <= upperThreshold) {
				discretizedSample[featureIndex + 1] = 1.0;
			} else {
				discretizedSample[featureIndex + 2] = 1.0;
			}

			featureIndex += 3;
		}

		return discretizedSample;
	}

	private static double percentile(double[][] data, int featureIndex, double p) {
		double[] featureData = new double[data.length];
		for (int i = 0; i < data.length; i++) {
			featureData[i] = data[i][featureIndex];
		}
		Arrays.sort(featureData);
		int index = (int) Math.ceil((p / 100) * featureData.length) - 1;
		return featureData[index];
	}

	private void trainRBM(double[][] data, double learningRate, int epochs, int batchSize) {
		int numSamples = data.length;

		for (int epoch = 0; epoch < epochs; epoch++) {
			double epochError = 0;
			shuffleData(data);

			for (int i = 0; i < numSamples; i += batchSize) {
				int endIndex = Math.min(i + batchSize, numSamples);
				double[][] batch = Arrays.copyOfRange(data, i, endIndex);

				for (double[] sample : batch) {
					double[] fullSample = new double[numVisible];
					double[] discretizedFeatures = discretizeDataSample(sample, data);
					System.arraycopy(discretizedFeatures, 0, fullSample, 0, NUM_VISIBLE_FEATURES);

					int trueClass = (int) sample[4];
					fullSample[NUM_VISIBLE_FEATURES + trueClass] = 1.0;
					double[] v0 = fullSample;

					InferenceResult inferenceResult = inference(v0); // Calculate hidden neuron activation probability
					double[] h0Prob = inferenceResult.hProb;
//					double[] h0Sample = inferenceResult.hSample;

					double[] v1Prob = inferenceResult.vProb;
					double[] v1Sample = new double[numVisible];
					for (int n = 0; n < numVisible; n++) {
						v1Sample[n] = random.nextDouble() < v1Prob[n] ? 1.0 : 0.0;
					}
					// Calculate the activation probabilities of each neuron in the hidden layer
					InferenceResult inferenceResult1 = inference(v1Sample);
					double[] h1Prob = inferenceResult1.hProb;

					for (int n = 0; n < numVisible; n++) {
						for (int m = 0; m < numHidden; m++) {
							W[n][m] += learningRate * (v0[n] * h0Prob[m] - v1Sample[n] * h1Prob[m]);
						}
					}
					for (int n = 0; n < numVisible; n++) {
						b[n] += learningRate * (v0[n] - v1Sample[n]);
					}
					for (int m = 0; m < numHidden; m++) {
						c[m] += learningRate * (h0Prob[m] - h1Prob[m]);
					}
					double reconstructionError = 0;
					for (int n = 0; n < numVisible; n++) {
						reconstructionError += Math.pow(v0[n] - v1Prob[n], 2);
					}
					epochError += reconstructionError;
				}
			}

			System.out.println("Epoch " + (epoch + 1) + "/" + epochs + ", Reconstruction Error: "
					+ String.format("%.2f", epochError));
		}
	}

	private void shuffleData(double[][] data) {
		for (int i = data.length - 1; i > 0; i--) {
			int j = random.nextInt(i + 1);
			double[] temp = data[i];
			data[i] = data[j];
			data[j] = temp;
		}
	}

	private static double[][] getTargets(double[][] data) {
		double[][] targets = new double[data.length][];
		for (int i = 0; i < data.length; i++) {
			targets[i] = new double[1];
			targets[i][0] = data[i][4];
		}
		return targets;
	}

	private static double[][] loadIrisData(String urlString) {
		List<double[]> dataList = new ArrayList<>();
		try {
			URL url = new URL(urlString);
			BufferedReader br = new BufferedReader(new InputStreamReader(url.openStream()));
			String line;

			// Skip the title line
			br.readLine();

			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				double[] sample = new double[values.length];
				for (int i = 0; i < values.length - 1; i++) {
					sample[i] = Double.parseDouble(values[i]);
				}
				switch (values[values.length - 1]) {
				case "setosa":
					sample[values.length - 1] = 0;
					break;
				case "versicolor":
					sample[values.length - 1] = 1;
					break;
				case "virginica":
					sample[values.length - 1] = 2;
					break;
				}
				dataList.add(sample);
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		double[][] data = new double[dataList.size()][];
		for (int i = 0; i < dataList.size(); i++) {
			data[i] = dataList.get(i);
		}
		return data;
	}

	private static String getSpeciesName(double speciesIndex) {
		switch ((int) speciesIndex) {
		case 0:
			return "setosa";
		case 1:
			return "versicolor";
		case 2:
			return "virginica";
		default:
			return "Unknown";
		}
	}

	private static double[][] normalizeData(double[][] data) {
		double[][] normalizedData = new double[data.length][data[0].length];
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length - 1; j++) {
				normalizedData[i][j] = data[i][j] / 10.0;
			}
			// Leave the last column (species) as is
			normalizedData[i][data[i].length - 1] = data[i][data[i].length - 1];
		}
		return normalizedData;
	}

	private static void printWeights(double[][] W) {
		for (int i = 0; i < W.length; i++) {
			System.out.print("[ ");
			for (int j = 0; j < W[i].length; j++) {
				System.out.printf("%.4f ", W[i][j]); // Display the weights with 4 digits after the decimal point
			}
			System.out.println("]");
		}
	}

	private static void printBiases(double[] b, double[] c) {
		System.out.println("Visible Biases (b):");
		for (int i = 0; i < b.length; i++) {
			System.out.printf("b[%d] = %.4f\n", i, b[i]);
		}

		System.out.println("\nHidden Biases (c):");
		for (int j = 0; j < c.length; j++) {
			System.out.printf("c[%d] = %.4f\n", j, c[j]);
		}
	}

//	private static void printProbabilities(String label, double[] probArray) {
//		System.out.print(label + ": [");
//		for (int i = 0; i < probArray.length; i++) {
//			System.out.printf("%.4f", probArray[i]);
//			if (i < probArray.length - 1) {
//				System.out.print(", ");
//			}
//		}
//		System.out.println("]");
//	}

	private static void printArrayTwoDecimals(String label, double[] array) {
		String[] formatted = new String[array.length];
		for (int i = 0; i < array.length; i++) {
			// "%.2f" מציין שתי ספרות אחרי הנקודה
			formatted[i] = String.format("%.2f", array[i]);
		}
		System.out.println(label + ": " + Arrays.toString(formatted));
	}

	public static void main(String[] args) {
		String irisUrl = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv";
		double[][] data = loadIrisData(irisUrl);
		data = normalizeData(data);
		double[][] targets = getTargets(data);

		Random random = new Random();
		int randomIndex = random.nextInt(data.length);
		System.out.println("randomIndex = " + randomIndex);
//		for (int i = 95; i < 105; i++) {
//			System.out.println(Arrays.toString(data[i]));
//		}

		double[] sampleData = data[randomIndex];
		double[] sampleTarget = targets[randomIndex];

//		System.out.println("Random Sample Data: " + Arrays.toString(sampleData));
		printArrayTwoDecimals("Random Sample Data", sampleData);

		double[] discretizedSample = discretizeDataSample(sampleData, data);
		double[] fullSample = new double[discretizedSample.length + NUM_OUTPUT];
		System.arraycopy(discretizedSample, 0, fullSample, 0, discretizedSample.length);
		fullSample[NUM_VISIBLE_FEATURES + (int) sampleTarget[0]] = 1.0;

		RBM rbm = new RBM(NUM_VISIBLE, NUM_HIDDEN);
		System.out.println("RBM initialized.");
		// Print the initial weights
		System.out.println("Initial Weights (W):");
		printWeights(rbm.W);
		// Print the initial biases
//		printBiases(rbm.b, rbm.c);

		RBM.InferenceResult result = rbm.inference(fullSample);

//		System.out.println("Hidden Probabilities: \"%.4f \"" + Arrays.toString(result.hProb));
//		printProbabilities("Hidden Probabilities", result.hProb);
//		printProbabilities("Visible Probabilities", result.vProb);

		System.out.println("Hidden Sample: " + Arrays.toString(result.hSample));
		System.out.println("Predicted Species (before training): " + getSpeciesName(result.yPredicted));
		System.out.println("Random Sample True Species: " + getSpeciesName(sampleTarget[0]));

		rbm.trainRBM(data, 0.08, 100, 10);
		System.out.println("RBM trained.");
		RBM.InferenceResult resultAfterTraining = rbm.inference(fullSample);
		System.out.println("Final Weights (W):");
		printWeights(rbm.W);
		System.out.println("\nFinal Biases after training:");
		printBiases(rbm.b, rbm.c);

		System.out.println("Predicted Species (after training): " + getSpeciesName(resultAfterTraining.yPredicted));

	}
}
