package com.mtheory7;

import org.datavec.image.loader.CifarLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

// Code taken from...
// https://github.com/deeplearning4j/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/quickstart/modeling/convolution/CIFARClassifier.java

@SuppressWarnings("FieldCanBeLocal")
public class CIFARClassifier {
    protected static final Logger log = LoggerFactory.getLogger(CIFARClassifier.class);

    private static int height = 32;
    private static int width = 32;
    private static int channels = 3;
    private static int numLabels = CifarLoader.NUM_LABELS;
    private static int batchSize = 96;
    private static long seed = 123L;
    private static int epochs = 4;
    
    private static final String modelParentPath = "C:/Users/thesp/Code/deeplearning4j-cifar10/models";
    private static final String testImagePath = "C:/Users/thesp/Code/deeplearning4j-cifar10/testImages";
    private static final String dl4jStatsFileName = "ui-stats.dl4j";
    private static final String modelFileName = "cifarmodel.dl4j.zip";
    private static final String modelPath = modelParentPath + "/" + modelFileName;
    private static final String imagePath = testImagePath + "/test_image.png";

    private static final String[] LABELS = new String[] {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
    };

    public static void main(String[] args) throws Exception {
        log.error("Starting");
        CIFARClassifier cf = new CIFARClassifier();
        log.error("Finished creating cf object");
        Cifar10DataSetIterator cifar = new Cifar10DataSetIterator(batchSize, new int[]{height, width}, DataSetType.TRAIN, null, seed);
        log.error("Done creating cifar");
        Cifar10DataSetIterator cifarEval = new Cifar10DataSetIterator(batchSize, new int[]{height, width}, DataSetType.TEST, null, seed);
        log.error("Done creating cifarEval");

        MultiLayerNetwork model = cf.getModel();
        log.error("Done making model object");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(modelParentPath, dl4jStatsFileName));
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener( statsStorage), new ScoreIterationListener(50), new EvaluativeListener(cifarEval, 1, InvocationType.EPOCH_END));

        log.error("Beginning .fit command");
        model.fit(cifar, epochs);

        log.error("Saving model...");
        model.save(new File(modelParentPath, modelFileName), true);

        /* END OF MODEL TRAINING/GENERATION CODE --- BEGIN OF CODE TO TEST TRAINED/SAVED MODEL */

        try {
            log.error("Loading model...");
            File modelFile = new File(modelPath);
            MultiLayerNetwork loadedModel = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            log.error("Loading and preprocessing image...");
            File imgFile = new File(imagePath);

            NativeImageLoader loader = new NativeImageLoader(32, 32, 3);
            INDArray imageTensor = loader.asMatrix(imgFile);

            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(imageTensor);

            log.error("Running inference...");
            INDArray output = loadedModel.output(imageTensor, false);

            int predictedClassIndex = Nd4j.argMax(output, 1).getInt(0);
            double confidence = output.getDouble(predictedClassIndex) * 100;

            log.error("\n--- Prediction Result ---");
            System.out.printf("Predicted Class: %s (Index: %d)%n", LABELS[predictedClassIndex], predictedClassIndex);
            System.out.printf("Confidence: %.2f%%%n", confidence);
            log.error("Full probabilities: " + output);

        } catch (IOException e) {
            System.err.println("Error reading model or image file: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public MultiLayerNetwork getModel()  {
        log.error("Building simple convolutional network...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new AdaDelta())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nIn(channels).nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(16).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(128).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(numLabels).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.AVG).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .dropOut(0.8)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }
}