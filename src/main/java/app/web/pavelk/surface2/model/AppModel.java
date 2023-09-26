package app.web.pavelk.surface2.model;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Slf4j
public class AppModel {

    public static void main(String[] args) throws IOException {
        test();
    }

    public static void create() throws IOException {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nIn(800)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(500)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        ModelSerializer.writeModel(model, new File("model/mnist-1.zip"), true);
        log.info("create model");
    }

    public static void train() throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("model/mnist-2.zip");

        DataSetIterator mnistTrain = new MnistDataSetIterator(128, true, 123);
        DataSetIterator mnistTest = new MnistDataSetIterator(128, false, 123);
        model.setListeners(new ScoreIterationListener(10), new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END));
        for (int i = 0; i < 2; i++) {
            model.fit(mnistTrain);
            log.info("epoch {}", i);
        }
        Evaluation eval = model.evaluate(mnistTest);
        log.info(eval.stats());
        ModelSerializer.writeModel(model, new File("model/mnist-2.zip"), true);
    }

    public static void retrain() throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("model/mnist-2.zip");

        int height = 28;
        int width = 28;
        int channels = 1;
        int outputNum = 10;
        int batchSize = 1;

        Random randNumGen = new Random(1234);

        FileSplit mew = new FileSplit(new File("new"), NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(new File("../mnist_png/testing"), NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit training = new FileSplit(new File("../mnist_png/training"), NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader imageRecordReader1 = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader imageRecordReader2 = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader imageRecordReader3 = new ImageRecordReader(height, width, channels, labelMaker);

        imageRecordReader1.initialize(mew);
        DataSetIterator dataSetIterator1 = new RecordReaderDataSetIterator(imageRecordReader1, batchSize, 1, outputNum);
        imageRecordReader2.initialize(test);
        DataSetIterator dataSetIterator2 = new RecordReaderDataSetIterator(imageRecordReader2, batchSize, 1, outputNum);
        imageRecordReader3.initialize(training);
        DataSetIterator dataSetIterator3 = new RecordReaderDataSetIterator(imageRecordReader3, batchSize, 1, outputNum);

        DataNormalization dataNormalization1 = new ImagePreProcessingScaler();
        DataNormalization dataNormalization2 = new ImagePreProcessingScaler();
        DataNormalization dataNormalization3 = new ImagePreProcessingScaler();
        dataNormalization1.fit(dataSetIterator1);
        dataNormalization2.fit(dataSetIterator2);
        dataNormalization3.fit(dataSetIterator3);
        dataSetIterator1.setPreProcessor(dataNormalization1);
        dataSetIterator2.setPreProcessor(dataNormalization2);
        dataSetIterator3.setPreProcessor(dataNormalization3);


        List<DataSet> list = new ArrayList<>();
        int id = 0;
        while (dataSetIterator1.hasNext()) {
            list.add(dataSetIterator1.next());
            id++;
        }
        while (dataSetIterator2.hasNext()) {
            list.add(dataSetIterator2.next());
            id++;
        }
        while (dataSetIterator3.hasNext()) {
            list.add(dataSetIterator3.next());
            id++;
        }
        log.info("{}", id);
        ListDataSetIterator iterator = new ListDataSetIterator(list, id);
        log.info("list size {} {}", list.size(), iterator.totalOutcomes());


        DataSet dataSetNew = iterator.next();
        dataSetNew.shuffle();
        SplitTestAndTrain testAndTrain = dataSetNew.splitTestAndTrain(0.9);

        DataSet trainSet = testAndTrain.getTrain();
        DataSet testSet = testAndTrain.getTest();
        log.info("trainSet {} testSet {}", trainSet.numExamples(), testSet.numExamples());


        DataSetIterator trainSetIterator = new ListDataSetIterator(trainSet.asList(), 128);
        DataSetIterator testSetIterator = new ListDataSetIterator(testSet.asList(), 128);


        model.setListeners(new ScoreIterationListener(1));

        EarlyStoppingModelSaver saver = new InMemoryModelSaver();
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(35)) //Max of 50 epochs
                .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(5))
                .evaluateEveryNEpochs(1).scoreCalculator(new DataSetLossCalculator(testSetIterator, true)) //Calculate test set score
                .modelSaver(saver)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainSetIterator);
        EarlyStoppingResult result = trainer.fit();
        log.info("Termination reason: " + result.getTerminationReason());
        log.info("Termination details: " + result.getTerminationDetails());
        log.info("Total epochs: " + result.getTotalEpochs());
        log.info("Best epoch number: " + result.getBestModelEpoch());
        log.info("Score at best epoch: " + result.getBestModelScore());


        Model bestModel = saver.getBestModel();
        ModelSerializer.writeModel(bestModel, new File("model/mnist-3.zip"), true);


    }
    public static void test() throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("model/mnist-3.zip");
        DataSetIterator mnistTest = new MnistDataSetIterator(128, false, 123);
        Evaluation eval = model.evaluate(mnistTest);
        log.info(eval.stats());
    }

}
