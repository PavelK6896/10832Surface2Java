package app.web.pavelk.surface2.service;

import app.web.pavelk.surface2.dto.Result;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

@Slf4j
@Service
@RequiredArgsConstructor
public class AppService {

    private final ImageAdaptService imageAdaptService;
    private MultiLayerNetwork multiLayerNetwork;

    @PostConstruct
    void init() throws IOException {
        multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("model/mnist-2.zip");
    }

    public Result images(MultipartFile multipartFile) throws IOException {
        BufferedImage bufferedImage = imageAdaptService.adapt(multipartFile);
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ImageIO.write(bufferedImage, "png", byteArrayOutputStream);
        InputStream inputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        INDArray indArray = nativeImageLoader.asMatrix(inputStream);
        INDArray predicted = multiLayerNetwork.output(indArray, false);
        return new Result(predicted.argMax(1).getInt(0));
    }
}
