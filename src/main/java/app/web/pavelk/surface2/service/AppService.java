package app.web.pavelk.surface2.service;

import app.web.pavelk.surface2.dto.ResultImages;
import app.web.pavelk.surface2.dto.ResultSave;
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
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class AppService {

    private final ImageAdaptService imageAdaptService;
    private MultiLayerNetwork multiLayerNetwork;

    @PostConstruct
    void init() throws IOException {
        multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("model/mnist-3.zip");
        for (int i = 0; i < 10; i++) {
            Files.createDirectories(Paths.get("new/" + i));
        }
    }

    public ResultImages images(MultipartFile multipartFile) throws IOException {
        BufferedImage bufferedImage = imageAdaptService.adapt(multipartFile);
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ImageIO.write(bufferedImage, "png", byteArrayOutputStream);
        InputStream inputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        INDArray indArray = nativeImageLoader.asMatrix(inputStream);
        INDArray predicted = multiLayerNetwork.output(indArray, false);
        return new ResultImages(predicted.argMax(1).getInt(0));
    }

    public ResultSave save(String right, MultipartFile multipartFile) throws IOException {
        BufferedImage bufferedImage = imageAdaptService.adapt(multipartFile);
        String uuid = UUID.randomUUID().toString();
        imageAdaptService.save(bufferedImage, right + "/" + uuid);
        return new ResultSave(uuid, right);
    }
}
