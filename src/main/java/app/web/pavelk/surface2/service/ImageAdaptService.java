package app.web.pavelk.surface2.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

@Slf4j
@Service
@RequiredArgsConstructor
public class ImageAdaptService {

    public void save(MultipartFile multipartFile, String name) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(name + ".png");
        fileOutputStream.write(multipartFile.getBytes());
        fileOutputStream.flush();
        fileOutputStream.close();
    }

    public void save(BufferedImage bufferedImage, String string) throws IOException {
        ImageIO.write(bufferedImage, "png", new File("new/" + string + ".png"));
    }

    public BufferedImage adapt(MultipartFile multipartFile) throws IOException {
        BufferedImage originalImage = ImageIO.read(multipartFile.getInputStream());
        BufferedImage bufferedImage = resizeImage(originalImage, 28, 28);
        bufferedImage = invertBlackAndWhite(bufferedImage);
        return bufferedImage;
    }

    public BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        Image resultingImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_AREA_AVERAGING);
        BufferedImage outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_ARGB);
        Graphics2D graphics = outputImage.createGraphics();
        Color color = new Color(250, 250, 250, 250);
        graphics.setColor(color);
        graphics.fillRect(0, 0, targetWidth, targetHeight);
        graphics.drawImage(resultingImage, 0, 0, null);
        graphics.dispose();
        return outputImage;
    }

    public BufferedImage invertBlackAndWhite(BufferedImage image) {
        BufferedImage imageOut = new BufferedImage(image.getWidth(), image.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY); //TYPE_BYTE_GRAY //TYPE_INT_ARGB
        for (int i = 0; i < image.getWidth() - 1; i++) {
            for (int j = 0; j < image.getHeight() - 1; j++) {
                imageOut.setRGB(i, j, Color.black.getRGB());
            }
        }
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                Color c = new Color(image.getRGB(i, j));
                if (c.equals(Color.white) || c.getRGB() < -400000) {
                    imageOut.setRGB(i, j, c.getRGB());
                }
            }
        }
        return imageOut;
    }

}
