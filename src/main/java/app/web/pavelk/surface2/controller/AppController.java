package app.web.pavelk.surface2.controller;

import app.web.pavelk.surface2.dto.Result;
import app.web.pavelk.surface2.service.AppService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Slf4j
@RestController
@RequiredArgsConstructor
public class AppController {

    private final AppService appService;

    @PostMapping(value = "/images")
    public Result images(
            @RequestPart(value = "image", required = false) MultipartFile multipartFile
    ) throws IOException {
        return appService.images(multipartFile);
    }

}
