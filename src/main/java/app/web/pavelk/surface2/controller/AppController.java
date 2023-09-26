package app.web.pavelk.surface2.controller;

import app.web.pavelk.surface2.dto.ResultImages;
import app.web.pavelk.surface2.dto.ResultSave;
import app.web.pavelk.surface2.service.AppService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
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
    public ResultImages images(
            @RequestPart(value = "image") MultipartFile multipartFile
    ) throws IOException {
        return appService.images(multipartFile);
    }

    @PostMapping(value = "/save")
    public ResultSave save(
            @RequestPart(value = "right") String right,
            @RequestPart(value = "image") MultipartFile multipartFile
    ) throws IOException {
        return appService.save(right, multipartFile);
    }

    @GetMapping(value = "/message")
    public String message() {
        return "{\"message\": \"Hello user\"}";
    }


}
