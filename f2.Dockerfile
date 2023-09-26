FROM openjdk:21-slim AS builder

RUN apt update
RUN apt install curl unzip zip sed bash -y

RUN curl -s "https://get.sdkman.io" | bash
RUN /bin/bash -c "source $HOME/.sdkman/bin/sdkman-init.sh; sdk version; sdk install gradle 8.3"


COPY --chown=gradle:gradle /src /app/src
COPY --chown=gradle:gradle build.gradle /app/build.gradle
COPY --chown=gradle:gradle gradlew.bat /app/gradlew.bat
COPY --chown=gradle:gradle settings.gradle /app/settings.gradle
COPY --chown=gradle:gradle gradlew /app/gradlew

WORKDIR /app
RUN /bin/bash -c "source $HOME/.sdkman/bin/sdkman-init.sh; gradle --no-daemon build"

FROM openjdk:21-slim
ENV NAME_APP=10832Surface2Java-0.0.1.jar

COPY --from=builder "/app/build/libs/$NAME_APP" app.jar

ENTRYPOINT java -jar app.jar


# docker build --progress=plain -t m2-f3 -f f2.Dockerfile .
# docker run -e PORT=8080 -p 8080:8080 --name m2-f3c -d m2-f3

# docker login --username oauth --password secret cr.yandex
# docker image tag m2-f3 cr.yandex/crpbtkqol2ing4gt1s4p/m2:v1
# docker push cr.yandex/crpbtkqol2ing4gt1s4p/m2:v1

