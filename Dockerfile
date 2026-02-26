# syntax=docker/dockerfile:1.4

FROM --platform=$BUILDPLATFORM golang:1.25-alpine AS build
LABEL maintainer="Jamie Nuttall"
LABEL description="PixMe Image Face recognition"
LABEL version="1.0"
LABEL org.opencontainers.image.source="https://github.com/papawattu/pixme-facerecognition"

RUN apk add --no-cache git

WORKDIR /app
COPY go.mod go.sum ./

RUN go mod download

COPY . .

# Set up cross-compilation for multi-arch
ARG TARGETOS
ARG TARGETARCH
ARG BUILD_VERSION

# Use correct values for cross-compilation
RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH:-amd64} go build -ldflags "-X main.BuildVersion=$BUILD_VERSION" -o pixme-facerecognition .

FROM scratch

COPY --from=build /app/pixme-facerecognition /pixme-facerecognition

EXPOSE 8080
CMD ["/pixme-facerecognition"]
