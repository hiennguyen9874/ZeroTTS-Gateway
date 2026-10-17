# audio.cpp (the upstream) server, built from the pinned submodule references/audio.cpp.
#
# Trimmed copy of references/audio.cpp/.devops/cuda.Dockerfile. Differences, all on purpose:
#   - only the OmniVoice family (AUDIOCPP_MODELS), deployment build (package specs embedded),
#     only the audiocpp_server target: much faster than the upstream image's full model set;
#   - CUDA 12.6 + GCC 13 by default (the toolkit tested on the host, docs/INSTALL.md;
#     nvcc 12.6 rejects the upstream default GCC 14);
#   - CUDA arch 80 (A100) by default. The upstream default list only has 80-virtual (PTX JIT).
#
# Build context is references/audio.cpp, so its own .dockerignore applies (excludes build/, docs/, ...).
# Used by compose.audiocpp.yaml; by hand, from the repo root:
#   docker build -f docker/audiocpp.Dockerfile -t omnivoice-audiocpp references/audio.cpp

ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.6.3

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# nvcc 12.6 supports host GCC <= 13; CUDA >= 12.8 also accepts 14.
ARG GCC_VERSION=13
# "80" = SASS + PTX for A100; "native" is not available in docker build. Several: "80;86;89".
ARG CUDA_DOCKER_ARCH=80
ARG AUDIOCPP_MODELS=omnivoice
ARG AUDIOCPP_VERSION=dev

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} g++-${GCC_VERSION} cmake ninja-build ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /src
COPY . .

# CPU_ALL_VARIANTS instead of native CPU flags: the image must not depend on the build machine's CPU.
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DAUDIOCPP_MODEL_SET=custom \
        -DAUDIOCPP_MODELS="${AUDIOCPP_MODELS}" \
        -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
        -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF \
        -DAUDIOCPP_VERSION="${AUDIOCPP_VERSION}" \
        -DENGINE_ENABLE_CPU_ALL_VARIANTS=ON \
        -DENGINE_ENABLE_CUDA=ON \
        -DENGINE_ENABLE_CUDA_GRAPHS=ON \
        -DENGINE_ENABLE_VULKAN=OFF \
        -DENGINE_ENABLE_OPENMP=ON \
        -DENGINE_BUILD_EXAMPLES=OFF \
        -DENGINE_BUILD_TESTS=OFF \
        -DENGINE_BUILD_WARMBENCH=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --parallel "$(nproc)" --target audiocpp_server && \
    mkdir -p /out && \
    find build -name "*.so*" -exec cp -P {} /out/ \; && \
    cp build/bin/audiocpp_server /out/

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# libgomp1: OpenMP; ffmpeg: decoding non-WAV voice references; curl: healthcheck.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Binary and ggml backend libraries side by side, as in the upstream image.
WORKDIR /app
COPY --from=build /out/ /app/

# Non-root (uid 1000, from the base image). Models, voices and config are mounted read-only.
USER ubuntu
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["/app/audiocpp_server"]
CMD ["--config", "/srv/configs/server.json", "--host", "0.0.0.0", "--port", "8080", "--no-ui"]
