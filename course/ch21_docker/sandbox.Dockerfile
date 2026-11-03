# 每次 bash 工具调用都在这个镜像的一次性容器里执行。
#   docker build -f course/ch21_docker/sandbox.Dockerfile -t eh-sandbox:latest course/ch21_docker
FROM alpine:3.20

# 非 root 运行；/workspace 由 run 参数以 rw 挂入，镜像内只留一个空挂载点。
RUN adduser -D -u 1000 -h /home/sandbox sandbox \
    && mkdir -p /workspace \
    && chown sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace

# 容器以 sleep 常驻，命令通过 docker exec 进入。
CMD ["sleep", "infinity"]
