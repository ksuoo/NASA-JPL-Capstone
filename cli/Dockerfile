FROM ubuntu:22.04 AS build

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN g++ -std=c++17 -O2 main.cpp Log.cpp -o app

FROM ubuntu:22.04

WORKDIR /app

COPY --from=build /app/app .
COPY mars.jpg .
COPY a.png .

ENTRYPOINT ["./app"]
