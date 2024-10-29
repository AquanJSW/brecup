# brecup

Simplify my 
[BililiveRecorder](https://github.com/BililiveRecorder/BililiveRecorder) → 
[biliup-rs](https://github.com/biliup/biliup-rs) workflow:

1. Embed danmaku into the video.
2. Clip the video.
3. Upload/append the video.

## Default Configuration

- DanmakuFactory:
    - no msg box
    - no override
    - half the opacity
- ffmpeg:
    - bitrate 8K
    - codec h264_nvenc

## Pre-requirements

- [DanmakuFactory](https://github.com/hihkm/DanmakuFactory)'s binary
    **`danmaku-factory` is in your PATH**.
- BililiveRecorder's binary **`brec` is in your PATH**.

    Also make sure the records' xml **danmaku files are ready to use**.
- biliup-rs's binary **`biliup` is in your PATH**.

    Also make sure the corresponding **`cookies.json` is prepared**.
- **`nvidia-smi`, `ffmpeg`, `ffprobe` are in your PATH**.
- **At lease one NVIDIA GPU is available**.

## Installation

```bash
pip install brecup
```

## Usage

The config file should be named as `*.brecup.y[a]ml` such as `test.brecup.yaml` or `test.brecup.yml`.

A schema file is provided [here](https://gist.githubusercontent.com/AquanJSW/c9002c3577e26b57c85b922ecb8c6bc8/raw/96c3f5262ad57afa3a7753d7c571c7f6112b3bc5/brecup.schema.json).

A sample:

```yaml
records:
- video: /path/to/rec_0.flv
  title: p0
  to: 00:00:30
  enabled: true
- video: /path/to/rec_1.flv
  title: p1
  ss: 00:06:50
  enabled: true
title: 'Test Collection'
output-dir: /path/to/output/Test-Collection/
tid: 172
tag: 直播回放,单机游戏
cookies: /path/to/cookies.json
cover: /path/to/cover.jpg
```

If you have more videos to append, just add more records and disable the previous ones:

```yaml
records:
- video: /path/to/rec_0.flv
  title: p0
  to: 00:00:30
  enabled: false
- video: /path/to/rec_1.flv
  title: p1
  ss: 00:06:50
  enabled: false
- video: /path/to/rec_2.flv
  title: p2
  ss: 00:06:50
  to: 00:07:50
  enabled: true
title: 'Test Collection'
output-dir: /path/to/output/Test-Collection/
tid: 172
tag: 直播回放,单机游戏
cookies: /path/to/cookies.json
cover: /path/to/cover.jpg
```

Then run:

```bash
brecup test.brecup.yaml
```