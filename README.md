# SeedVR2 — 사진 · 영상 업스케일

PromBoard 커뮤니티 플러그인입니다. 캔버스에 **SeedVR2 사진 업스케일** · **SeedVR2 영상 업스케일** 노드 두 개를 더합니다. ByteDance [SeedVR2](https://github.com/ByteDance-Seed/SeedVR)(한 단계 확산 복원)로 사진 · 영상을 2~4배 키우면서 흐린 부분과 잡티를 되살립니다. 추론 코드는 [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) v2.5.24 를 옮겨 썼습니다(Apache-2.0, [NOTICE](NOTICE)).

![왼쪽부터 1/4 로 줄인 사진을 바이큐빅으로 4배 키운 것, SeedVR2 로 4배 키운 것, 줄이기 전 원본](docs/hero.jpg)

왼쪽은 원본을 1/4(159×145)로 줄인 사진을 보통 방법(바이큐빅)으로 4배 키운 것, 가운데는 같은 작은 사진을 이 플러그인으로 4배(636×580) 키운 결과, 오른쪽은 줄이기 전 원본입니다. 잎맥 · 책 천 결 · 나뭇결이 원본에 가깝게 살아납니다. 결과는 이 저장소의 `nodes/main.py` 를 PromBoard 앱(격리 실행)에서 그대로 돌린 것입니다(판 3B · fp8, 색 맞추기 LAB, 시드 42).

## 노드

| 노드 | 입력 → 결과 | 비교 보기 |
|---|---|---|
| SeedVR2 사진 업스케일 | 사진 → PNG. 투명 PNG 는 투명한 부분도 함께 키웁니다 | 전/후 비교 슬라이더 |
| SeedVR2 영상 업스케일 | 영상 → mp4(H.264). 원본 소리를 그대로 담습니다(AAC · MP3 는 그대로, 그 밖은 AAC 로) | 프레임 띠 |

- 크기: 2배 · 3배 · 4배, 또는 결과의 짧은 변 1080 · 1440 · 2160px. 사진은 긴 변 4096px, 영상은 긴 변 2560px · 60초까지 받습니다.
- 색 맞추기: 원본 색(LAB, 권장) · 웨이블릿 · 안 함.
- 영상 '한 번에 처리할 프레임 수'(5 · 9 · 13 · 21 · 33): 클수록 프레임 사이가 매끄럽지만 그래픽카드 메모리를 많이 씁니다.
- 그래픽카드 메모리가 모자라 보이면 실행 전에 빈 메모리를 보고 BlockSwap(DiT 블록 일부를 시스템 메모리에 두기) · VAE 타일을 스스로 켜고, 그래도 메모리가 넘치면 가장 아끼는 설정으로 한 번 더 돕니다.

## 받는 것 · 크기

| 판 | 받는 크기 | 필요 그래픽카드 메모리(어림값) | 출처 |
|---|---|---|---|
| 3B · fp8 | 3.4GB | 8GB | numz/SeedVR2_comfyUI |
| 3B · fp16 | 6.8GB | 12GB | numz/SeedVR2_comfyUI |
| 3B · GGUF Q4 | 2.0GB | 6GB | AInVFX/SeedVR2_comfyUI |
| 7B · fp8 | 8.5GB | 12GB | AInVFX/SeedVR2_comfyUI |
| 7B 선명 · fp8 | 8.5GB | 12GB | AInVFX/SeedVR2_comfyUI |
| 7B · GGUF Q4 | 4.8GB | 8GB | AInVFX/SeedVR2_comfyUI |

처음 값 **자동**은 이 PC 그래픽카드 메모리가 8GB 이상이면 3B · fp8, 그 밖(앱이 메모리를 모르는 맥 · AMD 포함)은 3B · GGUF Q4 를 고르고 그 판만 받습니다. 필요 메모리는 DiT 가중치 + 계산 몫 약 3GB(노드가 BlockSwap 을 정하는 기준) + VAE 0.5GB 를 그래픽카드 크기로 올린 값입니다. 7B · GGUF Q4 는 numz 안내('7B 를 8GB 에서 GGUF 로')를 따랐습니다. 그보다 작은 그래픽카드에서도 BlockSwap 으로 돌지만 느려집니다.

모든 판에 VAE(0.5GB, numz)와 글 임베딩(1MB, ByteDance-Seed/SeedVR2-3B)이 함께 받아집니다. 고른 판만 받고, 파일마다 sha256 을 확인합니다. 가중치는 모두 Apache-2.0 이고 플러그인 zip 에는 들어 있지 않습니다.

## 그래픽카드 메모리 · 걸린 시간 (실측)

RTX 4070 SUPER 12GB · Windows 11, 판 3B · fp8:

| 입력 | 결과 | 그래픽카드 메모리 최대(전체) | 걸린 시간 |
|---|---|---|---|
| 사진 159×145, 4배 | 636×580 PNG | 5.8GB | 6분 34초 |

- 걸린 시간의 거의 전부가 가중치(3.4GB)를 디스크에서 읽는 시간입니다. 실제 업스케일 계산은 2초 남짓(DiT 1.2초 · VAE 0.4초)이었습니다. 시험 PC 는 여러 작업이 같은 하드디스크를 함께 쓰던 상태라 읽기가 아주 느렸고, 보통 SSD 에서는 훨씬 빠릅니다.
- **미검증 — 영상**: 480×320 · 3초 영상 2배를 돌렸지만, 위와 같은 디스크 읽기 때문에 앱의 노드 실행 한도(18분)를 넘겨 끝내지 못했습니다(그래픽카드 메모리는 최대 9.8GB 까지 썼습니다). 영상 결과는 확인하지 못했습니다.
- **미검증**: 3B · fp16, GGUF Q4, 7B 판들.
- 영상과 다른 판을 확인하지 못해 앱에는 '검증 안 됨'으로 보입니다(1.0.2 부터).
- 지원 OS: 윈도우(위 실측) · 리눅스 · 맥(애플 칩, MPS). 리눅스 · 맥은 잠금 파일과 코드 경로만 있고 돌려 보지 않았습니다.

## 라이선스

플러그인 코드 · 옮겨 온 numz 코드 · SeedVR2 원본 코드 · 가중치 모두 Apache-2.0 입니다([LICENSE](LICENSE), [NOTICE](NOTICE)). 옮기면서 바꾼 점(ComfyUI 노드 코드와 모델 내려받기 코드 빼기, 설정 파일의 eval 식을 값으로 바꾸기 등)은 NOTICE 에 적었습니다.

## 설치

1. PromBoard → 플러그인 매니저 → **커뮤니티 플러그인** → 둘러보기에서 설치합니다.
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. 처음 실행 전 실행 준비가 파이썬 패키지(앱이 고정한 torch 2.8.0 포함)와 고른 판의 가중치(자동이면 3.9GB, 그래픽카드 8GB 미만 · 맥 · AMD 는 2.5GB)를 받습니다.
4. 캔버스 빈 곳을 오른쪽 클릭하면 메뉴 맨 아래 플러그인 묶음에 두 노드가 있습니다.

## 플러그인 만드는 사람을 위한 메모

```
manifest.json            플러그인 정보 · 노드 2개(사진: views compare, 영상: views frameStrip) · 판별 모델(requires.models[].when)
nodes/main.py            NODES = {"image": ..., "video": ...} — 빈 그래픽카드 메모리로 BlockSwap · 타일 정하기, OOM 이면 한 번 더,
                         영상은 PyAV 로 조각(약 40프레임)씩 읽고 이어 붙여 H.264 로 쓰기, 소리 옮기기, 예상 시간이 한도를 넘으면 일찍 멈추기
nodes/seedvr2_core/      numz v2.5.24 의 src(ComfyUI 노드 · 내려받기 코드 뺌) + 3B · 7B 설정 파일
nodes/requirements-*.txt 운영체제별 해시 잠금 파일(torch 줄은 앱이 빼고 앱의 torch 2.8.0 을 먼저 깐다)
docs/                    이 README 의 그림 · 만드는 스크립트. 플러그인 zip 에는 넣지 않는다(.gitattributes export-ignore)
```

- 비교 그림 다시 만들기: `python docs/scripts/compare.py <작은 사진> <결과 PNG> <원본> docs/hero.jpg` (Pillow). 시험 사진은 PromBoard 가 만든 예제 사진(GPT Image)을 잘라 1/4 로 줄인 것입니다.

---

## English

A PromBoard community plugin. It adds two nodes, **SeedVR2 사진 업스케일** (photo upscale) and **SeedVR2 영상 업스케일** (video upscale). ByteDance [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) (one-step diffusion restoration) enlarges photos and videos 2–4× while restoring blur and artifacts. The inference code is taken from [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) v2.5.24 (Apache-2.0, see [NOTICE](NOTICE)).

![From left: a 1/4-size photo enlarged 4x with bicubic, the same photo enlarged 4x with SeedVR2, and the original](docs/hero.jpg)

Left: the original shrunk to 1/4 (159×145) and enlarged 4× with plain bicubic. Middle: the same small photo enlarged 4× (636×580) by this plugin. Right: the original before shrinking. Leaf veins, the book cloth and the wood grain come back close to the original. Produced by running this repository's `nodes/main.py` inside the PromBoard app (isolated run), model 3B · fp8, color LAB, seed 42.

### Nodes

- Photo → PNG; a transparent PNG keeps and enlarges its alpha. Shown with the before/after compare slider.
- Video → MP4 (H.264) with the original audio (AAC and MP3 copied, anything else re-encoded to AAC). Shown as a frame strip.
- Size: 2×, 3×, 4×, or a target short side of 1080, 1440 or 2160 px. Photos up to a 4096 px long side; videos up to a 2560 px long side and 60 seconds.
- Color matching: LAB (recommended), wavelet, or none. Video "frames per batch" (5, 9, 13, 21, 33): larger is smoother across frames but needs more GPU memory.
- Before each run the node checks free GPU memory and turns on BlockSwap (keeping some DiT blocks in system memory) and VAE tiling when needed; on an out-of-memory error it retries once with the most memory-saving settings.

### Downloads

3B · fp8 3.4 GB, 3B · fp16 6.8 GB (numz/SeedVR2_comfyUI); 3B · GGUF Q4 2.0 GB, 7B · fp8 8.5 GB, 7B sharp · fp8 8.5 GB, 7B · GGUF Q4 4.8 GB (AInVFX/SeedVR2_comfyUI). Estimated GPU memory: 3B · GGUF Q4 6 GB; 3B · fp8 and 7B · GGUF Q4 8 GB; 3B · fp16 and both 7B · fp8 12 GB (DiT weights + about 3 GB for compute, the node's BlockSwap rule, + 0.5 GB VAE, rounded up to a card size; 7B · GGUF Q4 follows numz's "7B on 8 GB with GGUF"). Smaller cards still run with BlockSwap, more slowly. The default, Auto, picks 3B · fp8 on an 8 GB+ GPU and 3B · GGUF Q4 otherwise (including Macs and AMD cards, whose memory the app cannot read). Every model also needs the VAE (0.5 GB, numz) and the text embeddings (1 MB, ByteDance-Seed/SeedVR2-3B). Only the chosen model is downloaded, each file pinned by sha256. All weights are Apache-2.0 and none are in the plugin zip.

### GPU memory and time (measured)

RTX 4070 SUPER 12 GB · Windows 11, 3B · fp8: photo 159×145 → 636×580 PNG, peak GPU memory 5.8 GB (whole card), 6 min 34 s. Almost all of that time was reading the 3.4 GB weights from disk — the upscale itself took about 2 seconds (DiT 1.2 s, VAE 0.4 s). The test PC's hard disk was shared by several jobs and very slow; a normal SSD is much faster. **Not verified — video**: a 480×320, 3-second clip at 2× was started but, because of the same disk reads, did not finish within the app's node time limit (18 minutes); it used up to 9.8 GB of GPU memory, and the video result was not checked. **Not verified**: 3B · fp16, GGUF Q4 and the 7B models. Because video and the other models are unchecked, the app shows the plugin as unverified (from 1.0.2). Supported OS: Windows (measured above), Linux and macOS (Apple silicon, MPS); Linux and macOS have lock files and code paths but were not run.

### Install

PromBoard → Plugin Manager → **Community plugins** → Browse → install, turn off **Restricted mode** and enable the plugin. The first prepare downloads the Python packages (including the app's pinned torch 2.8.0) and the chosen model (with Auto 3.9 GB, or 2.5 GB below 8 GB of GPU memory and on Macs and AMD cards). Right-click an empty spot on the canvas: the nodes are in the plugin group at the bottom of the menu.

The plugin texts in the app are Korean only for now — the plugin manifest takes a single language.

### License

Plugin code, the bundled numz code, the original SeedVR2 code and the weights are all Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The changes made to the bundled code (ComfyUI node and downloader code removed, the config `eval` expressions replaced by their values, and so on) are listed in NOTICE.
