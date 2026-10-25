"""promboard.seedvr2 — SeedVR2 사진 · 영상 업스케일 노드.

추론 코드는 numz/ComfyUI-SeedVR2_VideoUpscaler v2.5.24(Apache-2.0)를 nodes/seedvr2_core 로 옮겨 고친 것이다(NOTICE).
이 파일은 그 코드를 PromBoard 노드 약속(ctx, inputs, settings) → outputs 로 부르는 얇은 층이다.

  - 사진: 한 장을 1프레임 영상으로 넣는다(batch 1). 투명(RGBA) PNG 는 알파도 함께 키운다.
  - 영상: PyAV 로 프레임을 읽어 묶음(청크)마다 돌리고 H.264 mp4(yuv420p)로 쓴다. 원본 소리는 그대로 옮긴다.
  - 메모리: DiT · VAE 는 단계 사이에 CPU 로 내린다. 그래픽카드 메모리가 적으면 BlockSwap · VAE 타일을 켜고,
    그래도 메모리가 모자라면(OOM) 가장 아끼는 설정으로 한 번 더 돈다.
"""
from __future__ import annotations

import gc
import math
import os
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

# 판 → (받는 파일 이름, 3B/7B). 파일 이름은 manifest requires.models 의 path 와 같다.
MODELS: Dict[str, Tuple[str, str, str]] = {
    "3b-fp8": ("numz/SeedVR2_comfyUI", "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "3b"),
    "3b-fp16": ("numz/SeedVR2_comfyUI", "seedvr2_ema_3b_fp16.safetensors", "3b"),
    "3b-q4": ("AInVFX/SeedVR2_comfyUI", "seedvr2_ema_3b-Q4_K_M.gguf", "3b"),
    "7b-fp8": ("AInVFX/SeedVR2_comfyUI", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors", "7b"),
    "7b-sharp-fp8": ("AInVFX/SeedVR2_comfyUI", "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", "7b"),
    "7b-q4": ("AInVFX/SeedVR2_comfyUI", "seedvr2_ema_7b-Q4_K_M.gguf", "7b"),
}
# 판 → DiT 가중치 크기(GB, 그래픽카드에 올라가는 몫) — BlockSwap 블록 수 계산용
WEIGHT_GB = {"3b-fp8": 3.4, "3b-fp16": 6.8, "3b-q4": 2.0, "7b-fp8": 8.5, "7b-sharp-fp8": 8.5, "7b-q4": 4.8}
VAE = ("numz/SeedVR2_comfyUI", "ema_vae_fp16.safetensors")
EMBEDS = ("ByteDance-Seed/SeedVR2-3B", "pos_emb.pt", "neg_emb.pt")
# 블록 수(BlockSwap 상한) — configs_3b · configs_7b 의 num_layers
BLOCKS = {"3b": 32, "7b": 36}

IMAGE_MAX_EDGE = 4096  # 사진 결과 긴 변 상한(px)
VIDEO_MAX_EDGE = 2560  # 영상 결과 긴 변 상한(px)
VIDEO_MAX_SEC = 60.0  # 받는 영상 길이 상한(초)
# 노드 한 번 예산(서버 PLUGIN_NODE_BUDGET_SEC 900초)보다 조금 짧게 — 넘을 것 같으면 첫 묶음 뒤에 멈추고 알린다
RUN_DEADLINE_SEC = 840.0

OOM_HELP = (
    "그래픽카드 메모리가 모자랍니다. 크기를 '2배'로 줄이거나, 판을 '3B · fp8' · '3B · GGUF Q4' 처럼 가벼운 것으로 "
    "바꾸거나, 영상이면 '한 번에 처리할 프레임'을 줄여 주세요. 다른 프로그램이 그래픽카드를 쓰고 있으면 닫아 주세요."
)


class _Progress:
    """단계(인코딩 → 키우기 → 디코딩 → 마무리) 진행률을 [lo, hi] 구간에 맞춰 ctx.progress 로 보낸다."""

    PHASES = {"Phase 1": (0.0, 0.25), "Phase 2": (0.25, 0.75), "Phase 3": (0.75, 0.95), "Phase 4": (0.95, 1.0)}

    def __init__(self, ctx, lo: float, hi: float):
        self.ctx, self.lo, self.hi = ctx, lo, hi

    def __call__(self, current, total, _frames=None, phase: str = "") -> None:
        a, b = (0.0, 1.0)
        for key, span in self.PHASES.items():
            if str(phase).startswith(key):
                a, b = span
                break
        frac = a + (b - a) * (float(current) / max(1.0, float(total)))
        self.ctx.progress(self.lo + (self.hi - self.lo) * frac, "upscale")
        self.ctx.check_cancel()


def _core():
    """옮겨 온 추론 코드를 처음 부를 때 불러온다(불러오기만 몇 초 걸린다)."""
    from .seedvr2_core.src.core import generation_phases as phases
    from .seedvr2_core.src.core import generation_utils as utils
    from .seedvr2_core.src.core import model_cache
    from .seedvr2_core.src.optimization import memory_manager
    from .seedvr2_core.src.utils.debug import Debug

    return phases, utils, model_cache, memory_manager, Debug


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cudaerrormemoryallocation" in text or "mps backend out of memory" in text


def _free_vram_gb(device: str) -> float:
    """지금 비어 있는 그래픽카드 메모리(GB). 윈도우는 넘치면 오류 대신 시스템 메모리로 넘겨 아주 느려지므로 전체가 아니라 빈 몫을 본다."""
    import torch

    if device == "cuda" and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info(0)
        return free / 1024**3
    return 0.0


def _plan_memory(device: str, model: str, out_pixels: int, level: int) -> Dict[str, Any]:
    """메모리 설정. level 0 = 이 PC 그래픽카드 메모리에 맞춘 설정, 1 = 가장 아끼는 설정(OOM 뒤 다시 돌 때)."""
    blocks = BLOCKS[MODELS[model][2]]
    if device != "cuda":
        # mps 는 통합 메모리라 BlockSwap 이 뜻이 없다(numz 안내). 타일만 켠다.
        return {"blocks": 0, "tiled": out_pixels > 2_000_000 or level > 0, "tile": 768 if level > 0 else 1024}
    if level > 0:
        return {"blocks": blocks, "tiled": True, "tile": 512}
    free = _free_vram_gb(device)
    # DiT 가중치가 그래픽카드에 다 올라가고도 활성값 몫(약 3GB)이 남는지로 BlockSwap 블록 수를 고른다.
    weight = WEIGHT_GB[model]
    spare = free - weight - 3.0
    swap = 0 if spare >= 0 else min(blocks, int(math.ceil(-spare / (weight / blocks))) + 2)
    tiled = out_pixels > (2_500_000 if free >= 20 else 1_200_000)
    return {"blocks": swap, "tiled": tiled, "tile": 1024 if free >= 10 else 768, "free_gb": round(free, 1)}


class _Upscaler:
    """한 번의 노드 실행 동안 DiT · VAE 를 붙잡아 두고 프레임 묶음을 키운다."""

    def __init__(self, ctx, settings: Dict[str, Any], mem: Dict[str, Any]):
        self.ctx = ctx
        self.phases, self.utils, self.model_cache, self.mm, Debug = _core()
        self.debug = Debug(enabled=False)
        repo, fname, size = MODELS[settings["model"]]
        self.dit_path = ctx.model_path(f"{repo}/{fname}")
        self.vae_path = ctx.model_path(f"{VAE[0]}/{VAE[1]}")
        self.utils.TEXT_EMBEDS_PATHS = (
            ctx.model_path(f"{EMBEDS[0]}/{EMBEDS[1]}"),
            ctx.model_path(f"{EMBEDS[0]}/{EMBEDS[2]}"),
        )
        self.size = size
        self.mem = mem
        self.seed = int(settings.get("seed", 42))
        self.color = str(settings.get("color", "lab"))
        self.runner = None
        self.gen = None
        device = "cuda:0" if ctx.device == "cuda" else ctx.device
        self.gen = self.utils.setup_generation_context(
            dit_device=device,
            vae_device=device,
            dit_offload_device="cpu",
            vae_offload_device="cpu",
            tensor_offload_device="cpu",
            debug=self.debug,
        )
        self.gen["interrupt_fn"] = ctx.check_cancel

    def _prepare(self) -> None:
        keep = {"dit_device", "vae_device", "dit_offload_device", "vae_offload_device", "tensor_offload_device", "compute_dtype", "interrupt_fn"}
        for key in list(self.gen.keys()):
            if key not in keep:
                del self.gen[key]
        tile = (self.mem["tile"], self.mem["tile"])
        overlap = (128, 128) if self.mem["tile"] >= 768 else (64, 64)
        self.runner, cache_context = self.utils.prepare_runner(
            dit_model=self.dit_path,
            vae_model=self.vae_path,
            model_dir=os.path.dirname(self.dit_path),
            debug=self.debug,
            ctx=self.gen,
            dit_cache=True,
            vae_cache=True,
            dit_id="promboard_dit",
            vae_id="promboard_vae",
            block_swap_config={
                "blocks_to_swap": self.mem["blocks"],
                "swap_io_components": self.mem["blocks"] >= BLOCKS[self.size],
                "offload_device": "cpu",
            },
            encode_tiled=self.mem["tiled"],
            encode_tile_size=tile,
            encode_tile_overlap=overlap,
            decode_tiled=self.mem["tiled"],
            decode_tile_size=tile,
            decode_tile_overlap=overlap,
            tile_debug="false",
            attention_mode="sdpa",
            torch_compile_args_dit=None,
            torch_compile_args_vae=None,
        )
        self.gen["cache_context"] = cache_context
        self.gen["text_embeds"] = self.utils.load_text_embeddings(
            self.utils.script_directory, self.gen["dit_device"], self.gen["compute_dtype"], self.debug
        )

    def run(self, frames, resolution: int, max_edge: int, batch: int, overlap: int, progress: _Progress):
        """frames: [T, H, W, C] float16 0~1 (CPU) → 같은 꼴의 float32 결과(CPU)."""
        import torch

        self._prepare()
        frames, _info = self.utils.compute_generation_info(
            ctx=self.gen, images=frames, resolution=resolution, max_resolution=max_edge,
            batch_size=batch, uniform_batch_size=False, seed=self.seed, prepend_frames=0,
            temporal_overlap=overlap, debug=self.debug,
        )
        self.gen = self.phases.encode_all_batches(
            self.runner, ctx=self.gen, images=frames, debug=self.debug, batch_size=batch,
            uniform_batch_size=False, seed=self.seed, progress_callback=progress, temporal_overlap=overlap,
            resolution=resolution, max_resolution=max_edge, input_noise_scale=0.0, color_correction=self.color,
        )
        self.gen = self.phases.upscale_all_batches(
            self.runner, ctx=self.gen, debug=self.debug, progress_callback=progress,
            seed=self.seed, latent_noise_scale=0.0, cache_model=True,
        )
        self.gen = self.phases.decode_all_batches(
            self.runner, ctx=self.gen, debug=self.debug, progress_callback=progress, cache_model=True,
        )
        self.gen = self.phases.postprocess_all_batches(
            ctx=self.gen, debug=self.debug, progress_callback=progress, color_correction=self.color,
            prepend_frames=0, temporal_overlap=overlap, batch_size=batch,
        )
        out = self.gen["final_video"]
        if out.device.type != "cpu":
            out = out.cpu()
        if out.dtype != torch.float32:
            out = out.to(torch.float32)
        del self.gen["final_video"]
        return out

    def close(self) -> None:
        """붙잡아 둔 모델을 놓는다(노드 프로세스는 다음 실행을 위해 살아 있을 수 있다)."""
        import torch

        cache = self.model_cache.get_global_cache()
        for store in (cache._dit_models, cache._vae_models):
            for model, _cfg in list(store.values()):
                try:
                    self.mm.release_model_memory(model, debug=self.debug)
                except Exception:  # noqa: BLE001 - 정리 실패는 결과에 영향이 없다
                    pass
            store.clear()
        cache._runner_templates.clear()
        self.runner = None
        self.gen = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _target(settings: Dict[str, Any], w: int, h: int, max_edge: int) -> Tuple[int, int]:
    """(짧은 변 목표 px, 긴 변 상한 px). numz 의 resolution = 짧은 변."""
    scale = str(settings["scale"])
    short = min(w, h)
    if scale.startswith("x"):
        target = int(round(short * float(scale[1:])))
    else:
        target = int(scale)
    target = max(target, 16)
    target -= target % 2
    return target, max_edge


def _out_size(w: int, h: int, short: int, max_edge: int) -> Tuple[int, int]:
    k = short / float(min(w, h))
    ow, oh = w * k, h * k
    if max(ow, oh) > max_edge:
        k2 = max_edge / max(ow, oh)
        ow, oh = ow * k2, oh * k2
    return int(round(ow)), int(round(oh))


def _with_retry(ctx, settings, out_pixels: int, work, can_retry=lambda: True):
    """work(upscaler) 를 돌린다. 메모리가 모자라면(can_retry 가 참일 때) 가장 아끼는 설정으로 한 번 더 돈다."""
    last: Optional[BaseException] = None
    for level in (0, 1):
        mem = _plan_memory(ctx.device, settings["model"], out_pixels, level)
        ctx.log(f"memory plan level {level}: {mem}")
        up = None
        try:
            ctx.progress(0.02, "load-model")
            up = _Upscaler(ctx, settings, mem)
            return work(up)
        except Exception as exc:  # noqa: BLE001 - OOM 만 다시 돌고 나머지는 그대로 올린다
            if not _is_oom(exc):
                raise
            last = exc
            ctx.log(f"out of memory at level {level}: {str(exc)[:200]}", "warn")
            if not can_retry():
                break
        finally:
            if up is not None:
                up.close()
    raise RuntimeError(OOM_HELP) from last


# ── 사진 ──────────────────────────────────────────────────────────────────


def _load_image(path: str):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        im = im.convert("RGBA" if has_alpha else "RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr[None, ...]).to(torch.float16), has_alpha


def upscale_image(ctx, inputs, settings):
    import numpy as np
    from PIL import Image

    path = ctx.in_path(inputs["image"])
    frames, has_alpha = _load_image(path)
    h, w = int(frames.shape[1]), int(frames.shape[2])
    short, max_edge = _target(settings, w, h, IMAGE_MAX_EDGE)
    ow, oh = _out_size(w, h, short, max_edge)
    ctx.log(f"image {w}x{h} -> about {ow}x{oh} ({settings['model']}, alpha={has_alpha})")
    started = time.monotonic()

    def work(up: _Upscaler):
        return up.run(frames, short, max_edge, batch=1, overlap=0, progress=_Progress(ctx, 0.05, 0.97))

    out = _with_retry(ctx, settings, ow * oh, work)
    ctx.check_cancel()
    arr = (out[0].clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(arr, "RGBA" if arr.shape[2] == 4 else "RGB")
    dst = ctx.out_path("upscaled.png")
    img.save(dst, "PNG", compress_level=4)
    ctx.log(f"image done {img.width}x{img.height} in {time.monotonic() - started:.1f}s")
    ctx.progress(1.0, "done")
    return {"image": dst}


# ── 영상 ──────────────────────────────────────────────────────────────────


def _open_video(path: str):
    import av

    try:
        container = av.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"영상 파일을 열지 못했습니다. mp4 · webm · mov 영상인지 확인해 주세요. ({exc})") from None
    if not container.streams.video:
        container.close()
        raise ValueError("이 파일에는 영상이 없습니다.")
    return container


def _frame_rgb(frame):
    import numpy as np

    arr = frame.to_ndarray(format="rgb24")
    rot = 0
    try:
        rot = int(round(getattr(frame, "rotation", 0) or 0)) % 360
    except Exception:  # noqa: BLE001
        rot = 0
    if rot:
        # PyAV rotation 은 반시계 방향 각도다 — 보이는 방향으로 돌린다.
        arr = np.ascontiguousarray(np.rot90(arr, k=rot // 90))
    return arr


def upscale_video(ctx, inputs, settings):
    import av
    import numpy as np
    import torch

    path = ctx.in_path(inputs["video"])
    src = _open_video(path)
    vstream = src.streams.video[0]
    astream = src.streams.audio[0] if src.streams.audio else None
    rate = vstream.average_rate or vstream.guessed_rate or Fraction(30, 1)
    duration = float(src.duration / av.time_base) if src.duration else 0.0
    if duration > VIDEO_MAX_SEC + 0.5:
        src.close()
        raise ValueError(f"영상이 {duration:.0f}초입니다. {VIDEO_MAX_SEC:.0f}초까지 받습니다 — 잘라서 넣어 주세요.")
    expected = int(round(duration * float(rate))) if duration else int(vstream.frames or 0)

    batch = int(settings["batch"])
    overlap = 0 if batch <= 5 else (2 if batch <= 13 else 3)
    # 묶음(청크): batch 의 배수, 약 40프레임 — 결과 프레임을 CPU 에 쌓는 양을 줄인다.
    chunk = batch * max(1, int(round(40 / batch)))

    frames_iter = src.decode(vstream)
    # 첫 묶음은 미리 읽어 둔다 — 메모리가 모자라 다시 돌 때 처음부터 다시 넣는다.
    first_chunk: List[Any] = []
    try:
        for fr in frames_iter:
            first_chunk.append(_frame_rgb(fr))
            if len(first_chunk) >= chunk:
                break
    except Exception as exc:  # noqa: BLE001
        src.close()
        raise ValueError(f"영상을 읽지 못했습니다. ({exc})") from None
    if not first_chunk:
        src.close()
        raise ValueError("영상에서 프레임을 하나도 읽지 못했습니다.")
    h, w = first_chunk[0].shape[:2]
    short, max_edge = _target(settings, w, h, VIDEO_MAX_EDGE)
    ow, oh = _out_size(w, h, short, max_edge)
    ctx.log(f"video {w}x{h} {float(rate):.3f}fps ~{expected} frames -> about {ow}x{oh} batch {batch} chunk {chunk}")

    dst = ctx.out_path("upscaled.mp4")
    out = av.open(dst, mode="w")
    vout = out.add_stream("libx264", rate=rate)
    vout.pix_fmt = "yuv420p"
    vout.options = {"crf": "16", "preset": "medium"}
    aout = None
    if astream is not None:
        # mp4 에 그대로 담을 수 있는 소리(aac · mp3)는 패킷을 옮기고, 그 밖(webm 의 opus · vorbis 등)은 aac 로 다시 담는다.
        if astream.codec_context.name in ("aac", "mp3"):
            aout = out.add_stream_from_template(astream)
        else:
            aout = out.add_stream("aac", rate=astream.rate or 48000)
    sized = {"done": False}

    def write(result):
        arr = (result.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
        if arr.shape[3] == 4:
            arr = np.ascontiguousarray(arr[..., :3])
        if not sized["done"]:
            vout.width = arr.shape[2] - arr.shape[2] % 2
            vout.height = arr.shape[1] - arr.shape[1] % 2
            sized["done"] = True
        for fr in arr:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr[: vout.height, : vout.width]), format="rgb24")
            for pkt in vout.encode(vf):
                out.mux(pkt)

    started = time.monotonic()
    written = 0
    estimate_checked = False

    def chunks():
        yield first_chunk
        buf: List[Any] = []
        try:
            for fr in frames_iter:
                buf.append(_frame_rgb(fr))
                if len(buf) >= chunk:
                    yield buf
                    buf = []
        except av.error.FFmpegError as exc:  # 끝부분이 깨진 파일 — 읽은 데까지 낸다
            ctx.log(f"video decode stopped: {exc}", "warn")
        if buf:
            yield buf

    def work(up: _Upscaler):
        nonlocal written, estimate_checked
        tail = None
        total = max(expected, 1)
        for raw in chunks():
            ctx.check_cancel()
            new = torch.from_numpy(np.stack(raw).astype(np.float32) / 255.0)
            ctx_count = 0
            if tail is not None and overlap > 0:
                ctx_count = min(overlap, tail.shape[0])
                batch_in = torch.cat([tail[-ctx_count:], new], dim=0)
            else:
                batch_in = new
            lo = min(0.97, written / total)
            hi = min(0.97, (written + len(raw)) / total)
            result = up.run(batch_in.to(torch.float16), short, max_edge, batch, overlap, _Progress(ctx, lo, max(lo, hi)))
            if ctx_count:
                result = result[ctx_count:]
            write(result)
            written += int(result.shape[0])
            tail = new[-overlap:].clone() if overlap > 0 else None
            del result, batch_in
            if not estimate_checked and expected > written:
                estimate_checked = True
                spent = time.monotonic() - started
                guess = spent / max(1, written) * expected
                ctx.log(f"first chunk {written} frames in {spent:.1f}s, whole video about {guess:.0f}s")
                if guess > RUN_DEADLINE_SEC:
                    raise ValueError(
                        f"이 영상은 이 PC 에서 약 {guess / 60:.0f}분 걸릴 것 같아 멈췄습니다(한 번에 약 {RUN_DEADLINE_SEC / 60:.0f}분까지 돕니다). "
                        "영상을 짧게 자르거나, 크기를 '2배' · 판을 '3B · fp8' 로 낮춰 주세요."
                    )
        return written

    try:
        _with_retry(ctx, settings, ow * oh, work, can_retry=lambda: written == 0)
        ctx.check_cancel()
        for pkt in vout.encode():
            out.mux(pkt)
        if aout is not None:
            _copy_audio(src, astream, out, aout, ctx)
    finally:
        out.close()
        src.close()
    if written == 0:
        raise ValueError("영상에서 프레임을 하나도 읽지 못했습니다.")
    ctx.log(f"video done {written} frames in {time.monotonic() - started:.1f}s")
    ctx.progress(1.0, "done")
    return {"video": dst}


def _copy_audio(src, astream, out, aout, ctx) -> None:
    """원본 소리를 결과 mp4 에 옮긴다. 같은 코덱이면 그대로(패킷 복사), 아니면 aac 로 다시 담는다."""
    import av

    copy = astream.codec_context.name in ("aac", "mp3")
    try:
        src.seek(0)
        if copy:
            for pkt in src.demux(astream):
                if pkt.dts is None:
                    continue
                pkt.stream = aout
                out.mux(pkt)
        else:
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=aout.rate)
            for frame in src.decode(astream):
                for rf in resampler.resample(frame):
                    for pkt in aout.encode(rf):
                        out.mux(pkt)
            for pkt in aout.encode(None):
                out.mux(pkt)
    except Exception as exc:  # noqa: BLE001 - 소리를 못 옮겨도 영상은 낸다
        ctx.log(f"audio copy failed: {exc}", "warn")


NODES = {"image": upscale_image, "video": upscale_video}
