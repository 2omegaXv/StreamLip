# StreamLip 数据设计规范

> 本文档是预处理脚本与模型架构设计的**唯一接口约定**。两端均以此为准，不依赖彼此实现细节。

---

## 1. 源数据概况

### LRS3-TED 目录结构

```
/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3/
├── pretrain/pretrain/   # 118,516 clips，平均 ~13s，共 ~433h
├── trainval/trainval/   #  31,982 clips，平均 ~3.5s，共 ~31h
└── test/test/           #   1,321 clips，平均 ~2.5s，共 ~0.9h

每个 speaker 目录：
{speaker_id}/
├── 00001.mp4   # 原始视频，25fps，224×224（已面部裁剪）
├── 00001.txt   # 词级时间戳
└── ...
```

### LRS3 txt 文件格式

```
Text:  IT HAD A STRONG STURDY STEM ...
Conf:  4

WORD    START   END     ASDSCORE
IT      0.08    0.21    4.9
...
```

- `START` / `END`：秒，相对于 mp4 开头
- pretrain 有词级时间戳，test/trainval 部分无（words 为空，帧标签全为 SIL）
- 已是 forced alignment 结果，**无需额外运行 MFA**

---

## 2. 关键硬件常数

| 参数 | 值 | 说明 |
|------|----|------|
| 视频帧率 | **25 fps** | LRS3 固定 |
| 单帧时长 | **40 ms** | 1000ms / 25 |
| Mimi 输入采样率 | **24000 Hz** | 注意：不是 16kHz |
| Mimi 输出帧率 | **12.5 Hz** | 每 80ms 一个 latent 帧 |
| Mimi latent 维度 | **512** | encoder+transformer+downsample 输出（量化前），decoder 直接接受 |
| 视频/音频帧比 | **2 : 1** | 25fps / 12.5Hz，精确整除 |
| SmolLM2 词汇表大小 | **49152** | 含特殊 token |
| SIL token | **`<empty_output>`** | SmolLM2 special token，代表静音帧 |

### Chunk 对齐

| Chunk 时长 | 视频帧数 | Latent 帧数 | 推荐？ |
|-----------|---------|------------|--------|
| 160ms | 4 | 2 | ✅ |
| **240ms** | **6** | **3** | ✅ **默认** |
| 320ms | 8 | 4 | ✅ |
| 200ms | 5 | 2.5 | ❌ 不整除 |

---

## 3. 预处理管线

### 3.1 视频处理（GPU 加速）

```
decord 批量读帧 (N, H, W, 3)
    → SFD batch 检测（face_alignment GPU）→ bbox
    → kornia GPU warp_affine 批量对齐 → (N, 3, 256, 256)
    → FAN batch landmark（GPU，68pt）→ 精确嘴唇关键点
    → kornia GPU resize 批量裁唇 → lip (N, 96, 96, 3)
    → aligned GPU → CPU → face (N, 256, 256, 3)
```

与 AV-HuBERT 官方预处理差异：

| 步骤 | AV-HuBERT 官方 | 本项目 | 一致？ |
|------|---------------|--------|--------|
| 人脸检测 | RetinaFace | SFD（face_alignment 内置） | 近似 |
| 关键点 | FAN 68pt | FAN 68pt | ✅ |
| 稳定化 | similarity transform | 同上 | ✅ |
| 裁剪尺寸 | 96×96 | 96×96 | ✅ |
| 帧率 | 25fps | 25fps | ✅ |
| **颜色** | **灰度（1ch）** | **RGB（3ch）** | ❌ 唯一差异 |

### 3.2 音频处理

```
mp4 → ffmpeg → 24000Hz mono WAV
    → Mimi encoder（冻结）→ 连续 latent (T_a, 512) float16
```

latent 为量化**前**的连续编码器输出，FM head 直接以此为训练目标。

### 3.3 文本处理（DataLoader 运行时）

预处理只保存原始词+时间戳，运行时按需生成帧级标签。

---

## 4. 输出文件规范

### 目录结构

```
data/processed/
├── manifest.csv
└── {split}/
    └── {speaker_id}/
        └── {clip_id}/
            ├── lip.npy      # (T, 96, 96, 3) uint8，无压缩，mmap 读取
            ├── face.npz     # JPEG 压缩，见下方说明
            ├── audio.wav    # 24kHz mono PCM
            ├── latent.npz   # (T_a, 512) float16
            └── text.json    # 词+时间戳
```

### 存储格式选型依据

| 文件 | 格式 | 理由 |
|------|------|------|
| `lip.npy` | uint8 无压缩 | 训练主路径，mmap 读取 <1ms，零解码开销 |
| `face.npz` | JPEG 字节流 | 26x 压缩比，单文件无小文件问题，支持按帧随机读取 |
| `latent.npz` | float16 npz | 已较小（~100KB/clip），压缩可接受 |

### `lip.npy`

```python
frames = np.load("lip.npy", mmap_mode='r')
# shape: (T, 96, 96, 3)  dtype: uint8
```

### `face.npz` — JPEG 字节流格式

内部结构：
- `data`：所有帧 JPEG 字节拼接，dtype=uint8，shape=(总字节数,)
- `offsets`：帧边界，dtype=int64，shape=(T+1,)，`data[offsets[i]:offsets[i+1]]` = 第 i 帧 JPEG

```python
# 写入（preprocess_worker.py 中的 save_face_jpeg）
jpeg_bufs = [encode_jpeg(frame, quality=85) for frame in face_frames]
data    = np.frombuffer(b''.join(jpeg_bufs), dtype=np.uint8)
offsets = np.concatenate([[0], np.cumsum([len(b) for b in jpeg_bufs])]).astype(np.int64)
np.savez("face.npz", data=data, offsets=offsets)

# 读取所有帧
f = np.load("face.npz")
data, offsets = f['data'], f['offsets']
frames = [cv2.imdecode(data[offsets[i]:offsets[i+1]], cv2.IMREAD_COLOR)
          for i in range(len(offsets)-1)]  # BGR → RGB 需转换

# 读取单帧 i（训练时按需加载）
frame_bgr = cv2.imdecode(data[offsets[i]:offsets[i+1]], cv2.IMREAD_COLOR)
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
```

### `latent.npz`

```python
np.load("latent.npz")["latent"]
# shape: (T_a, 512)  dtype: float16
# T_a = T // 2（25fps / 12.5Hz = 2，精确整除）
```

### `audio.wav`

```
采样率：24000 Hz，声道：1（mono），位深：16-bit PCM
```

### `text.json`

```json
{
    "transcript": "IT HAD A STRONG STURDY STEM",
    "words": [
        {"word": "IT",     "start": 0.08, "end": 0.21},
        {"word": "HAD",    "start": 0.21, "end": 0.33}
    ],
    "n_frames": 250,
    "fps": 25
}
```

### `manifest.csv`

```
split,speaker_id,clip_id,path,n_frames,duration_sec,n_words
pretrain,00j9bKdiOjk,00001,pretrain/00j9bKdiOjk/00001,325,13.0,42
```

---

## 5. DataLoader 接口

### 输入（从磁盘加载）

```python
import numpy as np, cv2, json

# lip：内存映射，<1ms
lip = np.load("lip.npy", mmap_mode='r')               # (T, 96, 96, 3) uint8

# face：JPEG 解码，~25ms（由 prefetch 隐藏）
f = np.load("face.npz")
data, offsets = f['data'], f['offsets']
face = np.stack([
    cv2.cvtColor(cv2.imdecode(data[offsets[i]:offsets[i+1]], cv2.IMREAD_COLOR),
                 cv2.COLOR_BGR2RGB)
    for i in range(len(offsets)-1)
])                                                     # (T, 256, 256, 3) uint8

latent = np.load("latent.npz")["latent"]               # (T_a, 512) float16
words  = json.load(open("text.json"))["words"]
```

### 归一化

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
lip  = (lip  / 255.0 - mean) / std   # float32
face = (face / 255.0 - mean) / std   # float32
```

### 帧级文本标签生成

```python
SIL_TOKEN_ID = tokenizer.convert_tokens_to_ids("<empty_output>")
frame_labels = np.full(T, SIL_TOKEN_ID, dtype=np.int64)
for w in words:
    f_start = int(w["start"] * 25)
    f_end   = int(w["end"]   * 25)
    token_id = tokenizer.encode(w["word"], add_special_tokens=False)[0]
    frame_labels[f_start:f_end] = token_id
```

### Batch 结构

```python
batch = {
    "lip":          torch.float32,  # (B, T, 3, 96, 96)
    "face":         torch.float32,  # (B, T, 3, 256, 256)
    "latent":       torch.float32,  # (B, T_a, 512)    T_a = T // 2
    "frame_labels": torch.int64,    # (B, T)
    "mask":         torch.bool,     # (B, T)
}
```

---

## 6. Chunk 视角

```
时间轴（帧）：  0  1  2  3  4  5 | 6  7  8  9  10 11 | ...
                ←── chunk 0 ──→   ←── chunk 1 ──→

对应 latent：   0     1     2    |  3     4     5    | ...
```

每个 chunk（6帧，240ms）：
- lip：`lip[i*6:(i+1)*6]`  → (6, 3, 96, 96)
- latent：`latent[i*3:(i+1)*3]` → (3, 512)
- labels：`frame_labels[i*6:(i+1)*6]` → (6,)

---

## 7. 模型架构约束

### 7.1 视觉编码器

```
lip  encoder：(B, T, 3, 96, 96)   — AV-HuBERT，仅改 in_channels: 1→3
face encoder：(B, T, 3, 256, 256) — 独立编码器（架构待定）
```

### 7.2 AR Text Head

```
输出维度：49152（SmolLM2 词汇表）
loss：CrossEntropyLoss，SIL token 不 ignore（让模型学停顿）
```

### 7.3 FM Head

```
预测目标：(B, T_a, 512) — Mimi 量化前连续 latent
条件输入：backbone hidden state（stop-gradient）
推理：预测 latent → Mimi decoder → 24kHz waveform
```

### 7.4 时序对齐

```
text head：1 token / 视频帧（25fps）
FM head：  1 latent / 2 视频帧（12.5Hz）
chunk：    T % 6 == 0（DataLoader 截断或 pad）
```

---

## 8. 存储估算（全量 151,819 clips）

基于实测（3870 clip 采样，均值 8.2MB lip + 58.6MB face_raw/clip）：

| 文件 | 格式 | 均值/clip | 全量 |
|------|------|---------|------|
| lip.npy | uint8 无压缩 | 8.2 MB | ~1,245 GB |
| face.npz | JPEG q85 | 2.2 MB | ~334 GB |
| audio.wav | 24kHz PCM | 1.5 MB | ~228 GB |
| latent.npz | float16 | 0.1 MB | ~15 GB |
| text.json | JSON | ~0.01 MB | ~2 GB |
| **合计** | | **~12 MB** | **~1,824 GB** |

与备选方案对比：

| 方案 | 全量 | 训练读+/epoch |
|------|------|------------|
| lip.npy + face.npy（原始）| ~10 TB | 0 |
| **lip.npy + face.npz（当前）** | **~1.8 TB** | **+25min** |
| lip.npy（无face） | ~1.5 TB | 0 |
| lip.npz + 无face | ~1.1 TB | +19min |

---

## 9. 版本记录

| 日期 | 变更 |
|------|------|
| 2026-05-14 | 初始版本 |
| 2026-05-15 | 存储格式重构：face.npy→face.npz(JPEG)，lip.npy无压缩；基于实测估算全量存储 |
