# StreamLip 工作交接文档
**时间**：2026-05-29 13:40  
**项目路径**：`/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A`

---

## 项目目标

训练一个 **FM（Flow Matching）声码器头**，以唇形视频 + 文本先验 + 说话人身份为条件，生成 Mimi 音频 latent（512维，12.5Hz），最终解码为语音。

**核心消融实验**：对比"有文本条件"和"无文本条件"，验证文本先验对 lip-to-speech 的帮助。

---

## 系统架构

```
视频帧（Auto-AVSR）→ avsr_enc.npy   (T, 768)      ← 已预提取
文本（Auto-AVSR识别结果）→ SmolLM2 → smollm2_h.npy (L, 960) ← 正在预提取
说话人音频 → speaker_emb.npy        (256,)         ← 已预提取
Mimi 音频 latent → latent.npz       (T_a, 512)     ← 已预提取

训练时：
  v_down   = avsr_enc[::2][:T_a]    # 降采样到 12.5Hz，(T_a, 768)
  h_down   = smollm2_h[resample到T_a]  # (T_a, 960)
  cond     = cat[v_down, h_down, spk_expand] → Linear(1984, 512)
  FMHeadAVSR（6层 DiT，512维）→ OT-CFM loss on latent
```

---

## 数据

- **数据集**：LRS3-TED，路径 `/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3/`
- **预处理后**：`data/processed/`，manifest：`data/processed/manifest.csv`
- **有效 clip 缓存**：`data/processed/_fm_avsr_pretrain.txt`，共 **115,947** 条
- **每个 clip 目录内容**：
  ```
  avsr_enc.npy      (T, 768)      Auto-AVSR 编码特征
  avsr_text.txt     Auto-AVSR CTC 识别文本（WER ~20%）
  latent.npz        (T_a, 512)    Mimi 量化后 latent
  speaker_emb.npy   (256,)        说话人嵌入
  smollm2_h.npy     (L, 960)      SmolLM2 token 隐藏层（正在提取中！）
  audio.wav         原始音频
  ```

---

## 当前状态（2026-05-29 13:40）

### 正在运行的进程

| 进程 | PID | 状态 |
|------|-----|------|
| `extract_smollm2_h.py` | 3198907 | **运行中**，约 3% 完成（18/227批），预计再需 ~30分钟 |
| 另一个重复的 `extract_smollm2_h.py` | 3197093 | **需要手动 kill**（重复进程） |

**注意**：有两个 extract_smollm2 进程在跑，会互相抢 GPU 显存，需要先 kill 掉旧的：
```bash
kill 3197093 3197089
```

### 提取进度
```
logs/extract_smollm2.log   ← 主进程日志，~8.5s/batch × 227批 ≈ 32分钟总计
```

---

## 关键文件

### 训练脚本
- `scripts/train_fm_avsr.py` — FM 头训练主脚本
- `scripts/extract_smollm2_h.py` — SmolLM2 特征预提取（**新建，本次会话**）
- `scripts/extract_avsr_enc.py` — Auto-AVSR 特征预提取（已完成）

### 模型代码
- `src/streaminlip/v2/fm_head.py` — FMHead 基类（DiT，OT-CFM）
- `src/streaminlip/fm_avsr_dataset.py` — 数据集加载（**本次会话修改**）
- `src/streaminlip/auto_avsr.py` — Auto-AVSR 推理封装

### 预训练权重
- `pretrained/smollm2-360m/` — SmolLM2-360M
- `pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth` — Auto-AVSR

---

## 本次会话做的改动

### 1. `scripts/train_fm_avsr.py`

**修复了 val 循环中的三个 bug**：

| bug | 原代码 | 修复后 |
|-----|--------|--------|
| numpy 数组调用 `.to()` | `vb["enc"].to(device, ...)` | `torch.from_numpy(vb["enc"]).to(device, ...)` |
| 错误的 T_a_list（传 enc 长度而不是 latent 长度） | `texts_to_h_lm(..., lens_v)` | `texts_to_h_lm(..., [T_a_v]*B_v)` |
| 切片缺少维度 | `[:, :T_a_v]` | `[:, :T_a_v, :]` |

**未完成的改动**（需要下一步做）：
- 删除 SmolLM2 在线推理（`texts_to_h_lm` 函数和 SmolLM2 加载）
- 改为从 batch 的 `h_lm` 字段读取预提取的特征
- 恢复 `num_workers=8`

### 2. `src/streaminlip/fm_avsr_dataset.py`

**新增了 smollm2_h.npy 加载**：
- `__getitem__` 中加载 `smollm2_h.npy`（如果存在），以 `h_lm` 键返回
- `collate_fn` 新增 `h_lm`（B, max_L, 960）和 `lens_L` 字段

### 3. `scripts/extract_smollm2_h.py`（新建）

预提取脚本，对所有 clip 运行 SmolLM2，保存 `smollm2_h.npy`（L, 960, float16）。

---

## 下一步必须做的事

### Step 1：等 smollm2 提取完成，或确认进程正常
```bash
# 先杀掉重复进程
kill 3197093

# 查看进度
tail -f logs/extract_smollm2.log
```

### Step 2：修改 training loop，删掉 SmolLM2
在 `scripts/train_fm_avsr.py` 中：

1. **删除** SmolLM2 加载（`from transformers import ...` 和 `tok = ...`, `lm = ...`）
2. **删除** `texts_to_h_lm` 函数
3. **修改** training loop 中的 h_down 计算：
```python
# 原来（慢，在线推理）:
h_down = texts_to_h_lm(texts, tok, lm, device, [T_a] * B)

# 改为（快，从预提取读取）:
if batch["h_lm"] is not None and not args.no_text_cond:
    h_lm_np = batch["h_lm"]           # (B, max_L, 960) numpy
    h_lm_t  = torch.from_numpy(h_lm_np).to(device, dtype=torch.bfloat16)
    # 时间重采样: (max_L,) → (T_a,)
    lens_L  = batch["lens_L"]         # 每个样本实际 token 数
    h_downs = torch.zeros(B, T_a, 960, device=device, dtype=torch.bfloat16)
    for b in range(B):
        L = int(lens_L[b])
        idx = torch.clamp(torch.arange(T_a, device=device) * L // max(T_a, 1), 0, L-1)
        h_downs[b] = h_lm_t[b, idx]
    h_down = h_downs
else:
    h_down = torch.zeros(B, T_a, 960, device=device, dtype=torch.bfloat16)
```

4. **恢复** `num_workers=8`（因为不再需要 SmolLM2 在 GPU，显存够了）

### Step 3：启动正式训练
```bash
# with_text 版本
nohup uv run python scripts/train_fm_avsr.py \
  --run_name fm_avsr_with_text \
  --batch_size 128 --max_epochs 30 --num_workers 8 \
  > logs/fm_avsr_with_text.log 2>&1 &

# no_text 消融版本
nohup uv run python scripts/train_fm_avsr.py \
  --no_text_cond \
  --run_name fm_avsr_no_text \
  --batch_size 128 --max_epochs 30 --num_workers 8 \
  > logs/fm_avsr_no_text.log 2>&1 &
```

---

## 训练参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| FM head | 31.7M 参数 | FMHeadAVSR，6层 DiT，DIM=512 |
| COND_DIM | 1984 | 768（avsr）+ 960（smollm2）+ 256（speaker） |
| batch_size | 128 | 显存够用（无 SmolLM2 后） |
| lr | 2e-4 | AdamW，cosine decay |
| warmup | 3 epochs | |
| max_epochs | 30 | |
| 预计步数 | 13,290 | 443 steps/epoch |

---

## 已知问题

1. **shm 泄漏**：历史上多次 spawn/fork 失败导致系统 semaphore 耗尽。如果遇到 `unable to allocate shared memory`，执行：
   ```bash
   ipcs -s | awk 'NR>3{print $2}' | xargs -r ipcrm -s
   ```

2. **两个 extract_smollm2 进程**：需要手动 kill 旧的（3197093）

3. **training loop 未完成 SmolLM2 删除**：当前 `train_fm_avsr.py` 还有 SmolLM2 加载代码，会占显存，需要完成 Step 2 的修改后再训练。

---

## 实验目标

训练完成后用 `scripts/eval_audio_v4.py`（或类似脚本）生成音频样本，对比：
- `fm_avsr_with_text`：用 Auto-AVSR 识别文本引导 FM
- `fm_avsr_no_text`：纯视觉条件

评估指标：UTMOS（音质）、SECS（说话人相似度）、主观听感。

---

*2026-05-29 13:40*
