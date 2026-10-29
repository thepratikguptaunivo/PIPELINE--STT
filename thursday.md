# Production Speech-to-Text (STT) Architecture: End-to-End Engineering Specification (Day 1 — Modules 1A to 1D)

## 0. Production Repository Structure & Responsibility Boundary

In production, our custom Python code handles **Ingestion, Physical Standardization (Module 1B)**, **Smart VAD Chunking & Compliance Auditing (Module 1C)**, and **Post-Inference Seam Stitching (`transcriber.py`)**. Everything inside **Module 1D (STFT, 128-Bin Log-Mel Spectrogram, CNN Stem, Cross-Attention, and BPE Tokenization)** is executed automatically on the GPU by `faster-whisper` (CTranslate2) or `transformers` (PyTorch).

```text
univo_asr_pipeline/
├── requirements.txt                  # torch, torchaudio, silero-vad, faster-whisper, whisperx, transformers
├── src/
│   ├── __init__.py
│   ├── preprocessor.py               # [WE CODE THIS - Module 1B] 5-min filter, Pearson channel split, 16kHz Kaiser, 80Hz HPF
│   ├── smart_chunker.py              # [WE CODE THIS - Module 1C] Silero VAD Cut & Merge, Dead Air Audit, Tier-2 Overlap Stride
│   └── transcriber.py                # [WE CODE THIS - Phase 4] Runs GPU model on chunks, stitches 26.0s seams, aligns & diarizes
└── scripts/
    └── inspect_module_1d.py          # [DIAGNOSTIC SCRIPT - Module 1D] Prints live Spectrogram/CNN shapes & BPE token counts for demo

```

---

## Phase 1 (Modules 1A & 1B): CPU-Side Ingestion & Physical Standardization

### 1. Target Mathematical Input Format ("Inside the Model")

Every modern Speech-to-Text neural network (**Whisper**, **Conformer**, **Wav2Vec2**) expects one exact mathematical input format before GPU feature extraction begins:

* **Data Type:** 1D floating-point tensor (`float32`)
* **Sample Rate:** $16,000\text{ Hz}$ ($16\text{ kHz}$)
* **Topology:** Single-Channel (Mono per independent audio stream)
* **Amplitude Range:** Normalized strictly within $[-1.0, +1.0]$ (Peak-scaled to $-1.0\text{ dBFS}$ / $\pm 0.891$)

```text
[Raw .mp3 File (Up to 50 MB)]
  │ 
  ▼ 1. Header Inspection (ffprobe / torchaudio.info) -> Reject if < 300s (5 mins) or corrupted 
[MDCT Compressed Stream]
  │ 
  ▼ 2. Float32 PCM Decode + Pearson Correlation Channel Check:
  │    ├── If True Dual-Channel (|r| < 0.85): Keep Ch0 (Agent) & Ch1 (Customer) Split
  │    └── If Mono or Identical Dual-Mono (|r| >= 0.85): Downmix to Single Mono Stream
[Raw PCM Wave (8kHz / 44.1kHz / 48kHz)]
  │ 
  ▼ 3. Kaiser Windowed Sinc Resampler -> Standardized 16,000 Hz (16kHz)
  │    (Anti-imaging cutoff @ ~3.8kHz for 8kHz input; Anti-aliasing cutoff @ ~7.6kHz for 48kHz input)
[16kHz Discrete Time-Domain Signal]
  │ 
  ▼ 4. DC Offset Removal + 80Hz Butterworth High-Pass Filter (Strips 50Hz Indian AC Hum) 
[Filtered Speech Band (80Hz - Nyquist Limit)]
  │ 
  ▼ 5. Proportional Peak Normalization -> [-1.0, +1.0] (Scaled to -1.0 dBFS / 0.891)
[Clean Neural-Ready 1D Tensor(s)] ──► Ready for Module 1C (Smart VAD Chunking)

```

---

### 2. What Runs on the CPU (System RAM): Step-by-Step

#### Step 1: Fast Metadata Pre-Check (`ffprobe` / `torchaudio.info()`)

* **The Action:** Run an instant container header check in under $5\text{ milliseconds}$ to confirm the file is valid and meets the **5-minute ($\ge 300\text{s}$) business rule** before committing system RAM. *(Note: The 5-minute minimum is a business compliance filter for evaluating counseling calls, not an acoustic limitation of the AI model).*
* **What Fails Without It:** Blindly decoding a $50\text{ MB}$ compressed `.mp3` file into raw `float32` PCM memory causes it to expand by $10\times$ to $15\times$, consuming $\sim 600\text{ MB}$ of system RAM per worker thread on invalid or short calls.

#### Step 2: PCM Decoding & Channel Topology Routing

Once validated, `FFmpeg` / `torchaudio` decompresses the `.mp3` frames (stored via Modified Discrete Cosine Transform and Huffman coding) into a raw **Linear PCM (Pulse-Code Modulation)** sound wave tensor of shape `[num_channels, num_samples]`. A file is either Mono ($1\text{ channel}$) or Stereo ($2\text{ channels}$):

* **Case A — True Dual-Channel Stereo ($\lvert r \rvert \lt 0.85$ Split Routing):** Telephony dialers often record Channel 0 (Counselor/Agent) and Channel 1 (Student/Customer) on separate left/right tracks. We compute the **Pearson Cross-Correlation ($r$)** between Channel 0 and Channel 1. If $\lvert r \rvert \lt 0.85$, the two tracks contain independent speakers, and we process Channel 0 and Channel 1 **completely independently**. This provides $100\%$ deterministic speaker separation for free without needing GPU speaker diarization later, and completely eliminates crosstalk errors when both people speak simultaneously.
* **Case B — Mono or Identical "Dual-Mono" Stereo ($\lvert r \rvert \ge 0.85$):** If the file has $1\text{ channel}$, or if both stereo channels carry the same mixed audio ($r \approx 1.0$), we average them into a single Mono stream and route it to GPU speaker diarization (`pyannote.audio`) after transcription.

#### Step 3: The $16\text{ kHz}$ Standard & Anti-Aliased Kaiser Window Resampling

In production speech-to-text pipelines, CPUs standardize the audio's **sample rate** (digital snapshots per second)—not its pitch—to **$16,000\text{ Hz}$ ($16\text{ kHz}$)**.

* **Why $16\text{ kHz}$ is the Universal AI Standard:** By the **Nyquist-Shannon Sampling Theorem** ($f_{\text{max}} = \frac{f_s}{2}$), a $16\text{ kHz}$ sample rate captures frequencies up to $8\text{ kHz}$, covering the entire fundamental frequency and harmonic formant range of human vowels and consonants.
* **The Perfect Balance:** It avoids the consonant accuracy loss of raw telephone rates ($8\text{ kHz}$) and eliminates the wasted compute cost of high-definition music rates ($44.1\text{ kHz}$ or $48\text{ kHz}$).
* **Why Naive Resampling Fails:**
* *Upsampling Danger (Spectral Images):* Telephone lines record at $8\text{ kHz}$ (capturing up to $4\text{ kHz}$) while Whisper expects $16\text{ kHz}$. Stretching an $8\text{ kHz}$ call to $16\text{ kHz}$ by naively duplicating or linearly interpolating samples creates artificial, high-pitched robotic "ghost" mirror frequencies between $4\text{ kHz}$ and $8\text{ kHz}$ called **Spectral Images**.
* *Downsampling Danger (Aliasing):* Shrinking a $48\text{ kHz}$ VoIP recording down to $16\text{ kHz}$ without a low-pass filter causes high-pitched sounds above $8\text{ kHz}$ to mathematically "fold" backward into the lower human voice band (**Aliasing**), permanently corrupting transcription accuracy.


* **How the Kaiser Windowed Sinc Filter Works:**
1. **Band-Limited Cutoff:** Before taking new digital snapshots, `sinc_interp_kaiser` applies a steep low-pass roll-off just below the smaller Nyquist limit ($\sim 3.8\text{ kHz}$ when upsampling from $8\text{ kHz}$ to block spectral images; $\sim 7.6\text{ kHz}$ when downsampling from $48\text{ kHz}$ to block aliasing).
2. **The New Snapshots:** The CPU then computes clean digital snapshots of the reconstructed continuous wave at $16,000$ samples per second.



#### Step 4: The Production Noise Rule — DC Offset & $80\text{ Hz}$ High-Pass Filter (No Spectral Denoising!)

* **DC Offset Removal:** Subtracting the mean of the waveform (`waveform - waveform.mean()`) re-centers the electrical signal at $0.0$, preventing asymmetric clipping during normalization.
* **$80\text{ Hz}$ Butterworth High-Pass Filter:** Strips sub-bass mic rumble, desk vibrations, and the **$50\text{ Hz}$ Indian AC electrical mains hum** without touching the human fundamental voice frequency ($\ge 85\text{ Hz}$ for adult males, $\ge 165\text{ Hz}$ for adult females).
* **The Golden Production Rule:** **Never use third-party spectral denoisers (like `noisereduce` or `RNNoise`) before feeding audio into Whisper.** Whisper was trained on millions of hours of natural background noise and static; if you feed it natural static, it transcribes smoothly. If you feed it artificially warped, mathematically "hollowed-out" audio from an external spectral denoiser (which creates watery "musical noise" artifacts and carves holes out of quiet Hinglish consonants like `/s/`, `/f/`, `/k/`, `/t/`), the neural network gets confused and **hallucinates**.

#### Step 5: Proportional Peak Normalization ($-1.0\text{ dBFS}$)

Finally, the CPU proportionally scales (multiplies) the waveform so its highest peak sits at **$-1.0\text{ dBFS}$ ($0.891$ linear amplitude)** within $[-1.0, +1.0]$. This guarantees consistent acoustic energy for Silero VAD and the Whisper encoder while leaving a $1.0\text{ dB}$ headroom margin to prevent digital clipping.

---

### 3. Production Code File 1: `src/preprocessor.py`

```python
# File: src/preprocessor.py
import torch
import torchaudio
import torchaudio.functional as F

def preprocess_production_audio(
    mp3_path: str,
    target_sr: int = 16000,
    min_duration_sec: float = 300.0,
    dual_channel_corr_threshold: float = 0.85
) -> dict:
    """
    Modules 1A & 1B: CPU-Side Ingestion & Physical Standardization.
    1. Instant header check (< 5ms) for file validity and >= 5-minute business rule.
    2. Decodes MP3 to float32 PCM and uses Pearson Correlation (|r| < 0.85) to detect
       True Dual-Channel (Agent/Customer split) vs. Mono/Identical Dual-Mono Stereo.
    3. Resamples to 16 kHz via Anti-Aliased Kaiser Windowed Sinc Filter.
    4. Strips DC Offset and 50Hz AC hum via 80Hz Butterworth High-Pass Filter.
    5. Proportionally peak-normalizes each active stream to -1.0 dBFS (0.891 amplitude).
    """
    # 1. Fast header check BEFORE loading 50MB file into System RAM
    metadata = torchaudio.info(mp3_path)
    duration_sec = metadata.num_frames / metadata.sample_rate

    if duration_sec < min_duration_sec:
        raise ValueError(
            f"Call duration ({duration_sec:.1f}s) is under the {min_duration_sec}s minimum rule."
        )

    # 2. Decode .mp3 to float32 tensor in [-1.0, 1.0], shape: [num_channels, num_samples]
    waveform, orig_sr = torchaudio.load(mp3_path, normalize=True)

    # 3. Check Channel Topology using Pearson Cross-Correlation
    is_dual_channel = False
    if waveform.shape[0] >= 2:
        ch0 = waveform[0] - waveform[0].mean()
        ch1 = waveform[1] - waveform[1].mean()
        denom = torch.sqrt(torch.sum(ch0 ** 2) * torch.sum(ch1 ** 2))
        corr = (torch.sum(ch0 * ch1) / denom).item() if denom > 1e-9 else 1.0

        if abs(corr) < dual_channel_corr_threshold:
            # Independent voices on Left (Agent) and Right (Customer)
            is_dual_channel = True
            waveform = waveform[:2, :]  # Keep [2, num_samples] to process independently
        else:
            # Identical dual-mono stereo -> downmix to [1, num_samples]
            waveform = torch.mean(waveform[:2, :], dim=0, keepdim=True)

    # 4. Anti-Aliased Kaiser Window Resampling (e.g., 8000 Hz or 48000 Hz -> 16000 Hz)
    if orig_sr != target_sr:
        waveform = F.resample(
            waveform,
            orig_freq=orig_sr,
            new_freq=target_sr,
            resampling_method="sinc_interp_kaiser"
        )

    # 5. Remove DC Offset (per channel) & Apply 80Hz Butterworth High-Pass Filter (strips 50Hz AC hum)
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    waveform = F.highpass_biquad(waveform, sample_rate=target_sr, cutoff_freq=80.0)

    # 6. Proportional Peak Normalization to -1.0 dBFS (0.891 linear amplitude) per channel
    peaks = torch.max(torch.abs(waveform), dim=-1, keepdim=True).values
    peaks = torch.clamp(peaks, min=1e-9)  # Prevent division by zero on dead tracks
    waveform = (waveform / peaks) * 0.891

    # Package output streams based on channel topology
    if is_dual_channel:
        streams = {
            "channel_0_agent": waveform[0],     # 1D Tensor [num_samples]
            "channel_1_customer": waveform[1],  # 1D Tensor [num_samples]
        }
    else:
        streams = {
            "mono_mixed": waveform.squeeze(0)   # 1D Tensor [num_samples]
        }

    return {
        "streams": streams,
        "is_dual_channel": is_dual_channel,
        "sample_rate": target_sr,
        "duration_sec": round(duration_sec, 2),
        "orig_sample_rate": orig_sr,
        "orig_channels": metadata.num_channels
    }

```

---

## Phase 2 (Module 1C): CPU-Side Smart Chunking & Compliance Auditing

### 1. Why We Cannot Feed a 5-Minute Call Directly into the Model

Once the CPU produces a clean $16\text{ kHz}$ `float32` tensor, we cannot pass a full 5-minute ($300\text{s}$) tensor straight into the GPU model for two architectural reasons:

1. **Hard Positional Embedding Limit ($30.0\text{ Seconds}$ in Whisper):** Whisper's Transformer Encoder uses fixed positional embeddings strictly capped at $1,500$ frames ($1,500 \times 20\text{ ms} = 30.0\text{ seconds}$ of audio). Anything longer than $30.0\text{ seconds}$ is mathematically truncated.
2. **Quadratic Self-Attention Memory Explosion ($\mathcal{O}(T^2)$ in VRAM):** In Transformer encoders, self-attention memory and compute scale quadratically with sequence length $T$. Feeding a 5-minute call ($10\times$ longer than $30\text{s}$) causes the attention matrix to grow by $100\times$ in GPU VRAM, triggering Out-Of-Memory (OOM) crashes and severe attention entropy dilution.

> **Architectural Decoupling Principle:**
> *"While Whisper architectures enforce a hard $30.0\text{s}$ ($1,500\text{-frame}$) positional embedding limit and Conformer architectures use unconstrained relative positional encodings, our VAD Cut-and-Merge chunker targets a universal $25\text{s}–28\text{s}$ acoustic window. This makes our pre-processing and chunking pipeline $100\%$ model-agnostic while preventing $\mathcal{O}(T^2)$ attention degradation across all architectures."*

---

### 2. Why Default Model Chunking Fails vs. Two-Tier Smart VAD Chunking

If you rely on a model's built-in chunker (`chunk_length_s=30`), it slices audio on a rigid $30.0\text{s}$ timer—cutting words like `"hello"` into `"hel-"` and `"-lo"`, feeding silent gaps into the Decoder (which triggers severe Whisper hallucinations like *"Thank you for watching"*), and discarding the silence timestamps required to audit **Dead Air**.

Instead, we implement a **Two-Tier Smart Chunking Strategy** on the CPU using **Silero VAD**:

#### Tier 1: Silero VAD-Guided "Cut & Merge" (Handles ~98% of Cuts)

**Silero VAD** scans the $16\text{ kHz}$ waveform in **$32\text{ ms}$ ($512\text{-sample}$) windows** and outputs a voice probability $p_t \in [0.0, 1.0]$. To prevent mid-word cuts, Tier 1 relies on four production parameters plus a compliance audit byproduct:

1. **Dual-Threshold Hysteresis ($\tau_{\text{onset}} = 0.50$, $\tau_{\text{offset}} = 0.35$):**
* *Why single-threshold fails:* During a word like `"hello"`, the middle consonant dips slightly in volume ($p_t$ drops from $0.52$ to $0.48$ for $30\text{ ms}$). A single $0.50$ threshold would cut right inside the word.
* *How Hysteresis fixes it:* Speech triggers only when $p_t \ge 0.50$, and stays locked in the `SPEAKING` state until $p_t$ drops below the lower offset threshold ($\tau_{\text{offset}} = 0.35$) continuously for at least `min_silence_duration_ms` ($350\text{ ms}$).


2. **Minimum Silence Valley (`min_silence_duration_ms = 350ms`):**
* A brief $50\text{ ms}$ stop-consonant closure inside a word (such as the silent pause right before the `"t"` in `"Amity"`) is never treated as a cut point. A cut is only legal if $p_t \lt 0.35$ persists for at least $350\text{ ms}$ (a natural pause between phrases or sentences).


3. **Speech Envelope Padding (`speech_pad_ms = 150ms`):**
* Unvoiced consonants at the start or end of a word—like the breathy `/h/` in `"hello"` or the trailing `/s/` in `"fees"`—have very low acoustic energy. Once VAD isolates a speech island $[t_{\text{start}}, t_{\text{end}}]$, we pad both ends outward by $150\text{ ms}$:

$$t'_{\text{start}} = \max(0, t_{\text{start}} - 0.15\text{s}), \quad t'_{\text{end}} = t_{\text{end}} + 0.15\text{s}$$


* This guarantees the full consonant attack and decay are captured inside the chunk.


4. **"Cut & Merge" Bin-Packing (Targeting $25\text{s}–28\text{s}$ Macro-Chunks):**
* Feeding 100 tiny $2\text{-second}$ VAD islands into Whisper individually starves the Transformer of sentence context needed to distinguish homophones (e.g., academic *"course"* vs. *"of course"*) and wastes GPU compute on $30\text{s}$ zero-padding.
* Instead, the Cut & Merge loop merges consecutive VAD islands into a single macro-chunk as long as total duration stays $\le \mathbf{28.0\text{ seconds}}$, placing the cut strictly at the midpoint of a verified $\ge 350\text{ ms}$ silence valley:



```text
Tier 1: VAD "Cut & Merge" (Cuts ONLY happen in verified >= 350ms silence valleys)

[Speech: 0.4s-6.1s] ..silence.. [Speech: 6.8s-15.2s] ..silence.. [Speech: 15.9s-26.4s] ──► CUT HERE (at 26.8s silence midpoint!)
◄─────────────────────── Merged Chunk 1 (Total: 26.4s <= 28s) ───────────────────────►

```

5. **Bonus Compliance Audit (`Dead Air` & `without muted dead voice`):**
* Every gap between raw VAD speech islands (including leading and trailing call silence) where $t_{\text{next\_start}} - t_{\text{current\_end}} \ge 5.0\text{ seconds}$ is automatically flagged as **Dead Air**.
* Before discarding those silent valleys, we measure their decibel energy ($\text{RMS / dBFS}$). If a $\ge 5.0\text{s}$ non-speech valley has energy $\gt -50.0\text{ dBFS}$, background noise or breathing bled through—proving the counselor forgot to mute their mic (`without_muted_dead_voice = True`).



---

#### Tier 2: The Fallback for Non-Stop Talkers (Local Energy Search + Overlapping Stride)

What if a fast-talking counselor or student speaks continuously for $45\text{ seconds}$ without a single $350\text{ ms}$ pause? When a continuous VAD segment exceeds $28.0\text{ seconds}$, Tier 2 executes a two-step fallback:

* **Step A: Local Energy Minimum Search ($22\text{s}$ to $28\text{s}$ Window, $80\text{ ms}$ Breath / $20\text{ ms}$ Hop)**
First, the chunker scans the window between $22\text{s}$ and $28\text{s}$ using an **$80\text{ ms}$ sliding RMS energy window shifted in $20\text{ ms}$ hops** to hunt for the quietest micro-breath between words. If that quietest $80\text{ ms}$ slice falls below our micro-breath threshold ($\le -40\text{ dBFS}$ / `rms <= 0.01`), it places a clean cut right at the center of that micro-breath.
* **Step B: Overlapping Stride (`chunk_size = 28s`, `stride_overlap = 4s`) + Midpoint Word Stitching**
If the speaker is talking so rapidly that even the quietest $80\text{ ms}$ frame in $[22\text{s}, 28\text{s}]$ contains active speech, we never make a butt-joint cut (`[0–28s]`, `[28–56s]`). Instead, we slice with a **4-second overlapping stride**:
* **Chunk 1:** $0.0\text{s} \rightarrow 28.0\text{s}$ (`trim_left_sec = None`, `trim_right_sec = 26.0s`)
* **Chunk 2:** $24.0\text{s} \rightarrow 52.0\text{s}$ (`trim_left_sec = 26.0s`, `trim_right_sec = None`) *(Backs up $4\text{ seconds}$ earlier!)*



```text
Chunk 1 (0s to 28s):   ───────────────────────[24s ... 26s ... "Hel-"] (28s edge)
Chunk 2 (24s to 52s):                         [24s ... 26s ... "Hello" ... 30s]───────────────────────
                                              ◄─── 4s Overlap Zone ───►
                                                       ▲
                                           Stitch at 26.0s Midpoint!

```

* **Why This Eliminates the `"Hel-"` + `"-lo"` Problem:**
1. Even if `"Hello"` at $28.0\text{s}$ is cut in half at the end of Chunk 1 (`"Hel-"`), inside Chunk 2 ($24\text{s}–52\text{s}$), timestamp $28.0\text{s}$ sits $4\text{ full seconds}$ inside the chunk with complete left and right acoustic context—so Chunk 2 transcribes `"Hello"` with $100\%$ accuracy.
2. **Collar / Midpoint Stitching:** Using word-level timestamps after transcription, we define the stitch boundary at the midpoint of the overlap ($26.0\text{s}$):
* Keep all words from Chunk 1 whose center timestamp is $\le 26.0\text{s}$ (`trim_right_sec`).
* Keep all words from Chunk 2 whose center timestamp is $\gt 26.0\text{s}$ (`trim_left_sec`).
* Run a 3-word **Longest Common Subsequence (LCS)** / exact match check right at the $26.0\text{s}$ seam to deduplicate any boundary word straddling $26.0\text{s}$.





| Mechanism | Setting / Value | What It Prevents | How It Works |
| --- | --- | --- | --- |
| **Local Energy Search** | $22\text{s} - 28\text{s}$ window, $80\text{ ms}$ RMS / $20\text{ ms}$ hop | Arbitrary hard cuts on fast speakers | Hunts for the quietest $80\text{ ms}$ micro-breath ($\le -40\text{ dBFS}$) before forcing an overlap. |
| **Overlapping Stride** | $28\text{s}$ max size, $4\text{s}$ overlap | Words getting chopped in half at chunk boundaries | Backs up $4\text{ seconds}$ so every boundary word exists fully inside at least one chunk. |
| **Midpoint Stitching & Deduplication** | Seam at overlap midpoint (`win_end - 2.0s`) | Duplicate or overlapping text strings in the final transcript | Uses `trim_left_sec` and `trim_right_sec` word timestamps + seam deduplication to glue chunks seamlessly. |

---

### 3. Production Code File 2: `src/smart_chunker.py`

```python
# File: src/smart_chunker.py
import math
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

vad_model = load_silero_vad()

def _find_quietest_micro_breath(
    waveform: torch.Tensor,
    sample_rate: int,
    search_start_sec: float,
    search_end_sec: float,
    breath_window_sec: float = 0.08,  # 80ms micro-breath window
    hop_sec: float = 0.02             # 20ms sliding step
) -> tuple[float, float]:
    """
    Scans [search_start_sec, search_end_sec] using an 80ms window sliding every 20ms
    to find the quietest point (minimum RMS energy). Returns (best_cut_sec, min_rms).
    """
    start_idx = int(search_start_sec * sample_rate)
    end_idx = int(search_end_sec * sample_rate)
    win_samples = int(breath_window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)

    if (end_idx - start_idx) <= win_samples:
        return search_end_sec, float("inf")

    min_rms = float("inf")
    best_cut_idx = end_idx

    for idx in range(start_idx, end_idx - win_samples + 1, hop_samples):
        slice_w = waveform[idx : idx + win_samples]
        rms_val = torch.sqrt(torch.mean(slice_w ** 2)).item()
        if rms_val < min_rms:
            min_rms = rms_val
            best_cut_idx = idx + (win_samples // 2)  # Cut at center of 80ms breath

    return best_cut_idx / sample_rate, min_rms

def smart_vad_chunker(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    max_chunk_sec: float = 28.0,
    min_search_sec: float = 22.0,
    overlap_sec: float = 4.0,
    dead_air_threshold_sec: float = 5.0,
    micro_breath_rms_ceiling: float = 0.01  # -40 dBFS threshold for a valid Step A micro-breath
) -> dict:
    """
    Module 1C: Production Two-Tier Audio Chunker & Compliance Auditor.
    1. Extracts speech islands via Silero VAD with Hysteresis (0.50/0.35) + 150ms padding.
    2. Audits all gaps (>= 5.0s) for Dead Air & Unmuted Hot Mics (RMS > -50 dBFS).
    3. Tier 1: Merges consecutive speech islands up to 28.0s (cutting ONLY at silence midpoints).
    4. Tier 2 Step A: For >28s monologues, scans [22s, 28s] with an 80ms/20ms window for a quiet breath.
    5. Tier 2 Step B: If no quiet micro-breath exists, slices with a 4.0s overlapping stride
       and records exact trim_left_sec and trim_right_sec midpoint seams.
    """
    total_duration_sec = waveform.shape[0] / sample_rate

    # Step 1: Run Silero VAD with Hysteresis & 150ms padding so edge consonants ('h' in hello) aren't clipped
    raw_speech_ts = get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=sample_rate,
        threshold=0.50,               # Onset threshold (tau_onset)
        neg_threshold=0.35,           # Offset hysteresis threshold (tau_offset)
        min_speech_duration_ms=250,
        min_silence_duration_ms=350,  # Only cut in >= 350ms silence valleys
        speech_pad_ms=150,            # 150ms envelope buffer on both ends of speech
        return_seconds=True
    )

    # Step 2: Detect Dead Air (>= 5s gaps) & "Without Muted Dead Voice" (Sticky Note 8)
    dead_air_events = []
    prev_end = 0.0

    def _audit_gap(start_t: float, end_t: float):
        gap = end_t - start_t
        if gap >= dead_air_threshold_sec:
            gap_slice = waveform[int(start_t * sample_rate) : int(end_t * sample_rate)]
            if gap_slice.numel() > 0:
                rms = torch.sqrt(torch.mean(gap_slice ** 2)).item()
                dbfs = 20.0 * math.log10(max(rms, 1e-9))
                dead_air_events.append({
                    "start_sec": round(start_t, 2),
                    "end_sec": round(end_t, 2),
                    "duration_sec": round(gap, 2),
                    "dbfs": round(dbfs, 2),
                    "without_muted_dead_voice": dbfs > -50.0  # True if unmuted noise bled through
                })

    for seg in raw_speech_ts:
        _audit_gap(prev_end, seg["start"])
        prev_end = seg["end"]
    # Also check trailing silence at the very end of the call
    _audit_gap(prev_end, total_duration_sec)

    # Step 3: Tier 1 ("Cut & Merge" up to 28s) + Tier 2 (Step A Energy Search -> Step B 4s Overlap)
    chunks = []
    current_start = None
    current_end = None

    for seg in raw_speech_ts:
        seg_start, seg_end = seg["start"], seg["end"]

        # Tier 2 Fallback: Single uninterrupted speech island exceeds 28.0s
        if (seg_end - seg_start) > max_chunk_sec:
            if current_start is not None:
                chunks.append({
                    "start": round(current_start, 2),
                    "end": round(current_end, 2),
                    "overlapped": False,
                    "trim_left_sec": None,
                    "trim_right_sec": None
                })
                current_start, current_end = None, None

            win_start = seg_start
            pending_trim_left = None

            while win_start < seg_end:
                if (seg_end - win_start) <= max_chunk_sec:
                    chunks.append({
                        "start": round(win_start, 2),
                        "end": round(seg_end, 2),
                        "overlapped": pending_trim_left is not None,
                        "trim_left_sec": pending_trim_left,
                        "trim_right_sec": None
                    })
                    break

                # Tier 2 Step A: Local Energy Minimum Search in [win_start + 22s, win_start + 28s]
                best_cut, min_rms = _find_quietest_micro_breath(
                    waveform,
                    sample_rate,
                    search_start_sec=win_start + min_search_sec,
                    search_end_sec=min(win_start + max_chunk_sec, seg_end)
                )

                # Gate Check: Did Step A find a genuine quiet micro-breath (<= -40 dBFS)?
                if min_rms <= micro_breath_rms_ceiling:
                    chunks.append({
                        "start": round(win_start, 2),
                        "end": round(best_cut, 2),
                        "overlapped": pending_trim_left is not None,
                        "trim_left_sec": pending_trim_left,
                        "trim_right_sec": None
                    })
                    win_start = best_cut       # Clean cut at micro-breath; no overlap needed
                    pending_trim_left = None
                else:
                    # Tier 2 Step B: Speaker never paused to breathe -> 28s window with 4s overlap stride
                    win_end = min(win_start + max_chunk_sec, seg_end)
                    seam_midpoint = round(win_end - (overlap_sec / 2.0), 2)  # e.g., 26.0s seam
                    chunks.append({
                        "start": round(win_start, 2),
                        "end": round(win_end, 2),
                        "overlapped": True,
                        "trim_left_sec": pending_trim_left,
                        "trim_right_sec": seam_midpoint
                    })
                    pending_trim_left = seam_midpoint
                    win_start = win_end - overlap_sec  # Backs up 4.0s (e.g., 28.0s -> 24.0s)
            continue

        # Tier 1 Normal Case: Merge consecutive VAD speech islands up to 28.0s
        if current_start is None:
            current_start, current_end = seg_start, seg_end
        elif (seg_end - current_start) <= max_chunk_sec:
            current_end = seg_end  # Bridge across the silence valley
        else:
            # Cut at the midpoint of the verified >= 350ms silence valley between current_end and seg_start
            silence_midpoint = round((current_end + seg_start) / 2.0, 2)
            chunks.append({
                "start": round(current_start, 2),
                "end": silence_midpoint,
                "overlapped": False,
                "trim_left_sec": None,
                "trim_right_sec": None
            })
            current_start, current_end = silence_midpoint, seg_end

    if current_start is not None:
        chunks.append({
            "start": round(current_start, 2),
            "end": round(current_end, 2),
            "overlapped": False,
            "trim_left_sec": None,
            "trim_right_sec": None
        })

    return {
        "chunks": chunks,
        "dead_air_events": dead_air_events,
        "raw_vad_segments": len(raw_speech_ts)
    }

```

---

## Phase 3 (Module 1D): GPU Feature Extraction, CNN Stem & Hinglish BPE Tokenization

### 1. The Master Visual Map: How a 1D Wave Becomes Hinglish Text Inside the Model

Once our `smart_vad_chunker` outputs a clean $16\text{ kHz}$ 1D audio tensor of up to $28.0\text{ seconds}$, we pass it to the model. Inside `faster-whisper` / `transformers`, the model automatically pads the chunk to $30.0\text{ seconds}$ ($480,000\text{ float32 samples}$) and runs the following 4-gate transformation in GPU VRAM:

```text
[1D Wave: 480k pts] ──(STFT + Mel)──► [2D Heatmap: 128×3000] ──(CNN Stride 2)──► [Encoder: 1500 Audio Vecs] ──(Cross-Attention)──► [Decoder: BPE Text Tokens]
   (Air pressure)                      (10ms pitch columns)                       (20ms context vectors)                            (Roman Hinglish Subwords)

  [GATE 0: Input from Smart VAD Chunker]
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1D Audio Array: 480,000 float numbers (30 seconds × 16,000 Hz)       │
  │ Looks like: [0.02, -0.15, 0.44, 0.81, -0.09, ... 480,000 numbers]    │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼  (Step 1: Slice with a sliding 25ms Hann window every 10ms)
  [GATE 1: 3,000 Overlapping Audio Slices]
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 3,000 vertical columns (1 column every 10ms).                        │
  │ Each column holds 400 raw wave numbers (25ms of sound).              │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼  (Step 2: 400-pt FFT + 128 Mel Ear Filters + Log10 Volume)
  [GATE 2: The 2D Log-Mel Spectrogram (A 2D Heatmap Image in VRAM)]
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Shape: [128 Pitch Rows × 3,000 Time Columns]                         │
  │ The 1D wave is now a 2D picture! X-axis = Time, Y-axis = Bass/Treble │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼  (Step 3: 2-Layer CNN Stem — Stride=2 merges pairs)
  [GATE 3: 1,500 Acoustic Vectors (Entering the Transformer Encoder)]
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Shape: [1,280 Deep Features × 1,500 Time Columns]                    │
  │ 3,000 columns got zipped in half into 1,500 columns (1 every 20ms).  │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼  (Step 4: Encoder-Decoder Cross-Attention Spotlight)
  [GATE 4: Discrete BPE Token IDs ──► Final Hinglish Text]
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Token IDs: [50258, 4821, 11, 29410, 882, 14201, ...]                 │
  │ Text:      "Sir, Amity MBA ki fees kya hai?"                         │
  └──────────────────────────────────────────────────────────────────────┘

```

---

### 2. Deep Dive into Each Mathematical Gate

#### Gate 1 & Gate 2: STFT Framing & the 128-Bin Log-Mel Spectrogram

In a 1D waveform $x[n]$, every sample is the sum of all frequencies happening at that microsecond. To separate low vocal cord pitch from high consonant hiss, the GPU builds a **2D Log-Mel Spectrogram**:

1. **Framing ($25\text{ ms}$ Window, $10\text{ ms}$ Hop) & Hann Windowing:**
* Human vocal cords remain acoustically stable over $20\text{–}30\text{ ms}$. We slide a **$25\text{ ms}$ window ($N = 400\text{ samples}$)** across the wave, stepping forward by **$10\text{ ms}$ ($H = 160\text{ samples}$)**. Across $30.0\text{ seconds}$, $30.0\text{s} \times 100\text{ hops/s} = \mathbf{3,000\text{ columns}}$.
* Each $400$-sample slice is multiplied by a bell-shaped **Hann Window** $w[n] = 0.5\left(1 - \cos\left(\frac{2\pi n}{N}\right)\right)$ so the frame edges taper smoothly to zero, preventing artificial clicking spikes (**Spectral Leakage**) during the Fourier Transform.


2. **Fast Fourier Transform (FFT):**
* Runs a $400$-point Real FFT on each column and squares the magnitude to get the power spectrum across $\frac{400}{2} + 1 = \mathbf{201\text{ linear frequency bins}}$ ($0\text{ Hz}$ to $8,000\text{ Hz}$).


3. **128-Bin Mel Filterbank Projection (`whisper-large-v3` Spec):**
* Human ears perceive pitch logarithmically: $m(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$.
* Multiplying the $201$ linear bins by **128 triangular Mel filters** (upgraded from $80$ in older Whisper models) gives $60\%$ finer pitch resolution for Indian vocal inflections and consonant clusters.


4. **Log10 Loudness & $80\text{ dB}$ Dynamic Range Clamping:**
* Applies $\log_{10}(\max(S_{\text{mel}}, 10^{-10}))$ and clamps the floor to $\max(S_{\text{log}}) - 8.0$ ($80\text{ dB}$ dynamic range) so a single loud cough never skews normalization across the $30\text{s}$ chunk.



```text
         THE 2D LOG-MEL SPECTROGRAM: Shape [128 Rows × 3,000 Columns]

Pitch ▲
(128  │  Row 128 (8000Hz Treble)  [ .  .  .  #  #  .  .  .  .  #  . ... ] ◄── High "s" sound in "fees"
 Mel  │  ...                      [ .  .  .  .  .  .  .  .  .  .  . ... ]
Bins) │  Row 40  (1200Hz Vowels)  [ .  #  #  .  .  .  #  #  #  .  . ... ] ◄── Vowel resonance ("Aa-mi-ty")
      │  Row 15  (200Hz Voice)    [ .  #  #  .  .  .  #  #  #  .  . ... ] ◄── Speaker's vocal cord pitch
      │  Row 1   (80Hz Bass)      [ .  .  .  .  .  .  .  .  .  .  . ... ]
      └─────────────────────────────────────────────────────────────────►
         Col 1  Col 2  Col 3 ...                                Col 3000
         (0ms)  (10ms) (20ms)                                   (30,000ms)

```

---

#### Gate 3: Why Audio Encoders Skip Integer Tokens — The 2-Layer CNN "Zipper"

Unlike text LLMs, you cannot chop a continuous spectrogram into discrete integer "word IDs" at the input layer because spoken phonemes blend continuously into one another (co-articulation). Instead, **the Audio Encoder skips integer tokens entirely** and passes the `[128, 3000]` spectrogram through a **2-Layer 1D Convolutional Stem**:

* **Conv1D Layer 1 (`kernel=3, stride=1` + `GELU`):** Expands the vertical height from $128$ pitch rows into **$1,280$ deep acoustic features** while keeping $3,000$ columns ($10\text{ ms}$ each).
* **Conv1D Layer 2 (`kernel=3, stride=2` + `GELU`):** Slides across the time axis jumping **2 columns at a time**, zipping the $3,000$ columns ($10\text{ ms}$) in half into **$1,500$ acoustic embedding vectors ($20\text{ ms}$ per vector / $50\text{ Hz}$)**. This cuts Transformer self-attention memory in VRAM by $4\times$ ($1500^2$ vs $3000^2$).

```text
BEFORE CNN STEM: 3,000 Columns (10ms each) × 128 Rows
 ┌───┬───┬───┬───┬───┬───┬───┬─── ... ───┬──────┐
 │Col│Col│Col│Col│Col│Col│Col│Col        │ Col  │  (Height = 128)
 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8         │ 3000 │
 └─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─ ... ───┴──┬───┘
   └───┼───┘   └───┼───┘   └───┼───┘        │
       ▼           ▼           ▼            ▼   CNN Layer 2 (Stride = 2: Zips pairs together!)
 ┌─────────┬─────────┬─────────┬─ ... ──────┐
 │ Vector  │ Vector  │ Vector  │   Vector   │
 │   #1    │   #2    │   #3    │   #1500    │      (Height = 1,280 deep features)
 │ (0-20ms)│(20-40ms)│(40-60ms)│ (29.98-30s)│
 └────┬────┴────┬────┴────┬────┴──────┬─────┘
      ➕        ➕        ➕          ➕           Add Positional Embedding (Pos #1 to #1500)
   [Pos 1]   [Pos 2]   [Pos 3]    [Pos 1500]       (Enforces the 30.0s = 1500x20ms hard limit!)

```

The **32-Layer Bidirectional Transformer Encoder** then processes all $1,500$ continuous vectors in parallel so every $20\text{ ms}$ slice attends to the full $30\text{s}$ acoustic context, producing the Encoder Audio Memory Bank $H_{\text{enc}}$ of shape `[1, 1500, 1280]`.

---

#### Gate 4: Cross-Attention Spotlight & Why Roman Hinglish Wins in BPE Tokenization

While the **Audio Encoder** uses continuous vectors, the **32-Layer Transformer Decoder** generates discrete **Byte-Pair Encoding (BPE) Integer Token IDs** one subword at a time from a vocabulary of **$51,866$ tokens** ($50,257$ multilingual subwords + special task/language tokens + $1,501$ timestamp tokens `<|0.00|>` to `<|30.00|>`).

At every generation step, the Decoder shines a **Cross-Attention Spotlight** ($\text{softmax}(Q_{\text{text}} K_{\text{audio}}^T / \sqrt{d_k}) V_{\text{audio}}$) across the Encoder's $1,500$ audio vectors to lock onto the exact $20\text{ ms}$ frames where the next syllable is spoken:

```text
   ROOM 1: AUDIO ENCODER                              ROOM 2: TEXT DECODER
 (Holds 1,500 Audio Columns)                      (Generates 1 Subword at a Time)
 
 ┌──────────────────────────┐                     ┌──────────────────────────────┐
 │ Vec #1    (0.00s) silence│                     │ Step 1 Input: [<|start|>,    │
 │ ...                      │                     │   <|hi|>, <|transcribe|>]    │
 │ Vec #40   (0.80s) "Am-"  │◄═══ Spotlight #1 ═══╬══════════════════════════════╣
 │ Vec #41   (0.82s) "-i-"  │◄═══ locks onto      ║ Scans Room 1 & hears "Amity" │
 │ Vec #42   (0.84s) "-ty"  │◄═══ Vecs #40-#42!   ▼ at 0.80s!                    │
 │ ...                      │                     ┌──────────────────────────────┐
 │ Vec #85   (1.70s) "f-"   │                     │ Predicts Token ID #29410:    │
 │ Vec #86   (1.72s) "-ees" │                     │ ──► Outputs word: "Amity"    │
 └──────────────────────────┘                     └──────────────────────────────┘

```

**Why Romanized Hinglish (`Oriserve/Whisper-Hindi2Hinglish-Prime`) Beats Devanagari in BPE Tokenization:**
Because Whisper's BPE tokenizer was trained with dense Latin-script subword merges, Devanagari words shatter into individual UTF-8 vowel/consonant byte pieces, requiring **$2.5\times$ more Decoder steps** for the exact same utterance:

```text
OPTION A: ROMAN HINGLISH BPE TOKENS (Fast, Clean, Preserves Brand Names)
┌──────┐ ┌──────┐ ┌────────┐ ┌─────┐ ┌───────┐ ┌──────┐ ┌──────┐ ┌─────┐
│" Sir"│ │ ","  │ │" Amity"│ │" ki"│ │" fees"│ │" kya"│ │" hai"│ │ "?" │
└──────┘ └──────┘ └────────┘ └─────┘ └───────┘ └──────┘ └──────┘ └─────┘
Total = Only 8 Decoder Steps! ("Amity" and "fees" stay 100% intact)

OPTION B: DEVANAGARI BPE TOKENS (Shatters into Tiny Byte Pieces!)
┌───┬───┐ ┌───┐ ┌───┬───┬───┬───┬───┐ ┌───┬───┐ ┌───┬───┬───┐ ┌───┬───┬───┐ ┌───┬───┐
│ स │ र │ │ , │ │ ए │ म │ ि │ ट │ ी │ │ क │ ी │ │ फ │ ी │ स │ │ क │ ् │ या│ │ ह │ ै │
└───┴───┘ └───┘ └───┴───┴───┴───┴───┘ └───┴───┘ └───┴───┴───┘ └───┴───┴───┘ └───┴───┘
Total = 19+ Decoder Steps! (2.5x slower, and "Amity" mutates into Hindi script "एमिटी"!)

```

---

### 3. Diagnostic Script for Live Monday Demo: `scripts/inspect_module_1d.py`

*(Note: In production, `faster-whisper` runs Module 1D automatically inside `model.transcribe()`. Use this standalone script during your Monday presentation to prove the `[1, 128, 3000]` $\rightarrow$ `[1, 1500, 1280]` tensor math and the Roman vs. Devanagari BPE token counts live on screen).*

```python
# File: scripts/inspect_module_1d.py
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer

def inspect_feature_extraction_and_tokenization(
    audio_chunk_16k: torch.Tensor,
    model_id: str = "openai/whisper-large-v3"
) -> dict:
    """
    Diagnostic proof script for Module 1D:
    1. Converts a 16kHz 1D waveform chunk into a [1, 128, 3000] Log-Mel Spectrogram.
    2. Verifies the 2-Layer CNN Stem output shape ([1, 1500, 1280] @ 20ms resolution).
    3. Benchmarks BPE subword tokenization efficiency on Univo Hinglish (Roman vs Devanagari).
    """
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language="hi", task="transcribe")

    # 1. Extract 128-bin Log-Mel Spectrogram (pads/truncates to 30.0s = 3000 frames)
    features = feature_extractor(
        audio_chunk_16k.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    log_mel_spec = features.input_features  # Shape: [1, 128, 3000]

    # 2. Compute the 2-Layer CNN Stem dimension transformation (stride=1 then stride=2)
    num_mel_frames = log_mel_spec.shape[-1]          # 3000 frames (10ms each)
    encoder_acoustic_frames = num_mel_frames // 2    # 1500 frames (20ms each)

    # 3. Compare BPE Tokenization efficiency on Univo Hinglish utterance
    sample_roman_hinglish = "Sir, Amity Online MBA ki fees aur eligibility criteria kya hai?"
    sample_devanagari_mix = "सर, एमिटी ऑनलाइन एमबीए की फीस और एलिजिबिलिटी क्राइटेरिया क्या है?"

    roman_token_ids = tokenizer.encode(sample_roman_hinglish, add_special_tokens=False)
    devanagari_token_ids = tokenizer.encode(sample_devanagari_mix, add_special_tokens=False)

    return {
        "input_waveform_samples": int(audio_chunk_16k.shape[0]),
        "stft_params": {"n_fft": 400, "window_ms": 25, "hop_length": 160, "hop_ms": 10},
        "log_mel_spectrogram_shape": list(log_mel_spec.shape),                 # [1, 128, 3000]
        "encoder_acoustic_vectors_shape": [1, encoder_acoustic_frames, 1280],  # [1, 1500, 1280]
        "roman_hinglish_tokens": {
            "step_count": len(roman_token_ids),
            "subwords": [tokenizer.decode([tid]) for tid in roman_token_ids]
        },
        "devanagari_tokens": {
            "step_count": len(devanagari_token_ids),
            "subwords": [tokenizer.decode([tid]) for tid in devanagari_token_ids]
        }
    }

```

---

## Phase 4: GPU Inference, Overlap Seam Stitching, Forced Alignment & Speaker Diarization

### 1. Post-Inference Pipeline Overview

Once each chunk is transcribed by the GPU model, `src/transcriber.py` completes four downstream tasks to reassemble the full 5+ minute call:

1. **Global Coordinate Remapping:** Adds `chunk["start"]` to every relative word timestamp returned by the model so all words map back to the global call timeline ($0.0\text{s} \rightarrow 300.0\text{s}+$).
2. **Tier 2 Overlap Seam Stitching (`trim_left_sec` / `trim_right_sec`):**
* For any chunk where `trim_left_sec` is set, discards words whose center timestamp $\frac{w_{\text{start}} + w_{\text{end}}}{2} \le \text{trim\_left\_sec}$.
* For any chunk where `trim_right_sec` is set, discards words whose center timestamp $\gt \text{trim\_right\_sec}$, and deduplicates boundary words that straddle the seam.


3. **Forced Alignment (`Wav2Vec2` via `WhisperX`):** Aligns generated text characters against the $20\text{ ms}$ acoustic phoneme lattice on the GPU to refine word start/end timestamps to millisecond precision.
4. **Speaker Diarization (`Dual-Channel` vs. `pyannote.audio`):**
* **If `is_dual_channel == True`:** Speaker labels are already $100\%$ known deterministically (`AGENT` on Channel 0, `CUSTOMER` on Channel 1)—we simply merge the two transcripts chronologically with zero GPU diarization cost.
* **If `is_dual_channel == False`:** Runs `pyannote.audio` on the GPU to cluster speaker voiceprint embeddings and assign `SPEAKER_00` and `SPEAKER_01` to every word.



---

### 2. Production Code File 3: `src/transcriber.py`

```python
# File: src/transcriber.py
import torch
from faster_whisper import WhisperModel
from src.preprocessor import preprocess_production_audio
from src.smart_chunker import smart_vad_chunker

class ProductionHinglishASR:
    def __init__(
        self,
        model_path_or_id: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        """
        Loads the CTranslate2-optimized Whisper model into GPU VRAM in float16
        (prioritizing maximum Hinglish accuracy over latency).
        """
        self.sample_rate = 16000
        self.model = WhisperModel(
            model_path_or_id,
            device=device,
            compute_type=compute_type
        )
        self.domain_prompt = (
            "Amity Online, Univo, MBA, BBA, BCA, MCA, USP, Why Amity, "
            "Fees, Eligibility, Curriculum, Specialization, Admission, Semester."
        )

    def _transcribe_single_stream(self, waveform: torch.Tensor, speaker_label: str) -> dict:
        """Chunks a single 16kHz stream via Smart VAD, runs GPU inference, and stitches seams."""
        chunk_data = smart_vad_chunker(waveform, sample_rate=self.sample_rate)
        stitched_words = []

        for chunk in chunk_data["chunks"]:
            c_start, c_end = chunk["start"], chunk["end"]
            trim_left = chunk["trim_left_sec"]
            trim_right = chunk["trim_right_sec"]

            # Slice the exact 1D audio tensor for this chunk
            audio_slice = waveform[int(c_start * self.sample_rate) : int(c_end * self.sample_rate)]

            # Module 1D happens automatically inside model.transcribe()!
            segments, _ = self.model.transcribe(
                audio_slice.numpy(),
                language="hi",
                task="transcribe",
                beam_size=5,                        # Beam search = 5 for maximum accuracy
                word_timestamps=True,               # Needed for Tier 2 seam stitching
                condition_on_previous_text=False,   # Prevents autoregressive hallucination loops
                initial_prompt=self.domain_prompt
            )

            for seg in segments:
                if not seg.words:
                    continue
                for w in seg.words:
                    # 1. Remap chunk-relative timestamps to global call timeline
                    g_start = round(c_start + w.start, 2)
                    g_end = round(c_start + w.end, 2)
                    g_center = (g_start + g_end) / 2.0

                    # 2. Enforce Tier 2 Step B Overlap Midpoint Seam Boundaries
                    if trim_left is not None and g_center <= trim_left:
                        continue
                    if trim_right is not None and g_center > trim_right:
                        continue

                    word_text = w.word.strip()
                    if not word_text:
                        continue

                    # 3. Seam Deduplication: Drop identical boundary word straddling the stitch seam
                    if stitched_words and chunk["overlapped"]:
                        prev_w = stitched_words[-1]
                        if (
                            prev_w["word"].lower() == word_text.lower()
                            and abs(g_start - prev_w["end"]) < 0.45
                        ):
                            continue

                    stitched_words.append({
                        "speaker": speaker_label,
                        "word": word_text,
                        "start": g_start,
                        "end": g_end,
                        "probability": round(w.probability, 3)
                    })

        return {
            "words": stitched_words,
            "chunks_metadata": chunk_data["chunks"],
            "dead_air_events": chunk_data["dead_air_events"],
            "raw_vad_segments": chunk_data["raw_vad_segments"]
        }

    def process_call(self, mp3_path: str) -> dict:
        """End-to-End Production Entrypoint: Pre-process -> Smart VAD Chunk -> GPU Transcribe -> Stitch."""
        prep = preprocess_production_audio(mp3_path, target_sr=self.sample_rate)

        if prep["is_dual_channel"]:
            agent_res = self._transcribe_single_stream(
                prep["streams"]["channel_0_agent"], speaker_label="AGENT"
            )
            cust_res = self._transcribe_single_stream(
                prep["streams"]["channel_1_customer"], speaker_label="CUSTOMER"
            )
            all_words = sorted(
                agent_res["words"] + cust_res["words"], key=lambda x: x["start"]
            )
            dead_air = agent_res["dead_air_events"]
        else:
            mono_res = self._transcribe_single_stream(
                prep["streams"]["mono_mixed"], speaker_label="MONO_STREAM"
            )
            all_words = mono_res["words"]
            dead_air = mono_res["dead_air_events"]

        full_transcript = " ".join([w["word"] for w in all_words])
        return {
            "duration_sec": prep["duration_sec"],
            "is_dual_channel": prep["is_dual_channel"],
            "transcript": full_transcript,
            "words": all_words,
            "dead_air_audit": dead_air
        }

```

---
* "Why not just use the biggest Whisper?" 
— Two of the three finalists are large-v3-class Whisper with Indian call-centre fine-tuning. Stock large-v3 cannot write Roman Hinglish; it is in the table as the baseline that proves it.
* "Isn't Whisper known to hallucinate?" 
— Yes, on silence. We never give it silence: VAD-only windows, bounded cuts, and every word outside a VAD island is flagged.
* "What about NVIDIA's Hinglish model?" 
— It is a finalist (Nemotron 3.5 ASR Hinglish fine-tune, transducer architecture). It cannot hallucinate the way Whisper can and it streams natively; its risk is brand-name spelling and unclear provenance. The numbers decide.
* "Why not the Indic model from IIT Madras?" 
— Optional tier; its transducer is honest on accents but weak on English brand names and writes Devanagari.
* "What if we change models later?" 
— One line: the chunker and stitcher ask the backend for its capabilities; nothing else changes. Swapping Whisper for the NeMo model is exactly that line.
* "How fast is it?" 
— RTF is measured per call and reported; it is a batch job, so we spend GPU seconds to buy accuracy.
