## Phase 1 (Modules 1A & 1B): CPU-Side Ingestion & Physical Standardization

### Target Mathematical Input Format ("Inside the Model")

Every modern Speech-to-Text neural network (**Whisper**, **Conformer**, **Wav2Vec2**) expects one exact mathematical input format before feature extraction begins:

* **Data Type:** 1D floating-point tensor (`float32`)
* **Sample Rate:** $16,000\text{ Hz}$ ($16\text{ kHz}$)
* **Topology:** Single-Channel (Mono per stream)
* **Amplitude Range:** Normalized strictly within $[-1.0, +1.0]$

```text
[Raw .mp3 File (Up to 50 MB)]
  │ 
  ▼ 1. Header Inspection (ffprobe / torchaudio.info) -> Reject if < 300s (5 mins) or corrupted 
[MDCT Compressed Stream]
  │ 
  ▼ 2. PCM Decode + Channel Check -> Split if True Dual-Channel (Agent/Customer), else Downmix to Mono 
[Raw PCM Wave (8kHz / 44.1kHz / 48kHz)]
  │ 
  ▼ 3. Kaiser Windowed Sinc Resampler -> Standardized 16,000 Hz (16kHz) 
[16kHz Discrete Signal]
  │ 
  ▼ 4. DC Offset Removal + 80Hz Butterworth High-Pass Filter (Strips 50Hz Indian AC Hum) 
[Filtered Speech Band (80Hz - 7600Hz)]
  │ 
  ▼ 5. Float32 Conversion & Peak Normalization -> [-1.0, +1.0] (Capped at -1.0 dBFS / 0.891)
[Clean Neural-Ready 1D Tensor(s)] ──► Ready for Module 1C (Smart VAD Chunking)

```

---

### What Runs on the CPU (System RAM): Step-by-Step

#### 1. Fast Metadata Pre-Check (`ffprobe` / `torchaudio.info()`)

* **The Action:** Always run an instant container header check in under $5\text{ milliseconds}$ to confirm the file is valid and meets the **5-minute ($\ge 300\text{s}$) business rule** before committing system RAM. *(Note: The 5-minute minimum is a business compliance filter for evaluating sales/counseling calls, not an acoustic limitation of the AI model).*
* **What Fails Without It:** Blindly decoding a $50\text{ MB}$ compressed `.mp3` file into raw `float32` PCM memory causes it to expand by $10\times$ to $15\times$, instantly consuming $\sim 600\text{ MB}$ of system RAM per worker thread on invalid or short calls.

#### 2. PCM Decoding & Channel Topology Routing

Once validated, tools like FFmpeg or `torchaudio` decompress the MP3 into a raw **PCM (Pulse-Code Modulation)** sound wave tensor of shape `[num_channels, num_samples]`. A file is either Mono ($1\text{ channel}$) or Stereo ($2\text{ channels}$):

* **Case A — True Dual-Channel Stereo (Split Routing):** Telephony dialers often record Channel 0 (Counselor/Agent) and Channel 1 (Student/Customer) on separate left/right tracks. We verify that the two channels contain distinct signals and process Channel 0 and Channel 1 **completely independently**. This gives $100\%$ perfect speaker separation for free without needing GPU speaker diarization later.
* **Case B — Mono or Identical "Dual-Mono" Stereo:** If the file has 1 channel, or if both stereo channels contain the exact same mixed audio (correlation $\approx 1.0$), we downmix/keep a single Mono stream and route it to GPU speaker diarization (`pyannote.audio`) after transcription.

#### 3. The $16\text{ kHz}$ Standard & Anti-Aliased Kaiser Window Resampling

In production speech-to-text pipelines, CPUs standardize the audio's **sample rate** (digital snapshots per second)—not its actual pitch—to exactly **$16,000\text{ Hz}$ ($16\text{ kHz}$)**.

* **Why $16\text{ kHz}$ is the Universal AI Standard:** By the Nyquist-Shannon theorem, a $16\text{ kHz}$ sample rate captures frequencies up to $8\text{ kHz}$, which covers the entire fundamental frequency and harmonic formant range of human speech.
* **The Perfect Balance:** It avoids the consonant accuracy loss of raw telephone rates ($8\text{ kHz}$) and eliminates the wasted compute cost of high-definition music rates ($44.1\text{ kHz}$ or $48\text{ kHz}$).
* **The Need for the Kaiser Filter:**
* *Upsampling Danger (Spectral Images):* Telephone lines record at $8\text{ kHz}$ while Whisper expects $16\text{ kHz}$. If you stretch an $8\text{ kHz}$ call to $16\text{ kHz}$ by naively duplicating or linearly interpolating data points, the math creates artificial, high-pitched robotic "ghost" frequencies called **Spectral Images**.
* *Downsampling Danger (Aliasing):* Conversely, if you shrink a $48\text{ kHz}$ recording down to $16\text{ kHz}$ without proper low-pass filtering, high-pitched sounds above $8\text{ kHz}$ (like background hiss or sibilance) do not disappear—they mathematically "fold" backward into the lower human voice band. This is called **Aliasing**, and it permanently corrupts transcription accuracy.


* **How the Kaiser Windowed Sinc Filter Works:**
1. **Mathematical Reconstruction & Cutoff:** Before taking new digital snapshots, the Kaiser-windowed sinc filter applies a steep anti-aliasing roll-off that aggressively strips frequencies above $\sim 7.6\text{ kHz}$ (or $\sim 3.8\text{ kHz}$ when upsampling from $8\text{ kHz}$), ensuring zero ghost images or folded frequencies survive.
2. **The New Snapshots:** The CPU then takes clean, freshly interpolated digital snapshots of that smooth wave at exactly $16,000$ times per second.



#### 4. The Production Noise Rule: DC Offset & $80\text{ Hz}$ High-Pass Filter (No Spectral Denoising!)

* **DC Offset Removal:** Subtracting the mean of the waveform (`waveform - waveform.mean()`) re-centers the electrical signal at $0.0$, preventing asymmetric clipping during normalization.
* **$80\text{ Hz}$ Butterworth High-Pass Filter:** Gently strips sub-bass rumble and the **$50\text{ Hz}$ Indian AC electrical hum** without touching the human fundamental voice frequency ($\ge 85\text{ Hz}$).
* **The Golden Production Rule:** **Never use third-party spectral denoisers (like RNNoise or spectral gating) before feeding audio into Whisper.** Whisper was trained on massive quantities of natural background noise and static; if you feed it natural static, it transcribes smoothly. If you feed it artificially warped, mathematically "hollowed-out" audio from an external denoiser (which creates musical noise artifacts and erases quiet consonants), the neural network gets confused and **hallucinates**.

#### 5. Peak Normalization ($-1.0\text{ dBFS}$)

Finally, the CPU scales the waveform so its highest peak sits at **$-1.0\text{ dBFS}$ ($0.891$ linear amplitude)** within $[-1.0, +1.0]$. This guarantees consistent acoustic energy for the VAD and Whisper encoder while leaving a $1\text{ dB}$ headroom margin to prevent digital clipping.

---

### Complete Code: Modules 1A & 1B (`preprocess_production_audio`)

```python
import torch
import torchaudio
import torchaudio.functional as F

def preprocess_production_audio(
    mp3_path: str,
    target_sr: int = 16000,
    min_duration_sec: float = 300.0
) -> dict:
    """
    Modules 1A & 1B: CPU-Side Ingestion & Physical Standardization.
    1. Instant header check (< 5ms) for file validity and >= 5-minute business rule.
    2. Decodes MP3 to float32 PCM and detects True Dual-Channel vs. Mono/Identical Stereo.
    3. Resamples to 16 kHz via Anti-Aliased Kaiser Windowed Sinc Filter.
    4. Strips DC Offset and 50Hz AC hum via 80Hz Butterworth High-Pass Filter.
    5. Peak-normalizes each active stream to -1.0 dBFS (0.891 amplitude).
    """
    # 1. Fast header check BEFORE loading 50MB file into System RAM
    metadata = torchaudio.info(mp3_path)
    duration_sec = metadata.num_frames / metadata.sample_rate

    if duration_sec < min_duration_sec:  # Sticky Note 1: Business filter (>= 5 mins)
        raise ValueError(
            f"Call duration ({duration_sec:.1f}s) is under the {min_duration_sec}s minimum rule."
        )

    # 2. Decode .mp3 to float32 tensor in [-1.0, 1.0], shape: [num_channels, num_samples]
    waveform, orig_sr = torchaudio.load(mp3_path, normalize=True)

    # 3. Check Channel Topology (Mono vs. Identical Stereo vs. True Dual-Channel Split)
    is_dual_channel = False
    if waveform.shape[0] >= 2:
        # Check if Left (Channel 0) and Right (Channel 1) are genuinely different tracks
        # If max absolute difference > 1e-3, the dialer recorded Agent & Customer separately
        channel_diff = torch.max(torch.abs(waveform[0] - waveform[1])).item()
        if channel_diff > 1e-3:
            is_dual_channel = True
            waveform = waveform[:2, :]  # Keep [2, num_samples] to process independently
        else:
            # Identical dual-mono stereo -> downmix to [1, num_samples]
            waveform = torch.mean(waveform[:2, :], dim=0, keepdim=True)

    # 4. Anti-Aliased Kaiser Window Resampling (e.g., 8000 Hz or 48000 Hz -> 16000 Hz)
    # PyTorch F.resample natively processes all channels in parallel
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

    # 6. Peak Normalization to -1.0 dBFS (0.891 linear amplitude) per channel
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

### Why We Cannot Feed a 5-Minute Call Directly into the Model

Once the CPU produces a clean $16\text{ kHz}$ `float32` tensor, we cannot pass a full 5-minute ($300\text{s}$) tensor straight into the GPU model for two architectural reasons:

1. **Hard Positional Embedding Limit ($30.0\text{ Seconds}$ in Whisper):** Whisper's Transformer Encoder uses fixed learned sinusoidal/absolute positional embeddings strictly capped at $1,500$ frames ($30.0\text{ seconds}$ of audio). Anything longer is truncated or rejected.
2. **Quadratic Self-Attention Memory Explosion ($\mathcal{O}(T^2)$ in VRAM):** In standard Transformer encoders, self-attention memory and compute scale quadratically with sequence length $T$. Feeding a 5-minute call ($10\times$ longer than $30\text{s}$) causes the attention matrix to explode by $100\times$ in GPU VRAM, triggering Out-Of-Memory (OOM) crashes and severe attention dilution.

**Architectural Decoupling Principle:**
*"While Whisper architectures enforce a hard $30.0\text{s}$ absolute positional embedding limit and Conformer architectures use unconstrained relative positional encodings, our VAD Cut-and-Merge chunker targets a universal $25\text{s}–28\text{s}$ acoustic window. This makes our pre-processing and chunking pipeline $100\%$ model-agnostic while preventing $\mathcal{O}(T^2)$ attention degradation across all architectures."*

---

### Tier 1: Silero VAD-Guided "Cut & Merge"

Instead of slicing by the clock every $30\text{ seconds}$ (which chops words like *"hello"* into *"hel-"* and *"-lo"*), we run **Silero VAD** on the CPU. Silero VAD scans the $16\text{ kHz}$ waveform in **$32\text{ ms}$ ($512\text{-sample}$) windows** and outputs a voice probability $p_t \in [0.0, 1.0]$ answering: *"What is the probability that a human voice is active in this exact $32\text{ ms}$ frame?"*

To prevent mid-word cuts, Tier 1 relies on four production parameters plus a compliance audit byproduct:

1. **Dual-Threshold Hysteresis ($\tau_{\text{onset}} = 0.50$, $\tau_{\text{offset}} = 0.35$):**
* *Why single-threshold fails:* During a word like *"hello"*, the middle consonant dips slightly in volume ($p_t$ drops from $0.52$ to $0.48$ for $30\text{ ms}$). A single $0.50$ threshold would cut right inside the word.
* *How Hysteresis fixes it:* Speech only triggers when $p_t \ge 0.50$, and stays locked in the `SPEAKING` state until $p_t$ drops below the lower offset threshold ($\tau_{\text{offset}} = 0.35$) continuously for at least `min_silence_duration_ms`.


2. **Minimum Silence Valley (`min_silence_duration_ms` $= 300\text{ ms}$ to $500\text{ ms}$):**
* A brief $50\text{ ms}$ stop-consonant closure inside a word (such as the silent pause right before the *"t"* in *"Amity"*) is never treated as a cut point. A cut is only legal if $p_t \lt 0.35$ persists for at least $300\text{ ms}$ (a natural pause between phrases or sentences).


3. **Speech Envelope Padding (`speech_pad_ms` $= \pm 150\text{ ms}$):**
* Unvoiced consonants at the start or end of a word—like the breathy `/h/` in *"hello"* or the trailing `/s/` in *"fees"*—have very low acoustic energy. Once VAD isolates a speech island $[t_{\text{start}}, t_{\text{end}}]$, we pad both ends outward by $150\text{ ms}$:

$$t'_{\text{start}} = \max(0, t_{\text{start}} - 0.15\text{s}), \quad t'_{\text{end}} = t_{\text{end}} + 0.15\text{s}$$


* This guarantees the full consonant attack and decay are captured inside the chunk.


4. **"Cut & Merge" Bin-Packing (Targeting $25\text{s}–28\text{s}$ Macro-Chunks):**
* Feeding 100 tiny $2\text{-second}$ VAD islands into Whisper individually starves the Transformer of sentence context needed to distinguish homophones (e.g., academic *"course"* vs. *"of course"*) and wastes GPU compute on padding.
* Instead, the Cut & Merge loop concatenates consecutive VAD islands into a single macro-chunk as long as total duration stays $\le \mathbf{28.0\text{ seconds}}$, placing the cut strictly inside a verified $\ge 300\text{ ms}$ silence valley:



```text
Tier 1: VAD "Cut & Merge" (Cuts ONLY happen in verified >= 300ms silence valleys)

[Speech: 0.4s-6.1s] ..silence.. [Speech: 6.8s-15.2s] ..silence.. [Speech: 15.9s-26.4s] ──► CUT HERE (at 26.8s silence!)
◄─────────────────────── Merged Chunk 1 (Total: 26.0s <= 28s) ───────────────────────►

```

5. **Bonus Connection to Sticky Note 8 (Dead Air & Unmuted Hot Mic Detection):**
* Every gap between raw VAD speech islands where $t_{\text{next\_start}} - t_{\text{current\_end}} \ge 5.0\text{ seconds}$ is automatically flagged as **Dead Air**.
* Before discarding those silent valleys, we measure their decibel energy ($\text{RMS / dBFS}$). If a $\ge 5.0\text{s}$ non-speech valley has energy $\gt -50\text{ dBFS}$, background noise/breathing bled through—proving the counselor forgot to mute their mic (`without_muted_dead_voice = True`)!



---

### Tier 2: The Fallback for Non-Stop Talkers (Overlapping Stride + Word Deduplication)

What if a fast-talking counselor or student speaks continuously for $45\text{ seconds}$ straight without a single $300\text{ ms}$ pause? When a continuous VAD segment exceeds $28.0\text{ seconds}$, Tier 2 executes a two-step fallback:

* **Step A: Local Energy Minimum Search ($22\text{s}$ to $29\text{s}$ Window)**
First, the chunker scans the window between $22\text{s}$ and $29\text{s}$ using a tiny $20\text{ ms}$ sliding RMS energy window to hunt for the quietest micro-pause (such as an $80\text{ ms}$ intake of breath between words). If that quietest $20\text{ ms}$ frame falls below our quiet micro-breath threshold ($\le -40\text{ dBFS}$ / `rms < 0.01`), it places a clean cut right at that micro-breath.
* **Step B: Overlapping Stride (`chunk_size = 28s`, `stride_overlap = 4s`) + Midpoint Word Stitching**
If the speaker is talking so rapidly that even the quietest point in $[22\text{s}, 29\text{s}]$ still contains loud active speech, we never make a butt-joint cut (`[0–28s]`, `[28–56s]`). Instead, we slice with a **4-second overlapping stride**:
* **Chunk 1:** $0.0\text{s} \rightarrow 28.0\text{s}$
* **Chunk 2:** $24.0\text{s} \rightarrow 52.0\text{s}$ *(Backs up $4\text{ seconds}$ earlier!)*



```text
Chunk 1 (0s to 28s):   ───────────────────────[24s ... 26s ... "Hel-"] (28s edge)
Chunk 2 (24s to 52s):                         [24s ... 26s ... "Hello" ... 30s]───────────────────────
                                              ◄─── 4s Overlap Zone ───►
                                                       ▲
                                           Stitch at 26.0s Midpoint!

```

* **Why This Eliminates the "Hel-" + "-lo" Problem:**
1. Even if *"Hello"* at $28.0\text{s}$ is cut in half at the end of Chunk 1 (*"Hel-"*), inside Chunk 2 ($24\text{s}–52\text{s}$), timestamp $28.0\text{s}$ sits $4\text{ full seconds}$ inside the chunk with complete left and right acoustic context—so Chunk 2 transcribes *"Hello"* with $100\%$ accuracy.
2. **Collar / Midpoint Stitching:** Using word-level timestamps after transcription, we define the stitch boundary at the exact midpoint of the overlap ($26.0\text{s}$):
* Keep all words from Chunk 1 whose center timestamp is $\le 26.0\text{s}$.
* Keep all words from Chunk 2 whose center timestamp is $\gt 26.0\text{s}$.
* Run a 3-word **Longest Common Subsequence (LCS)** / fuzzy match right at the $26.0\text{s}$ seam to deduplicate any boundary word straddling $26.0\text{s}$.





| Mechanism | Setting / Value | What It Prevents | How It Works |
| --- | --- | --- | --- |
| **Local Energy Search** | $22\text{s} - 29\text{s}$ window, $20\text{ ms}$ RMS | Arbitrary hard cuts on fast speakers | Hunts for the quietest micro-breath ($\le -40\text{ dBFS}$) before forcing an overlap. |
| **Overlapping Stride** | $28\text{s}$ size, $4\text{s}$ overlap | Words getting chopped in half at chunk boundaries | Backs up $4\text{ seconds}$ so every boundary word exists fully inside at least one chunk. |
| **Midpoint Stitching & LCS** | Seam at $26.0\text{s}$ midpoint | Duplicate or overlapping text strings in the final transcript | Uses word timestamps and 3-word LCS fuzzy matching to glue overlapping chunks seamlessly. |

---

### Complete Code: Module 1C (`smart_vad_chunker`)

```python
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

vad_model = load_silero_vad()

def smart_vad_chunker(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    max_chunk_sec: float = 28.0,
    overlap_sec: float = 4.0,
    dead_air_threshold_sec: float = 5.0,
    micro_breath_rms_ceiling: float = 0.01  # -40 dBFS threshold for a valid Step A micro-breath
) -> dict:
    """
    Module 1C: Production Two-Tier Audio Chunker & Compliance Auditor.
    1. Extracts speech islands via Silero VAD with Hysteresis (0.50/0.35) + 150ms padding.
    2. Audits gaps (>= 5.0s) for Dead Air & Unmuted Hot Mics (RMS > -50 dBFS).
    3. Tier 1: Merges consecutive speech islands up to 28.0s (cutting ONLY at silence midpoints).
    4. Tier 2 Step A: For >28s monologues, scans [22s, 29s] in 20ms windows for a quiet micro-breath.
    5. Tier 2 Step B: If no quiet micro-breath exists, slices with a 4.0s overlapping stride.
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
                dbfs = 20 * torch.log10(torch.tensor(max(rms, 1e-9))).item()
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
                    "stitch_midpoint_sec": None
                })
                current_start, current_end = None, None

            win_start = seg_start
            while win_start < seg_end:
                if (seg_end - win_start) <= max_chunk_sec:
                    chunks.append({
                        "start": round(win_start, 2),
                        "end": round(seg_end, 2),
                        "overlapped": False,
                        "stitch_midpoint_sec": None
                    })
                    break

                # Tier 2 Step A: Local Energy Minimum Search in [win_start + 22s, win_start + 29s]
                search_start = win_start + 22.0
                search_end = min(win_start + 29.0, seg_end)
                min_rms = float("inf")
                best_cut = win_start + max_chunk_sec
                step_sec = 0.02  # 20 ms sliding RMS window

                t_cursor = search_start
                while t_cursor + step_sec <= search_end:
                    idx_s = int(t_cursor * sample_rate)
                    idx_e = int((t_cursor + step_sec) * sample_rate)
                    slice_w = waveform[idx_s:idx_e]
                    if slice_w.numel() > 0:
                        rms_val = torch.sqrt(torch.mean(slice_w ** 2)).item()
                        if rms_val < min_rms:
                            min_rms = rms_val
                            best_cut = t_cursor + (step_sec / 2.0)
                    t_cursor += step_sec

                # Gate Check: Did Step A find a genuine quiet micro-breath (<= -40 dBFS)?
                if min_rms <= micro_breath_rms_ceiling:
                    chunks.append({
                        "start": round(win_start, 2),
                        "end": round(best_cut, 2),
                        "overlapped": False,
                        "stitch_midpoint_sec": None
                    })
                    win_start = best_cut  # Clean cut at micro-breath; no overlap needed
                else:
                    # Tier 2 Step B: Speaker never paused to breathe -> 28s window with 4s overlap stride
                    win_end = min(win_start + max_chunk_sec, seg_end)
                    next_win_start = win_end - overlap_sec  # Backs up 4.0s (e.g., 28.0s -> 24.0s)
                    stitch_midpoint = round(win_end - (overlap_sec / 2.0), 2)  # e.g., 26.0s seam
                    chunks.append({
                        "start": round(win_start, 2),
                        "end": round(win_end, 2),
                        "overlapped": True,
                        "stitch_midpoint_sec": stitch_midpoint
                    })
                    win_start = next_win_start
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
                "stitch_midpoint_sec": None
            })
            current_start, current_end = silence_midpoint, seg_end

    if current_start is not None:
        chunks.append({
            "start": round(current_start, 2),
            "end": round(current_end, 2),
            "overlapped": False,
            "stitch_midpoint_sec": None
        })

    return {
        "chunks": chunks,
        "dead_air_events": dead_air_events,
        "raw_vad_segments": len(raw_speech_ts)
    }

```

---

## Phase 3: GPU-Side Feature Extraction, Neural Inference & Post-Processing (VRAM)

Once Module 1C produces clean, $\le 28\text{s}$ standardized $16\text{ kHz}$ chunks, they are passed to the GPU (VRAM) for neural transcription and enrichment.
our smart VAD chunker outputs a clean $16\text{ kHz}$ 1D audio tensor of up to $28.0\text{ seconds}$ (padded to $30.0\text{ seconds}$ = $480,000\text{ float32 samples}$ for Whisper)

### 1. Log-Mel Spectrogram Generation (4 Mathematical Steps)

The standardized $16\text{ kHz}$ audio chunk is mathematically converted into a **Log-Mel Spectrogram** in four steps:

1. **Slicing (Framing):** Cuts the continuous 1D audio wave into tiny, overlapping **$25\text{-millisecond}$ frames** (sliding forward by a $10\text{ ms}$ hop step).
2. **Finding Frequencies (FFT):** Runs a **Fast Fourier Transform** on each $25\text{ ms}$ frame to decompose the complex wave into exact frequency magnitudes.
3. **Mimicking Human Ears (Mel Scale):** Passes those frequencies through $80$ (or $128$ in Whisper-large-v3) triangular **Mel filterbanks** to match human biological hearing—giving high resolution to low human voice pitches while clumping high pitches together.
4. **Adjusting Volume (Log Scale):** Applies a logarithmic compression ($\log_{10}$) to the energy values so the AI perceives loudness ratios the same way the human brain does (decibels).

*Result:* A **2D visual acoustic map** of the chunk (Time frames on the X-axis, Mel pitch bins on the Y-axis) residing directly in GPU VRAM.

---

### 2. Inside the Neural Network (Why Audio Skips Integer Tokens)

Unlike text LLMs, you cannot chop a continuous spectrogram into discrete "word IDs" before the model sees it because spoken sounds blend continuously together (co-articulation). Therefore, **audio models skip the integer input token step entirely** and use a mini-neural network to translate the 2D spectrogram directly into continuous mathematical vectors:

* **The CNN "Stem" (The Audio Slicer):** The GPU passes the 2D Log-Mel Spectrogram through a 2-layer **1D Convolutional Neural Network (CNN)** with GELU activations and a stride of 2. This CNN slides a window horizontally across the time axis of the spectrogram.
* **Direct to Vector:** As the CNN slides across, it downsamples and compresses every vertical slice of the spectrogram (representing $20\text{ ms}$ of sound per output frame) directly into a dense mathematical embedding vector (e.g., $d_{\text{model}} = 1280$).
* **Positional Encoding:** Because the continuous audio is now a sequence of vectors, the GPU adds sinusoidal **Positional Encodings** ("timestamps") to each vector so the self-attention mechanism knows their exact chronological order.
* **The Transformer Encoder & Decoder:** These continuous acoustic vectors are fed directly into the **Transformer Encoder**. To the Transformer, these vectors look mathematically identical to word embedding vectors in an LLM, allowing the Decoder to autoregressively predict text tokens.

---

### 3. Forced Alignment (Millisecond Word Timestamps)

At this point, the pipeline has raw transcribed text per chunk, plus any overlapping 4-second zones from Tier 2 that require stitching.

* A secondary acoustic model—typically **Wav2Vec2** (via WhisperX)—is executed on the GPU to perform **Forced Alignment** (CTC phoneme alignment).
* It aligns the generated text characters directly against the $20\text{ ms}$ acoustic probability lattice to calculate precise millisecond-level `[start, end]` timestamps for every single word.
* These exact word timestamps are also used to execute the **Tier 2 Step B Midpoint Stitching ($26.0\text{s}$ seam) and 3-word LCS deduplication**.

---

### 4. Speaker Diarization (Identifying the Speakers)

* **If True Dual-Channel Stereo (Detected in Module 1A):** Speaker identity is already $100\%$ known deterministically (`channel_0_agent` vs. `channel_1_customer`), completely bypassing GPU diarization overhead!
* **If Mono / Identical Stereo:** Simultaneously or immediately following alignment, the pipeline runs **`pyannote.audio`** on the GPU to extract speaker voiceprint embeddings, cluster them, and assign `SPEAKER_00` (Counselor) and `SPEAKER_01` (Student) labels to every word timestamp.

---
## Phase 4: Audio Feature Extraction, Acoustic Embeddings & Hinglish BPE Tokenization
* Smart VAD chunker outputs a clean $16\text{ kHz}$ 1D audio tensor of up to $28.0\text{ seconds}$ (padded to $30.0\text{ seconds}$ = $480,000\text{ float32 samples}$ for Whisper).

* In a 1D waveform x[n], every single number is the sum of all frequencies happening at that exact microsecond. To make speech recognizable to a neural network, we must decompose the 1D wave into its underlying frequencies over time using the Short-Time Fourier Transform (STFT).

[1D Wave: 480k pts] ──(STFT + Mel)──► [2D Heatmap: 128×3000] ──(CNN Stride 2)──► [Encoder: 1500 Audio Vecs] ──(Cross-Attention)──► [Decoder: BPE Text Tokens]
   (Air pressure)                      (10ms pitch columns)                       (20ms context vectors)                            (Roman Hinglish Subwords)

   [GATE 0: Input from Smart VAD Chunker]
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1D Audio Array: 480,000 float numbers (30 seconds × 16,000 Hz)       │
  │ Looks like: [0.02, -0.15, 0.44, 0.81, -0.09, ... 480,000 numbers]    │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼  (Step 1: Slice with a sliding 25ms window every 10ms)
  [GATE 1: 3,000 Overlapping Audio Slices]
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 3,000 vertical columns (1 column every 10ms).                        │
  │ Each column holds 400 raw wave numbers (25ms of sound).              │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼  (Step 2: FFT + 128 Mel Ear Filters + Log Volume)
  [GATE 2: The 2D Log-Mel Spectrogram (A 2D Heatmap Image!)]
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
