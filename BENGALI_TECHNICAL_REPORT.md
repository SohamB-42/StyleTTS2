# Technical Report: StyleTTS2 Bengali Adaptation for Emotional Storytelling

## 1. Modification Summary (Bengali Adaptation)
The adaptation successfully transitions StyleTTS2 from an English-centric model to a native Bengali pipeline by addressing phonetic and script-level requirements.

*   **Text Frontend & Script Support:** 
    *   **Unicode Integration:** `text_utils.py` was modified to include the Bengali Unicode block (`U+0980` - `U+09FF`) in the `_letters_bengali` variable (Line 7). This allows the model to tokenize native script directly if needed, though the primary pipeline uses IPA.
    *   **IPA-based G2P:** The implementation of `bengali_g2p.py` leverages `espeak-ng` with the `bn` language code (Line 42). This is critical for handling Bengali's complex **conjuncts (juktakkhor)**, as `espeak-ng` decomposes them into their phonetic constituents before they reach the `TextEncoder`.
    *   **Token Expansion:** The `n_token` parameter in `Configs/config_bn_finetune.yml` is increased to **418** (Line 62). This supports the larger phoneme inventory and specific diacritical marks required for Bengali nasalization and vowel shifts.

*   **Semantic Context:**
    *   **Bengali BERT:** Replaced the English PL-BERT with `sagorsarker/bangla-bert-base` in the configuration (Line 37). This provides the `ProsodyPredictor` with true Bengali semantic embeddings, vital for a syllable-timed language.

## 2. Architectural & Performance Optimizations
The current implementation shows significant deviations from the standard StyleTTS2 to accommodate cross-lingual transfer and hardware efficiency.

*   **Dynamic ASR Resizing (Transfer Learning):**
    *   `models.py` includes the `load_ASR_models` function (Lines 595-651) which **dynamically resizes** ASR embedding layers (`asr_s2s.embedding`) to match the new `n_token` size (418). This is the key "bridge" that allows English checkpoints to work with Bengali text.
*   **Mixed Precision & GPU Optimization:**
    *   `train_finetune_accelerate.py` uses `accelerate` with `mixed_precision='bf16'` (Line 35), specifically optimized for RTX 4000 series hardware.
*   **Fine-tuning Strategy:**
    *   `TMA_epoch: 0` and `diff_epoch: 0` in `config_bn_finetune.yml` allow joint training immediately, bypassing the 200-epoch Stage 1 requirement of the original paper.

## 3. Gap Analysis (Storytelling & Emotion)
Specific "English biases" and missing components for high-fidelity storytelling:

*   **Inference Style Randomness:** 
    *   `inference_bengali.py` (Line 79) uses `torch.randn(1, 128)` for style. This results in random prosody. Storytelling requires **Reference Audio** embeddings or **Category-based** emotional embeddings.
*   **Prosodic Mismatch (Stress vs. Syllable):**
    *   Bengali is syllable-timed. English is stress-timed. Pre-trained English weights in the `ProsodyPredictor` may still impose English rhythmic biases until further fine-tuned on long-form narration.
*   **Contextual Limitation:**
    *   `inference_bengali.py` processes sentences in isolation. Storytelling requires **cross-sentence prosody coherence**.

## 4. Strategic Next Steps
Priority tasks to enhance the emotional range for Bengali:

### Task 1: Emotional Reference Style Embedding
*   **Action:** Modify `inference_bengali.py` to accept a `ref_audio` path. 
*   **Detail:** Instead of `randn`, use `model.style_encoder(get_mel(ref_audio))` to extract style from emotional Bengali clips.

### Task 2: Fine-tuning on Emotional Storytelling Datasets
*   **Action:** Incorporate storytelling-specific datasets (e.g., Bengali audiobooks).
*   **Detail:** Freeze the `decoder` and train the `ProsodyPredictor` and `Diffusion Transformer` to capture narration-specific pitch shifts.

### Task 3: Tweak Style Factor Parameters
*   **Action:** Adjust the `noise` scale and `diffusion` steps during inference to amplify emotional range.

### Task 4: Evaluation of Naturalness
*   **Proposed Method:** Use **CMOS (Comparison Mean Opinion Score)** to evaluate the model against standard Bengali TTS engines.
