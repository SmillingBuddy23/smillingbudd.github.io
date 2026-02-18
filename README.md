 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/subconscious_ai/README.md b/subconscious_ai/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..edb3041f549ba7dcd5c47b707dbc6a63badd5f6b
--- /dev/null
+++ b/subconscious_ai/README.md
@@ -0,0 +1,98 @@
+# Subconscious AI — Edge Edition
+
+Privacy-first, CPU-only local intent prediction that estimates likely next user action 3–7 seconds ahead from lightweight behavioral metadata.
+
+## Features
+- CPU-first GRU inference pipeline (PyTorch CPU).
+- In-memory event capture with no raw character logging.
+- Sliding 3s/7s feature engineering with adaptive throttling.
+- Fast early-exit heuristic + occlusion explanations.
+- Optional ONNX export and int8 quantization workflow.
+- Synthetic-data training presets for rapid reproducibility.
+
+## Quickstart
+```bash
+python -m venv .venv && source .venv/bin/activate
+pip install -r requirements.txt
+python main.py --mode demo --duration 20
+```
+
+Single-command demo:
+```bash
+python main.py --mode demo --duration 20
+```
+
+## Windows & Linux notes
+- Linux: run commands above in bash.
+- Windows PowerShell: `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; python main.py --mode demo`.
+- Packaging: `bash system/packaging.sh` (Linux) or install PyInstaller and run equivalent on Windows.
+
+## Data schema
+Event queue item:
+```json
+{"type":"keyboard|mouse|audio|webcam","ts":1711111111.11,"payload":{}}
+```
+Feature vector: `np.ndarray(float32)` length `N=18`.
+Model `predict()` output:
+```json
+{"predicted_action":"idle","probability":0.91,"explanation":{"features":[["short_term_confidence_score",0.4]],"text":"..."},"timestamp":1711111111.11}
+```
+
+## Explicit feature definitions (normalized to [0,1] at runtime)
+- `typing_speed_3s`, `typing_speed_7s` keys/s, nominal raw range 0–8.
+- `avg_interkey_interval`, `std_interkey_interval` seconds, 0–1.
+- `backspace_rate` ratio 0–1.
+- `burst_count` count/window, 0–20.
+- `avg_burst_length` keys/burst, 0–20.
+- `pause_count` count/window, 0–20.
+- `longest_pause` seconds, 0–5.
+- `mouse_speed_mean` px/s, 0–800.
+- `mouse_jitter` px/s std, 0–600.
+- `click_rate` clicks/s, 0–2.
+- `avg_dwell_time` seconds, 0–5.
+- `voice_energy_mean` normalized energy 0–1 (optional sensor).
+- `pitch_mean` Hz 0–300 (optional sensor).
+- `pitch_std` Hz 0–80 (optional sensor).
+- `eye_focus_confidence` ratio 0–1 (optional sensor).
+- `short_term_confidence_score` ratio 0–1.
+
+## Performance targets
+- Idle CPU target: <5% average.
+- RAM target: <200 MB.
+- Inference target: <200 ms.
+Measure with:
+```bash
+python benchmarks/bench_cpu.py
+python benchmarks/bench_latency.py
+```
+CSV example:
+```text
+metric,p50_ms,p90_ms,p99_ms
+inference_latency,2.1,3.8,6.5
+```
+
+## Training + evaluation
+```bash
+python model/trainer.py --out artifacts/demo_model.pt
+python evaluate.py
+python model/export.py --weights artifacts/demo_model.pt --out artifacts/demo_model.onnx
+```
+Int8 quantization (example): `python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('artifacts/demo_model.onnx','artifacts/demo_model.int8.onnx',weight_type=QuantType.QInt8)"`
+
+Reproduce demo model in <20 min:
+1. `pip install -r requirements.txt`
+2. `python model/trainer.py --out artifacts/demo_model.pt`
+3. `python main.py --mode demo --duration 20`
+
+## Privacy, consent, ethics
+- Non-diagnostic disclaimer: not for medical/mental-health/employment diagnosis.
+- Consent flow required before enabling optional audio/webcam.
+- Default no-disk raw capture policy for keystrokes/audio/webcam.
+- Local-only execution; no cloud APIs.
+- See `docs/ethics_and_privacy.md` and `docs/threat_model.md`.
+
+## Contributing
+PRs welcome. Run `pytest -q` before submitting.
+
+## License
+MIT (`LICENSE`).
 
EOF
) 
