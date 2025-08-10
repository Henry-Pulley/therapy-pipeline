#!/usr/bin/env python3
"""
Therapy Pipeline CLI (staged, review-first)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import yaml  # pip install pyyaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent


# ------------------------------ utils ------------------------------ #
def read_cfg():
    """Load config.yaml; fill sensible defaults if keys are missing."""
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        sys.exit("Missing config.yaml in project root.")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    # Replace env var placeholders with actual values from environment
    for section in cfg:
        if isinstance(cfg[section], dict):
            for key, val in cfg[section].items():
                # Skip substitution for hf_token_env - it should remain as the env var name
                if section == 'diarization' and key == 'hf_token_env':
                    continue
                if isinstance(val, str) and val in os.environ:
                    env_val = os.environ[val]
                    # Convert booleans and numbers
                    if env_val.lower() in ('true', 'false'):
                        cfg[section][key] = env_val.lower() == 'true'
                    else:
                        try:
                            cfg[section][key] = int(env_val) if '.' not in env_val else float(env_val)
                        except ValueError:
                            cfg[section][key] = env_val

    # Defaults
    cfg.setdefault("io", {})
    cfg["io"].setdefault("input_dir", "templates/in")
    cfg["io"].setdefault("output_dir", "templates/out")
    cfg["io"].setdefault("templates_dir", "templates")

    cfg.setdefault("preprocess", {})
    cfg["preprocess"].setdefault("sample_rate", 16000)
    cfg["preprocess"].setdefault("mono", True)
    cfg["preprocess"].setdefault("loudnorm", True)

    cfg.setdefault("diarization", {})
    cfg["diarization"].setdefault("hf_token_env", "HF_TOKEN")
    cfg["diarization"].setdefault("max_speakers", 2)

    cfg.setdefault("asr", {})
    cfg["asr"].setdefault("model", "large-v3")
    cfg["asr"].setdefault("language", "en")
    cfg["asr"].setdefault("vad", True)
    cfg["asr"].setdefault("compute_type", "float32")

    cfg.setdefault("note", {})
    cfg["note"].setdefault("mode", "ollama")
    cfg["note"].setdefault("ollama_model", "llama3.1:8b-instruct")
    cfg["note"].setdefault("template", str(ROOT / "templates" / "soap.j2"))

    return cfg


def io_paths(cfg):
    """Return (IN, OUT, TPL) Paths; ensure IN/OUT exist."""
    IN = ROOT / cfg["io"]["input_dir"]
    OUT = ROOT / cfg["io"]["output_dir"]
    TPL = ROOT / cfg["io"]["templates_dir"]
    IN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    return IN, OUT, TPL


def sid_dir(session_id: str, OUT: Path) -> Path:
    """Create/return output subdir for session_id."""
    d = OUT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_audio(IN: Path, session_id: str) -> Path | None:
    """Find input audio whose stem matches session_id in IN/."""
    exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
    matches = [p for p in IN.iterdir() if p.is_file() and p.suffix.lower() in exts and p.stem == session_id]
    if len(matches) > 1:
        print(f"Warning: multiple inputs for {session_id}, using first: {[m.name for m in matches]}", file=sys.stderr)
    return matches[0] if matches else None


def sh(cmd: list[str]):
    """Run a subprocess, showing the command."""
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def require(path: Path, msg: str):
    if not Path(path).exists():
        sys.exit(f"✗ {msg} (missing: {path})")


# ------------------------------ commands ------------------------------ #
def cmd_prep(args):
    """Preprocess input audio → prep.wav (mono, 16kHz, loudnorm)."""
    cfg = read_cfg()
    IN, OUT, _ = io_paths(cfg)
    d = sid_dir(args.session_id, OUT)
    raw = find_audio(IN, args.session_id)
    if not raw:
        sys.exit(f"Put your audio at {IN}/{args.session_id}.<wav|mp3|m4a|aac|flac|ogg>")

    out_wav = d / "prep.wav"
    ff = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(raw), "-ac", "1", "-ar", str(cfg["preprocess"]["sample_rate"])]
    if cfg["preprocess"].get("loudnorm", True):
        ff += ["-af", "loudnorm=I=-24:TP=-2:LRA=11:print_format=summary"]
    ff += [str(out_wav)]
    sh(ff)

    # gate hint
    print(f"✓ Preprocessed → {out_wav}")
    print(f"Review audio, then create the gate file to proceed:\n  touch {d/'APPROVED_PREP'}")


def cmd_diarize(args):
    """Speaker diarization → diarization.rttm (requires HF token)."""
    cfg = read_cfg()
    IN, OUT, _ = io_paths(cfg)
    d = sid_dir(args.session_id, OUT)
    wa = d / "prep.wav"
    require(wa, "Run 'prep' first")

    # Require approval gate
    gate = d / "APPROVED_PREP"
    require(gate, "Approve PREP before diarization (touch APPROVED_PREP)")

    rttm = d / "diarization.rttm"
    token_env = cfg["diarization"]["hf_token_env"] or "HF_TOKEN"
    hf = os.environ.get(token_env, "")
    if not hf:
        sys.exit(f"Set your Hugging Face token env var: export {token_env}=<token>")

    # Use inline Python to keep dependencies contained
    code = f"""from pyannote.audio import Pipeline
import torch
import os
import time
import threading

# Detect available device
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "CUDA GPU"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon GPU (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"Using device: {{device_name}}")
print("Loading diarization pipeline...")

# Progress indicator thread
stop_progress = False
def progress_indicator():
    start_time = time.time()
    while not stop_progress:
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        print(f"\\rDiarization running... [{{mins:02d}}:{{secs:02d}}]", end="", flush=True)
        time.sleep(5)  # Update every 5 seconds

progress_thread = threading.Thread(target=progress_indicator)
progress_thread.start()

try:
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["{token_env}"])
    pipe.to(device)
    print("\\nProcessing audio file (this may take several minutes)...")
    dz = pipe("{wa}", num_speakers={args.speakers})
    open("{rttm}","w").write(dz.to_rttm())
    stop_progress = True
    progress_thread.join()
    print("\\n✓ Wrote {rttm}")
except Exception as e:
    stop_progress = True
    progress_thread.join()
    raise e"""
    sh(["python", "-c", code])
    print(f"Review RTTM if desired, then gate next step:\n  touch {d/'APPROVED_DIARIZATION'}")


def cmd_asr(args):
    """ASR + alignment (WhisperX) → asr.aligned.json (uses diarization.rttm)."""
    cfg = read_cfg()
    IN, OUT, _ = io_paths(cfg)
    d = sid_dir(args.session_id, OUT)
    wa = d / "prep.wav"
    rttm = d / "diarization.rttm"
    require(wa, "Run 'prep' first")
    require(rttm, "Run 'diarize' first")

    # Require approval gate
    gate = d / "APPROVED_DIARIZATION"
    require(gate, "Approve DIARIZATION before ASR (touch APPROVED_DIARIZATION)")

    compute = args.compute or cfg["asr"].get("compute_type", "float32")
    model = cfg["asr"]["model"]
    language = cfg["asr"].get("language", "en")

    cmd = [
        "whisperx", str(wa),
        "--model", model,
        "--language", language,
        "--output_dir", str(d),
        "--align_model", "en",
        "--device", "cpu",
        "--compute_type", compute,
        "--threads", str(args.threads),
        "--batch_size", str(args.batch)
    ]

    # Diarization-aware (WhisperX performs its own diarization when --diarize is passed)
    # Temporarily disabled due to segfault issue
    # if rttm.exists():
    #     cmd += ["--diarize"]
    sh(cmd)

    # WhisperX writes <basename>.json next to prep.wav; standardize name:
    base_json = d / f"{wa.stem}.json"
    out_json = d / "asr.aligned.json"
    if not base_json.exists():
        # Try to find *any* .json whisperx wrote in this dir as a fallback
        cands = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            sys.exit("ASR completed but no JSON found. Check WhisperX output.")
        base_json = cands[0]
    shutil.copyfile(base_json, out_json)
    print(f"✓ ASR complete → {out_json}")
    print("Next, render a human-readable transcript:\n  python cli.py render", args.session_id)


def cmd_render(args):
    """Render transcript.md for human review; scaffold edits.yaml on first run."""
    cfg = read_cfg()
    IN, OUT, _ = io_paths(cfg)
    d = sid_dir(args.session_id, OUT)

    src = d / ("transcript.fixed.json" if args.fixed else "asr.aligned.json")
    require(src, "Missing transcript JSON (run 'asr' or 'apply-edits')")
    data = json.loads(src.read_text())

    md = d / "transcript.md"
    lines = ["# Transcript\n"]
    for seg in data.get("segments", []):
        spk = seg.get("speaker", "SPK")
        t0 = float(seg.get("start", 0.0))
        txt = (seg.get("text", "") or "").strip()
        if not txt:
            continue
        lines.append(f"- [{t0:06.2f}s] **{spk}**: {txt}")
    md.write_text("\n".join(lines))
    print(f"✓ Wrote human-readable transcript → {md}")

    if args.preview:
        shown = 0.0
        print(f"\nPreview (first ~{args.preview} seconds worth of segments):")
        for seg in data.get("segments", []):
            if shown > args.preview:
                break
            print(f"[{float(seg.get('start',0.0)):.2f}s] {seg.get('speaker','SPK')}: {(seg.get('text','') or '').strip()}")
            shown = float(seg.get("end", shown))

    # Scaffold edits.yaml if not present
    ey = d / "edits.yaml"
    if not ey.exists():
        ey.write_text(textwrap.dedent("""\
        # Edit this file to correct speakers or reassign segments, then run:
        #   python cli.py apply-edits <session_id>
        #
        # 1) Swap speakers globally (useful if roles flipped):
        swap_speakers: false
        #
        # 2) Rename labels (optional):
        rename:
          SPEAKER_00: Therapist
          SPEAKER_01: Client
        #
        # 3) Reassign ranges (optional): change speaker for segments fully within [start,end]
        # reassign:
        #   - start: 120.0
        #     end: 140.0
        #     to: Therapist
        """))
        print(f"✍️  Created {ey}. Edit if needed, then run: python cli.py apply-edits {args.session_id}")


def cmd_apply_edits(args):
    """Apply edits.yaml → transcript.fixed.json (speaker renames/swaps/range reassign)."""
    cfg = read_cfg()
    IN, OUT, _ = io_paths(cfg)
    d = sid_dir(args.session_id, OUT)

    src = d / "asr.aligned.json"
    ey = d / "edits.yaml"
    require(src, "Missing asr.aligned.json (run 'asr')")
    require(ey, "Missing edits.yaml (run 'render' first)")

    data = json.loads(src.read_text())
    edits = yaml.safe_load(ey.read_text()) or {}

    # Rename speakers
    rename = edits.get("rename", {}) or {}
    for seg in data.get("segments", []):
        spk = seg.get("speaker", "")
        if spk in rename:
            seg["speaker"] = rename[spk]

    # Global swap (Therapist <-> Client)
    if edits.get("swap_speakers"):
        for seg in data.get("segments", []):
            sp = seg.get("speaker")
            if sp == "Therapist":
                seg["speaker"] = "Client"
            elif sp == "Client":
                seg["speaker"] = "Therapist"

    # Range-based reassignment
    for r in (edits.get("reassign") or []):
        s, e, to = float(r["start"]), float(r["end"]), r["to"]
        for seg in data.get("segments", []):
            st = float(seg.get("start", 0.0))
            en = float(seg.get("end", 0.0))
            if st >= s and en <= e:
                seg["speaker"] = to

    fixed = d / "transcript.fixed.json"
    fixed.write_text(json.dumps(data, indent=2))
    print(f"✓ Applied edits → {fixed}")
    print("Re-render to confirm:\n  python cli.py render", args.session_id, "--fixed")
    print(f"If satisfied, approve transcript:\n  touch {d/'APPROVED_TRANSCRIPT'}")


def cmd_note(args):
    """Generate SOAP note (LLM via Ollama or rules/template)."""
    cfg = read_cfg()
    IN, OUT, TPL = io_paths(cfg)
    d = sid_dir(args.session_id, OUT)

    # Require approval gate
    gate = d / "APPROVED_TRANSCRIPT"
    require(gate, "Approve TRANSCRIPT before generating notes (touch APPROVED_TRANSCRIPT)")

    # Choose transcript source
    src = d / ("transcript.fixed.json" if (d / "transcript.fixed.json").exists() else "asr.aligned.json")
    data = json.loads(src.read_text())

    # Extract simple highlights (top N long/meaningful utterances)
    utts = []
    for seg in data.get("segments", []):
        txt = (seg.get("text", "") or "").strip()
        if not txt:
            continue
        dur = float(seg.get("end", 0.0) - seg.get("start", 0.0))
        score = len(txt.split()) * (1 + 0.1 * max(dur, 0.0))
        utts.append((score, float(seg.get("start", 0.0)), seg.get("speaker", ""), txt))
    utts.sort(key=lambda x: x[0], reverse=True)
    top = utts[:6]
    highlights = [{"speaker": spk, "time": f"{st:0.1f}s", "quote": txt} for _, st, spk, txt in top]

    # Build context
    full_text = " ".join((seg.get("text", "") or "").strip() for seg in data.get("segments", []))
    context = {
        "summary": full_text[:1500],
        "observations": highlights,
        "assessment": "Key themes discussed this session; clinical impressions based on session content.",
        "plan": "- Continue current treatment plan.\n- Assign homework as agreed."
    }

    out_md = d / "note_soap.md"
    mode = args.mode or cfg["note"].get("mode", "ollama")

    if mode == "ollama":
        import requests  # pip install requests
        sys_prompt = (
            "You are a licensed therapist writing a concise, factual SOAP note from the provided session data. "
            "No diagnosis unless explicitly present. Quote the client verbatim where relevant. "
            "Keep tone professional; avoid speculation."
        )
        payload = {
            "model": args.model or cfg["note"].get("ollama_model", "llama3.1:8b-instruct"),
            "prompt": f"{sys_prompt}\n\nDATA:\n{json.dumps(context)}",
            "options": {"temperature": 0.1}
        }
        print("→ Calling Ollama… (ensure `ollama serve` is running)")
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600, stream=True)
        r.raise_for_status()
        text = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text += obj.get("response", "")
        out_md.write_text(text.strip())

    else:
        # Rules/template-only path (no LLM)
        from jinja2 import Template  # pip install jinja2
        tpl_path = Path(cfg["note"].get("template", str(TPL / "soap.j2")))
        if not tpl_path.exists():
            sys.exit(f"Missing template: {tpl_path}")
        tpl = Template(tpl_path.read_text())
        note = tpl.render(
            subjective=context["summary"],
            observations=context["observations"],
            assessment=context["assessment"],
            plan=context["plan"],
        )
        out_md.write_text(note)

    print(f"✓ Wrote SOAP note → {out_md}")


def cmd_status(args):
    """Show per-session artifact & gate status."""
    cfg = read_cfg()
    IN, OUT, _ = io_paths(cfg)
    d = sid_dir(args.session_id, OUT)

    def has(p: str) -> str:
        return "✓" if (d / p).exists() else "—"

    rows = [
        ("prep.wav", has("prep.wav")),
        ("APPROVED_PREP", has("APPROVED_PREP")),
        ("diarization.rttm", has("diarization.rttm")),
        ("APPROVED_DIARIZATION", has("APPROVED_DIARIZATION")),
        ("asr.aligned.json", has("asr.aligned.json")),
        ("transcript.md", has("transcript.md")),
        ("edits.yaml", has("edits.yaml")),
        ("transcript.fixed.json", has("transcript.fixed.json")),
        ("APPROVED_TRANSCRIPT", has("APPROVED_TRANSCRIPT")),
        ("note_soap.md", has("note_soap.md")),
    ]
    print(f"Status for session '{args.session_id}':\n")
    for name, mark in rows:
        print(f"{name:22} {mark}")


# ------------------------------ main ------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Therapy Pipeline (staged, manual review between steps)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prep", help="Preprocess input audio → prep.wav")
    s.add_argument("session_id")
    s.set_defaults(func=cmd_prep)

    s = sub.add_parser("diarize", help="Speaker diarization → diarization.rttm")
    s.add_argument("session_id")
    s.add_argument("--speakers", type=int, default=None, help="Number of speakers (default from config, usually 2)")
    s.set_defaults(func=cmd_diarize)

    s = sub.add_parser("asr", help="ASR + alignment (WhisperX) → asr.aligned.json")
    s.add_argument("session_id")
    s.add_argument("--compute", choices=["float32", "int8"], help="Override compute_type (default from config)")
    s.add_argument("--threads", type=int, default=8, help="WhisperX threads (default 8)")
    s.add_argument("--batch", type=int, default=8, help="WhisperX batch size (default 8)")
    s.set_defaults(func=cmd_asr)

    s = sub.add_parser("render", help="Render transcript.md for human review")
    s.add_argument("session_id")
    s.add_argument("--fixed", action="store_true", help="Render from transcript.fixed.json")
    s.add_argument("--preview", type=int, help="Print a short preview (seconds worth of segments)")
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("apply-edits", help="Apply edits.yaml → transcript.fixed.json")
    s.add_argument("session_id")
    s.set_defaults(func=cmd_apply_edits)

    s = sub.add_parser("note", help="Generate SOAP note (requires APPROVED_TRANSCRIPT)")
    s.add_argument("session_id")
    s.add_argument("--mode", choices=["ollama", "rules"], help="Override note mode (default from config)")
    s.add_argument("--model", help="Ollama model override (e.g., llama3.1:8b-instruct)")
    s.set_defaults(func=cmd_note)

    s = sub.add_parser("status", help="Show artifact & gate status for a session")
    s.add_argument("session_id")
    s.set_defaults(func=cmd_status)

    args = ap.parse_args()

    # If user didn't pass --speakers, pull from config default
    if args.cmd == "diarize" and getattr(args, "speakers", None) is None:
        cfg = read_cfg()
        args.speakers = cfg["diarization"].get("max_speakers", 2)

    args.func(args)

if __name__ == "__main__":
    main()
