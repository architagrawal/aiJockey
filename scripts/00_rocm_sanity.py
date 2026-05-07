"""ROCm/CUDA + Demucs + CLAP smoke test on actual GPU.

Usage: python scripts/00_rocm_sanity.py path/to/test_clip.wav
Run on first MI300X session. Budget ~$5.
"""
import sys
from pathlib import Path
import torch
import torchaudio


def main(clip_path: str) -> int:
    print(f"torch: {torch.__version__}")
    print(f"hip:   {getattr(torch.version, 'hip', None)}")
    print(f"cuda:  {torch.version.cuda}")
    print(f"devices: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() = False")
        return 1
    print(f"device 0: {torch.cuda.get_device_name(0)}")

    # Load audio
    p = Path(clip_path)
    if not p.exists():
        print(f"FAIL: clip not found: {p}")
        return 1
    wav, sr = torchaudio.load(str(p))
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    if sr != 44100:
        wav = torchaudio.functional.resample(wav, sr, 44100)
        sr = 44100
    print(f"loaded {p.name}: {wav.shape} @ {sr}Hz")

    # Demucs
    print("\n[Demucs] loading htdemucs...")
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    m = get_model('htdemucs').cuda()
    m.eval()
    x = wav[:, :sr * 10].unsqueeze(0).cuda()  # first 10s only
    with torch.no_grad():
        sources = apply_model(m, x, split=True, overlap=0.25)[0]
    print(f"  stems shape: {sources.shape}, sources: {m.sources}")

    # CLAP
    print("\n[CLAP] loading model + ckpt (downloads on first run)...")
    from laion_clap import CLAP_Module
    c = CLAP_Module(enable_fusion=False)
    c.load_ckpt()
    mono = wav[:, :sr * 10].mean(0, keepdim=True).numpy()
    # CLAP wants 48kHz
    import librosa
    mono_48 = librosa.resample(mono[0], orig_sr=sr, target_sr=48000)
    emb = c.get_audio_embedding_from_data(mono_48[None, :], use_tensor=False)
    print(f"  embedding shape: {emb.shape}")

    print("\nALL SANITY CHECKS PASSED. Proceed to analyze pipeline.")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python scripts/00_rocm_sanity.py path/to/test_clip.wav")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
