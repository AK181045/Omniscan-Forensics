# ⚖️ OmniScan: AI-Powered Court-Admissible Digital Forensics Platform

> **AI-Powered Court-Admissible Digital Forensics Platform for Deepfake and Document Tampering Detection**  
> Final Year B.E. Project • Dr. D. Y. Patil College of Engineering and Innovation (DYPCOEI), Varale, Talegaon, Pune  
> Department of Computer Engineering • Academic Year 2026-2027  
> **Under Guidance of:** Dr. Suresh Mali, Director (DYPCOEI)  
> **Team:** Anil Balwan Khichar [72342233J] (Lead), Tanpure Siddhi Suresh [72342426J], Bhusari Suvarna Ganesh [72342253C], Bhakti Bibhishan Bhoskar [72455504J]

---

## 🎯 Platform Overview

OmniScan is designed to bridge the critical gap between raw deep learning detection scores and legal admissibility standards. Operating on Universal (Blind) Steganalysis, OmniScan functions independently of file headers or metadata containers. It extracts raw pixel matrices directly and analyzes higher-order statistical moments, sensor noise patterns, and high-frequency spectral artifacts to expose minute edits—even when tampered videos are re-encoded, altered in frequency, or converted back to a static state.

---

## 🔬 Core Engineering Innovations

1. **Header-Agnostic Raw Pixel Extraction Engine**:
   - Bypasses file headers, containers, and EXIF metadata tags.
   - Demuxes raw pixel matrices directly from input bitstreams using decoupled OpenCV and FFmpeg stream readers.
   - Full immunity to modified, spoofed, or stripped file headers.

2. **Multi-Modal Routing Engine**:
   - **Dynamic Video Bitstreams (MP4, AVI, WebM)**: Evaluated across spatial, 2D spectral (2D-FFT/DCT grid leakages), and 3D temporal / Farneback Optical Flow pipelines (detecting frame dropping, velocity jerk, and unnatural pauses).
   - **Static Documents & Certificates (PDF, PNG, JPG, ID Cards, Seals)**: Evaluated via Error Level Analysis (ELA) and Photo-Response Non-Uniformity (PRNU) noise fingerprinting to expose forged signatures and altered document text.

3. **Dual-Domain (2D-FFT / 2D-DCT / DWT Haar) Spectral Decomposition**:
   - Implements Suryawanshi & Mali (2018) higher-order statistical moments ($\mu$, $\sigma$, $S$, $K$) across AC/DC DCT coefficients.
   - Uncovers periodic grid artifacts and spectral leakage introduced by generative AI upscaling layers (GANs, Diffusion models).

4. **Courtroom Admissibility (ISO/IEC 27037 Compliance)**:
   - Immediate SHA-256 cryptographic hash calculated upon file ingestion, logged in an immutable chain-of-custody audit trail.
   - **Explainable AI via Grad-CAM**: Maps deep learning feature activations back to raw spatial coordinates using Grad-CAM visual heatmaps (Selvaraju et al., 2017) to provide verifiable visual evidence.
   - **Automated Court-Admissible Forensic PDF Certificate**: Compiles detection scores, 2D-FFT spectral plots, ELA maps, Grad-CAM overlays, and SHA-256 logs into a structured PDF certificate using ReportLab.

---

## 📚 Research Literature Implemented

- **Suryawanshi & Mali (2018)**: Spatial-domain steganography analysis on DCT statistical moments (mean, std, skewness, kurtosis).
- **Li et al. (2023)**: Deep learning feature-based steganalysis with multi-classifier voting ensemble.
- **De La Croix et al. (2025)**: HSDetect-Net fuzzy-based deep learning steganalysis.
- **Swarnkar & Thomas (2025)**: StegEfficientNetB0 compound-scaled transfer learning.
- **Boroumand et al. (2018)**: SRNet un-quantified residual convolutions for steganalysis.
- **Lukas et al. (2006)**: Photo-Response Non-Uniformity (PRNU) sensor pattern noise extraction.
- **Krawetz (2007)**: Error Level Analysis (ELA) intentional differential recompression mapping.
- **Selvaraju et al. (2017)**: Grad-CAM gradient-weighted class activation mapping for spatial explainability.

---

## 🏗️ Architecture Workflow

```
[ Upload Input Media (Video / Document) ]
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│     Step 1: Ingestion & Integrity Preservation        │
│     • Generate Immutable SHA-256 Hash Log              │
│     • Extract Decoupled Raw Pixels / Frame Matrices    │
└──────────────────────────┬─────────────────────────────┘
                           │
                 [ MIME Auto-Routing ]
                 ┌─────────┴─────────┐
                 ▼                   ▼
     [ Video Steganalysis ]   [ Document Forensics ]
     ├── 2D-FFT / DCT Grid    ├── Error Level Analysis (ELA)
     ├── PRNU Sensor Noise    ├── PRNU Signature Noise
     └── Farneback Flow       └── Layout / OCR Checks
                 │                   │
                 └─────────┬─────────┘
                           │
                           ▼
          [ Dual-Stream Fusion Neural Network ]
          ├── SRNet Residual Noise Extractor
          └── Grad-CAM Explainable Activation Map
                           │
                           ▼
          [ Forensic Evidence Output Engine ]
          ├── Interactive Grad-CAM Heatmap Viewer
          ├── Frequency Anomaly Graphs (2D-FFT/DCT)
          └── Court-Admissible PDF Forensic Certificate
```

---

## 🚀 Running the Project

### Double-Click Launch (Windows)
Simply double-click `RUN_PROJECT.bat` in the repository root. This will:
1. Validate required dependencies.
2. Clean up port `8000`.
3. Launch the FastAPI forensic backend.
4. Open the OmniScan Web Dashboard in your browser.

### Manual Launch
```bash
# Start Backend API
cd api
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Open Web Dashboard
# Open index.html or visual_demo.html in any browser
```

---

## 🧪 Running Forensic Tests

```bash
python -c "
from tests.unit.test_omniscan_forensics import *
test_header_agnostic_ingestion()
test_frequency_analyzer_and_moments()
test_ela_analyzer()
test_prnu_extractor()
test_optical_flow_analyzer()
test_srnet_grad_cam()
test_pdf_certificate_service()
print('All OmniScan tests passed!')
"
```
