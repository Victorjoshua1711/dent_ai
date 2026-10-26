import os
import io
import base64
import time
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template_string
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

# Resolve model path flexibly
def get_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'best.pt'),
        os.path.join(os.getcwd(), 'best.pt'),
        os.path.join(os.path.dirname(script_dir), 'best.pt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return 'best.pt'

MODEL_PATH = get_model_path()
print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
    (30, 144, 255), (0, 191, 255), (135, 206, 250), (70, 130, 180), (123, 104, 238),
    (72, 61, 139), (138, 43, 226), (148, 0, 211), (186, 85, 211), (255, 20, 147),
    (255, 105, 180), (255, 160, 122), (255, 69, 0), (255, 99, 71), (218, 112, 214),
    (238, 130, 238), (255, 222, 173)
]

CLASS_NAMES = [
    "Caries", "Crown", "Filling", "Implant", "Malaligned", "Mandibular Canal", "Missing teeth",
    "Periapical lesion", "Retained root", "Root Canal Treatment", "Root Piece", "Impacted tooth",
    "Maxillary sinus", "Bone Loss", "Fractured teeth", "Permanent Teeth", "Supra Eruption", "TAD",
    "Abutment", "Attrition", "Bone defect", "Gingival former", "Metal band", "Orthodontic brackets",
    "Permanent retainer", "Post-core", "Plating", "Wire", "Cyst", "Root resorption", "Primary teeth"
]

PATHOLOGY_CLASSES = {
    "Caries", "Periapical lesion", "Retained root", "Root Piece", "Bone Loss",
    "Fractured teeth", "Attrition", "Bone defect", "Cyst", "Root resorption"
}

RESTORATION_CLASSES = {
    "Crown", "Filling", "Implant", "Root Canal Treatment", "Post-core",
    "Abutment", "Metal band", "Orthodontic brackets", "Permanent retainer", "Plating", "Wire", "TAD"
}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    return response

def process_dental_image(img_pil, show_boxes=False, fill_alpha=0.45, conf_thresh=0.25, filter_category='all'):
    """
    Process image and highlight the exact affected/damaged areas clearly.
    - show_boxes: If False, omits the confusing outer bounding box rectangles around teeth.
    - fill_alpha: Opacity for the affected area highlight mask (e.g. 0.45).
    """
    t0 = time.time()
    
    # Convert PIL Image to OpenCV BGR
    img_rgb = np.array(img_pil.convert('RGB'))
    orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = orig_bgr.shape[:2]

    # Run YOLO prediction
    results = model.predict(img_pil, conf=conf_thresh)
    res = results[0]

    annotated = orig_bgr.copy()
    overlay = orig_bgr.copy()

    annotations = []
    condition_counts = {}
    detected_classes = set()
    polygons_to_draw = []
    boxes_to_draw = []

    has_boxes = hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0
    has_masks = hasattr(res, 'masks') and res.masks is not None and len(res.masks.xy) > 0

    if has_boxes:
        for i, detection in enumerate(res.boxes):
            cls_idx = int(detection.cls[0].item())
            class_name = res.names[cls_idx] if cls_idx in res.names else (
                CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Class {cls_idx}"
            )

            # Apply category filtering if requested
            if filter_category == 'pathology' and class_name not in PATHOLOGY_CLASSES:
                continue
            elif filter_category == 'restoration' and class_name not in RESTORATION_CLASSES:
                continue

            conf = float(detection.conf[0].item()) if hasattr(detection, 'conf') and detection.conf is not None else 0.0
            rgb = PALETTE[cls_idx % len(PALETTE)]
            bgr = (rgb[2], rgb[1], rgb[0])
            x1, y1, x2, y2 = detection.xyxy[0].tolist()

            ann = {
                "label": class_name,
                "confidence": round(conf, 3),
                "bounding_box": {
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1)
                }
            }

            has_seg_mask = False
            if has_masks and i < len(res.masks.xy) and len(res.masks.xy[i]) > 0:
                seg = res.masks.xy[i]
                pts = np.int32(seg)
                if len(pts) >= 3:
                    cv2.fillPoly(overlay, [pts], bgr)
                    polygons_to_draw.append((pts, bgr, class_name, conf))
                    ann["segmentation"] = [{"x": round(float(pt[0]), 1), "y": round(float(pt[1]), 1)} for pt in seg]
                    has_seg_mask = True

            # If no segmentation mask exists for this object or user explicitly requested boxes
            if (not has_seg_mask) or show_boxes:
                boxes_to_draw.append((int(x1), int(y1), int(x2), int(y2), bgr, class_name, conf, has_seg_mask))

            annotations.append(ann)
            condition_counts[class_name] = condition_counts.get(class_name, 0) + 1
            detected_classes.add((cls_idx, class_name, rgb))

    # Blend the semi-transparent overlay to clearly highlight affected portions
    if len(polygons_to_draw) > 0:
        fill_alpha = max(0.1, min(0.9, fill_alpha))
        cv2.addWeighted(overlay, fill_alpha, annotated, 1.0 - fill_alpha, 0, annotated)

        # Draw smooth, distinct contour border on the affected region
        for pts, bgr, cname, conf in polygons_to_draw:
            cv2.polylines(annotated, [pts], isClosed=True, color=bgr, thickness=2, lineType=cv2.LINE_AA)

    # If outer bounding boxes are explicitly enabled or needed for maskless detections
    for x1, y1, x2, y2, bgr, cname, conf, has_seg in boxes_to_draw:
        if show_boxes or not has_seg:
            thickness = 2 if show_boxes else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, thickness, lineType=cv2.LINE_AA)

    # Build Side Legend panel showing detected findings
    legend_w = 240
    legend = np.zeros((h, legend_w, 3), dtype=np.uint8)
    legend[:] = (20, 24, 32)  # Dark slate background

    # Legend Header
    cv2.putText(legend, "AFFECTED FINDINGS", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.line(legend, (14, 38), (legend_w - 14, 38), (51, 65, 85), 1)

    y_pos = 65
    if len(detected_classes) == 0:
        cv2.putText(legend, "No anomalies detected", (14, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)
    else:
        for cls_idx, cname, rgb in sorted(detected_classes, key=lambda x: x[0]):
            if y_pos > h - 30:
                cv2.putText(legend, f"+ more findings...", (14, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)
                break
            bgr = (rgb[2], rgb[1], rgb[0])
            count = condition_counts.get(cname, 1)
            # Color indicator swatch
            cv2.circle(legend, (26, y_pos), 7, bgr, -1, lineType=cv2.LINE_AA)
            cv2.circle(legend, (26, y_pos), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            # Condition label and count
            label_text = f"{cname} ({count})"
            cv2.putText(legend, label_text, (42, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 245, 249), 1, cv2.LINE_AA)
            y_pos += 30

    # Combine into side-by-side composite: [ Original | Clear Affected Area | Legend ]
    composite_bgr = np.hstack([orig_bgr, annotated, legend])
    composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
    composite_pil = Image.fromarray(composite_rgb)

    elapsed = round(time.time() - t0, 2)
    return composite_pil, annotations, condition_counts, elapsed

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dental AI - Clear Affected Area Detection</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    .dragover { border-color: #38bdf8 !important; background-color: rgba(56, 189, 248, 0.08) !important; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
  
  <!-- Navbar -->
  <header class="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <i class="fa-solid fa-tooth text-white text-xl"></i>
        </div>
        <div>
          <h1 class="text-base sm:text-lg font-bold text-white tracking-wide">Dental AI</h1>
          <p class="text-xs text-slate-400">Panoramic Radiograph Disease & Restoration Segmentation</p>
        </div>
      </div>
      <div class="flex items-center space-x-3">
        <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span class="w-1.5 h-1.5 mr-1.5 bg-emerald-400 rounded-full animate-ping"></span>
          Clear Affected Mode Active
        </span>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
    
    <!-- Title / Instructions -->
    <div class="text-center max-w-3xl mx-auto space-y-2">
      <h2 class="text-2xl sm:text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white via-slate-200 to-cyan-400">
        Clear Dental Affected Area Detection
      </h2>
      <p class="text-slate-400 text-xs sm:text-sm">
        Highlights the exact affected lesion, cavity, or restoration boundary without confusing outer tooth outline boxes.
      </p>
    </div>

    <!-- Upload & Controls Card -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        <!-- Drag & Drop Zone -->
        <div class="lg:col-span-2">
          <div id="dropzone" class="border-2 border-dashed border-slate-700 hover:border-cyan-500/60 rounded-xl p-8 text-center cursor-pointer transition bg-slate-950/40 hover:bg-slate-950/70 group">
            <input type="file" id="fileInput" accept="image/*" class="hidden">
            <div class="space-y-3">
              <div class="w-14 h-14 mx-auto rounded-full bg-slate-800 group-hover:bg-cyan-500/10 text-cyan-400 flex items-center justify-center transition">
                <i class="fa-solid fa-cloud-arrow-up text-2xl"></i>
              </div>
              <div>
                <p class="text-sm font-semibold text-slate-200">
                  <span class="text-cyan-400 underline">Click to upload</span> or drag and drop
                </p>
                <p class="text-xs text-slate-400 mt-1">Panoramic dental X-rays (JPG, PNG, JPEG)</p>
              </div>
              <p id="selectedFileName" class="text-xs font-mono text-cyan-300 hidden"></p>
            </div>
          </div>
        </div>

        <!-- Quick actions -->
        <div class="space-y-4 flex flex-col justify-center border-t lg:border-t-0 lg:border-l border-slate-800 pt-4 lg:pt-0 lg:pl-6">
          <div>
            <h3 class="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Preset Radiograph</h3>
            <button id="loadSampleBtn" class="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition">
              <i class="fa-solid fa-file-image text-cyan-400"></i>
              Load Sample Panoramic X-Ray
            </button>
          </div>

          <div class="pt-2">
            <button id="analyzeBtn" disabled class="w-full inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl text-sm font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/25 disabled:opacity-40 disabled:cursor-not-allowed transition">
              <i class="fa-solid fa-brain"></i>
              <span>Detect Affected Areas</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Detection Options Bar -->
      <div class="border-t border-slate-800 pt-5 grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
        <!-- Display Mode Toggle -->
        <div class="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 space-y-1.5">
          <label class="font-semibold text-slate-300 flex items-center gap-1.5">
            <i class="fa-solid fa-eye text-cyan-400"></i> Display Mode
          </label>
          <div class="flex items-center gap-3 pt-1">
            <label class="inline-flex items-center cursor-pointer text-slate-200">
              <input type="radio" name="boxMode" value="clean" checked class="text-cyan-500 focus:ring-cyan-400">
              <span class="ml-1.5 font-medium">Affected Area Only (Clean)</span>
            </label>
            <label class="inline-flex items-center cursor-pointer text-slate-400">
              <input type="radio" name="boxMode" value="boxes" class="text-cyan-500 focus:ring-cyan-400">
              <span class="ml-1.5">Include Boxes</span>
            </label>
          </div>
        </div>

        <!-- Opacity Slider -->
        <div class="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 space-y-1.5">
          <div class="flex justify-between items-center">
            <label class="font-semibold text-slate-300 flex items-center gap-1.5">
              <i class="fa-solid fa-sliders text-cyan-400"></i> Mask Opacity
            </label>
            <span id="opacityVal" class="text-cyan-400 font-mono">45%</span>
          </div>
          <input type="range" id="opacitySlider" min="15" max="85" value="45" class="w-full accent-cyan-500 cursor-pointer">
        </div>

        <!-- Findings Filter -->
        <div class="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 space-y-1.5">
          <label class="font-semibold text-slate-300 flex items-center gap-1.5">
            <i class="fa-solid fa-filter text-cyan-400"></i> Category Filter
          </label>
          <select id="categoryFilter" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500">
            <option value="all">All Findings (Pathology & Restorations)</option>
            <option value="pathology">Damaged / Pathologies Only (Caries, Lesions, Defects)</option>
            <option value="restoration">Restorations Only (Crowns, Fillings, RCT)</option>
          </select>
        </div>
      </div>

    </div>

    <!-- Loading State -->
    <div id="loadingIndicator" class="hidden bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center space-y-4">
      <div class="inline-block relative">
        <div class="w-16 h-16 rounded-full border-4 border-cyan-500/20 border-t-cyan-400 animate-spin"></div>
        <i class="fa-solid fa-tooth text-cyan-400 text-lg absolute inset-0 flex items-center justify-center"></i>
      </div>
      <div>
        <p class="text-slate-200 font-semibold text-base">Segmenting Affected Areas...</p>
        <p class="text-slate-400 text-xs mt-1">Isolating precise cavity, restoration, and lesion contours</p>
      </div>
    </div>

    <!-- Results Section -->
    <div id="resultsCard" class="hidden space-y-6">
      
      <!-- Stats Summary -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-4">
          <div class="w-12 h-12 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center text-xl">
            <i class="fa-solid fa-bullseye"></i>
          </div>
          <div>
            <p class="text-xs text-slate-400">Affected Sites Identified</p>
            <p id="totalDetections" class="text-2xl font-bold text-white">0</p>
          </div>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-4">
          <div class="w-12 h-12 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center text-xl">
            <i class="fa-solid fa-tags"></i>
          </div>
          <div>
            <p class="text-xs text-slate-400">Condition Types</p>
            <p id="uniqueConditions" class="text-2xl font-bold text-white">0</p>
          </div>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-4">
          <div class="w-12 h-12 rounded-lg bg-purple-500/10 text-purple-400 flex items-center justify-center text-xl">
            <i class="fa-solid fa-stopwatch"></i>
          </div>
          <div>
            <p class="text-xs text-slate-400">Inference Time</p>
            <p id="inferenceTime" class="text-2xl font-bold text-white">0s</p>
          </div>
        </div>
      </div>

      <!-- Detected conditions pills -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <h3 class="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <i class="fa-solid fa-notes-medical text-cyan-400"></i>
          Detected Dental Findings
        </h3>
        <div id="conditionsList" class="flex flex-wrap gap-2"></div>
      </div>

      <!-- Output Image Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div class="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 class="text-base font-bold text-white">Clear Affected Area Result</h3>
            <p class="text-xs text-slate-400">Left: Original Panoramic &nbsp;|&nbsp; Center: Isolated Affected Region &nbsp;|&nbsp; Right: Legend</p>
          </div>
          <div class="flex items-center gap-2">
            <a id="downloadResultBtn" href="#" download="dental_affected_areas.png" class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white transition">
              <i class="fa-solid fa-download"></i>
              Download Result
            </a>
          </div>
        </div>

        <div class="overflow-x-auto rounded-xl border border-slate-800 bg-black/60 flex justify-center p-3">
          <img id="resultImage" src="" alt="Analyzed Dental Result" class="max-w-full h-auto rounded shadow-2xl object-contain border border-slate-700/60" />
        </div>
      </div>
    </div>

    <!-- API Reference for Developers -->
    <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6 space-y-3 text-xs text-slate-400">
      <h4 class="text-sm font-semibold text-slate-300 flex items-center gap-2">
        <i class="fa-solid fa-terminal text-cyan-400"></i>
        Direct API Endpoint
      </h4>
      <p>By default, <code class="text-cyan-300 bg-slate-800 px-1.5 py-0.5 rounded">/coordinates</code> produces clean affected-area segmentation masks without confusing rectangular boxes:</p>
      <pre class="bg-slate-950 p-3 rounded-lg border border-slate-800 text-cyan-300 font-mono overflow-x-auto">curl -X POST http://localhost:5000/coordinates -F "image=@sample_img.jpg" --output output.jpg</pre>
    </div>

  </main>

  <footer class="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
    Dental AI Panoramic Disease Detection &copy; 2026. Built with Flask, Ultralytics YOLOv8 & PyTorch.
  </footer>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const selectedFileName = document.getElementById('selectedFileName');
    const loadSampleBtn = document.getElementById('loadSampleBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsCard = document.getElementById('resultsCard');
    const resultImage = document.getElementById('resultImage');
    const downloadResultBtn = document.getElementById('downloadResultBtn');
    const totalDetections = document.getElementById('totalDetections');
    const uniqueConditions = document.getElementById('uniqueConditions');
    const inferenceTime = document.getElementById('inferenceTime');
    const conditionsList = document.getElementById('conditionsList');
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityVal = document.getElementById('opacityVal');
    const categoryFilter = document.getElementById('categoryFilter');

    let currentFile = null;

    opacitySlider.addEventListener('input', (e) => {
      opacityVal.textContent = `${e.target.value}%`;
    });

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    ['dragleave', 'dragend'].forEach(type => {
      dropzone.addEventListener(type, () => dropzone.classList.remove('dragover'));
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        setFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        setFile(e.target.files[0]);
      }
    });

    function setFile(file) {
      currentFile = file;
      selectedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      selectedFileName.classList.remove('hidden');
      analyzeBtn.disabled = false;
    }

    loadSampleBtn.addEventListener('click', async () => {
      try {
        loadSampleBtn.innerHTML = '<i class="fa-solid fa-spinner animate-spin"></i> Loading...';
        const res = await fetch('/sample');
        if (!res.ok) throw new Error('Failed to load sample');
        const blob = await res.blob();
        const file = new File([blob], 'sample_img.jpg', { type: 'image/jpeg' });
        setFile(file);
      } catch (err) {
        alert('Could not load sample image: ' + err.message);
      } finally {
        loadSampleBtn.innerHTML = '<i class="fa-solid fa-file-image text-cyan-400"></i> Load Sample Panoramic X-Ray';
      }
    });

    analyzeBtn.addEventListener('click', async () => {
      if (!currentFile) return;

      loadingIndicator.classList.remove('hidden');
      resultsCard.classList.add('hidden');
      analyzeBtn.disabled = true;

      const formData = new FormData();
      formData.append('image', currentFile);
      
      const showBoxes = document.querySelector('input[name="boxMode"]:checked').value === 'boxes';
      const fillAlpha = (parseInt(opacitySlider.value) / 100).toFixed(2);
      const filterCat = categoryFilter.value;

      formData.append('show_boxes', showBoxes ? 'true' : 'false');
      formData.append('fill_alpha', fillAlpha);
      formData.append('filter_category', filterCat);

      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errData = await response.json();
          throw new Error(errData.error || 'Prediction failed');
        }

        const data = await response.json();

        // Render stats
        totalDetections.textContent = data.annotations.length;
        const counts = data.condition_counts || {};
        const uniqueKeys = Object.keys(counts);
        uniqueConditions.textContent = uniqueKeys.length;
        inferenceTime.textContent = `${data.elapsed_seconds}s`;

        // Render badges
        conditionsList.innerHTML = '';
        if (uniqueKeys.length === 0) {
          conditionsList.innerHTML = '<span class="text-xs text-slate-400">No matching conditions detected.</span>';
        } else {
          uniqueKeys.forEach(cond => {
            const count = counts[cond];
            const span = document.createElement('span');
            span.className = 'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-slate-800 border border-slate-700 text-cyan-300';
            span.innerHTML = `<span class="w-2 h-2 rounded-full bg-cyan-400"></span>${cond}: <strong class="text-white">${count}</strong>`;
            conditionsList.appendChild(span);
          });
        }

        // Render result image
        resultImage.src = data.image_base64;
        downloadResultBtn.href = data.image_base64;

        resultsCard.classList.remove('hidden');
      } catch (err) {
        alert('Error analyzing image: ' + err.message);
      } finally {
        loadingIndicator.classList.add('hidden');
        analyzeBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/sample')
def get_sample_image():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'sample_img.jpg'),
        os.path.join(os.getcwd(), 'sample_img.jpg'),
        os.path.join(os.path.dirname(script_dir), 'sample_img.jpg'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return send_file(p, mimetype='image/jpeg')
    return jsonify({"error": "Sample image not found"}), 404

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 400

    show_boxes = request.form.get('show_boxes', 'false').lower() == 'true'
    fill_alpha = float(request.form.get('fill_alpha', 0.45))
    conf_thresh = float(request.form.get('conf', 0.25))
    filter_cat = request.form.get('filter_category', 'all')

    composite_img, annotations, counts, elapsed = process_dental_image(
        img,
        show_boxes=show_boxes,
        fill_alpha=fill_alpha,
        conf_thresh=conf_thresh,
        filter_category=filter_cat
    )

    img_io = io.BytesIO()
    composite_img.save(img_io, format='PNG')
    img_io.seek(0)
    img_b64 = "data:image/png;base64," + base64.b64encode(img_io.getvalue()).decode('utf-8')

    return jsonify({
        "status": "success",
        "annotations": annotations,
        "condition_counts": counts,
        "elapsed_seconds": elapsed,
        "image_base64": img_b64
    })

@app.route('/coordinates', methods=['POST'])
def get_coordinates():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 400

    # Default to clean mode (no outer boxes) unless requested
    show_boxes = request.form.get('show_boxes', request.args.get('show_boxes', 'false')).lower() == 'true'
    fill_alpha = float(request.form.get('fill_alpha', request.args.get('fill_alpha', 0.45)))
    conf_thresh = float(request.form.get('conf', request.args.get('conf', 0.25)))

    composite_img, annotations, counts, _ = process_dental_image(
        img,
        show_boxes=show_boxes,
        fill_alpha=fill_alpha,
        conf_thresh=conf_thresh
    )

    result = {
        "annotations": annotations,
        "status": "OPEN"
    }
    print(result)

    img_io = io.BytesIO()
    composite_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
