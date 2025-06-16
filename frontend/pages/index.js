import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

function ImageGrid({ images }) {
  if (!images || images.length === 0) return null;
  return (
    <div className="grid">
      {images.map((img) => (
        <div key={img.id} className="piece">
          <img src={img.src} alt="piece" />
          <div className="pid">#{img.id}</div>
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const [inputFile, setInputFile] = useState(null);
  const [pieces, setPieces] = useState([]);
  const [bgPieces, setBgPieces] = useState([]);
  const [corners, setCorners] = useState({});
  const [types, setTypes] = useState({});
  const [descs, setDescs] = useState({});
  const [manualImg, setManualImg] = useState(null);
  const [status, setStatus] = useState("");

  const handleFile = (e) => {
    if (e.target.files && e.target.files[0]) {
      setInputFile(e.target.files[0]);
    }
  };

  const segmentPieces = async () => {
    if (!inputFile) return;
    setStatus("Segmenting pieces...");
    const form = new FormData();
    form.append("image", inputFile);
    const res = await fetch(`${API_URL}/segment_pieces`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (data.pieces) {
      const out = data.pieces.map((b, i) => ({
        id: i,
        src: `data:image/png;base64,${b}`,
        originalId: i,
      }));
      setPieces(out);
      setStatus("Extraction complete");
    } else {
      setStatus("No pieces found");
    }
  };

  const removeBackground = async () => {
    const outputs = [];
    for (let i = 0; i < pieces.length; i++) {
      const p = pieces[i];
      setStatus(`Removing background (${i + 1}/${pieces.length})...`);
      const blob = await (await fetch(p.src)).blob();
      const form = new FormData();
      form.append("image", blob, "piece.png");
      const res = await fetch(`${API_URL}/remove_background`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      outputs.push({
        id: p.id,
        src: `data:image/png;base64,${data.image}`,
        mask: `data:image/png;base64,${data.mask}`,
        originalId: p.originalId,
      });
    }
    setBgPieces(outputs);
    setStatus("Background removal complete");
  };

  const detectCorners = async () => {
    const cdata = {};
    for (let i = 0; i < bgPieces.length; i++) {
      const p = bgPieces[i];
      setStatus(`Detecting corners (${i + 1}/${bgPieces.length})...`);
      const blob = await (await fetch(p.src)).blob();
      const form = new FormData();
      form.append("image", blob, "piece.png");
      const res = await fetch(`${API_URL}/detect_corners`, {
        method: "POST",
        body: form,
      });
      const d = await res.json();
      cdata[p.id] = d.corners;
    }
    setCorners(cdata);
    setStatus("Corner detection complete");
  };

  const classifyPieces = async () => {
    const pdata = {};
    for (let i = 0; i < bgPieces.length; i++) {
      const p = bgPieces[i];
      setStatus(`Classifying pieces (${i + 1}/${bgPieces.length})...`);
      const blob = await (await fetch(p.src)).blob();
      const form = new FormData();
      form.append("image", blob, "piece.png");
      const res = await fetch(`${API_URL}/classify_piece`, {
        method: "POST",
        body: form,
      });
      const d = await res.json();
      pdata[p.id] = d.type;
    }
    setTypes(pdata);
    setStatus("Classification complete");
  };

  const edgeDescriptors = async () => {
    const edata = {};
    for (let i = 0; i < bgPieces.length; i++) {
      const p = bgPieces[i];
      setStatus(`Computing descriptors (${i + 1}/${bgPieces.length})...`);
      const blob = await (await fetch(p.src)).blob();
      const form = new FormData();
      form.append("image", blob, "piece.png");
      const res = await fetch(`${API_URL}/edge_descriptors`, {
        method: "POST",
        body: form,
      });
      const d = await res.json();
      edata[p.id] = d.metrics;
    }
    setDescs(edata);
    setStatus("Descriptor computation complete");
  };

  const manualAdjust = async () => {
    if (!inputFile) return;
    setStatus("Adjusting image...");
    const form = new FormData();
    form.append("image", inputFile);
    const res = await fetch(`${API_URL}/adjust_image`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    setManualImg(`data:image/png;base64,${data.image}`);
    setStatus("Adjustment complete");
  };

  return (
    <div className="container">
      <h1>Codex Puzzle</h1>

      <section>
        <h2>1. Extract &amp; Clean</h2>
        <p>Upload a puzzle image and isolate each piece.</p>
        <input type="file" onChange={handleFile} />
        <button onClick={segmentPieces}>Run Extraction</button>
        <ImageGrid images={pieces} />
      </section>

      <section>
        <h2>2. Remove Background</h2>
        <p>Use the watershed algorithm to remove background from each piece.</p>
        <button onClick={removeBackground}>Remove Background</button>
        <ImageGrid images={bgPieces} />
      </section>

      <section>
        <h2>3. Detect Corners</h2>
        <p>Find the four main corners of every piece.</p>
        <button onClick={detectCorners}>Detect Corners</button>
        {Object.entries(corners).map(([id, pts]) => (
          <div key={id}>Piece {id}: {JSON.stringify(pts)}</div>
        ))}
      </section>

      <section>
        <h2>4. Classify Piece</h2>
        <p>Label each piece as corner, edge or middle.</p>
        <button onClick={classifyPieces}>Classify</button>
        {Object.entries(types).map(([id, t]) => (
          <div key={id}>Piece {id}: {t}</div>
        ))}
      </section>

      <section>
        <h2>5. Edge Descriptors</h2>
        <p>Compute color and shape metrics per edge.</p>
        <button onClick={edgeDescriptors}>Compute</button>
        {Object.entries(descs).map(([id, d]) => (
          <div key={id}>
            <strong>Piece {id}</strong>
            <pre>{JSON.stringify(d, null, 2)}</pre>
          </div>
        ))}
      </section>

      <section>
        <h2>6. Batch Remove Background</h2>
        <p>Run background removal on all uploaded images.</p>
        <button onClick={removeBackground}>Run Batch</button>
      </section>

      <section>
        <h2>7. Segment Pieces</h2>
        <p>Split an image containing many pieces.</p>
        <button onClick={segmentPieces}>Segment</button>
      </section>

      <section>
        <h2>8. Manual Adjust</h2>
        <p>Overlay a color mask on the selected image.</p>
        <button onClick={manualAdjust}>Adjust</button>
        {manualImg && <img src={manualImg} alt="adjusted" />}
      </section>
      {status && <p className="status">{status}</p>}
    </div>
  );
}
