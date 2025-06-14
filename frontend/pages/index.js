import { useState, useEffect, useRef } from 'react';
import Draggable from 'react-draggable';

export default function Home() {
  const [file, setFile] = useState(null);
  const [files, setFiles] = useState([]);
  const [result, setResult] = useState({});
  const [batchResults, setBatchResults] = useState([]);
  const [pieces, setPieces] = useState([]);

  const [images, setImages] = useState([]);
  const [selected, setSelected] = useState([]);
  const [segments, setSegments] = useState({});
  const [contours, setContours] = useState({});
  const [loading, setLoading] = useState(false);
  const [thresholdValue, setThresholdValue] = useState(128);
  const [blurValue, setBlurValue] = useState(0);
  const [colorValue, setColorValue] = useState('red');
  const [adjustedImage, setAdjustedImage] = useState(null);
  const [thresholdLow, setThresholdLow] = useState('');
  const [thresholdHigh, setThresholdHigh] = useState('');
  const [kernelSize, setKernelSize] = useState('');
  // canvas with placed pieces and groups
  const [canvasItems, setCanvasItems] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [suggestions, setSuggestions] = useState([]);

  // load canvas layout from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('canvasLayout');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          setCanvasItems(parsed);
        }
      } catch (e) {
        console.error('Failed to parse saved layout', e);
      }
    }
  }, []);

  // persist layout when it changes
  useEffect(() => {
    localStorage.setItem('canvasLayout', JSON.stringify(canvasItems));
  }, [canvasItems]);
  const inputRef = useRef(null);

  const handleFiles = (newFiles) => {
    const arr = Array.from(newFiles);
    setFiles(arr);
    if (arr.length > 0) {
      setFile(arr[0]);
    }
    const imgs = arr.map((f) => ({
      url: URL.createObjectURL(f),
      name: f.name,
      file: f,
    }));
    setImages((prev) => [...prev, ...imgs]);
  };

  const handleChange = (e) => {
    handleFiles(e.target.files);
    e.target.value = null;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const postImage = async (endpoint, imageFile = file, extra = {}) => {
    if (!imageFile) return null;
    const formData = new FormData();
    formData.append('image', imageFile);
    Object.entries(extra).forEach(([key, val]) => {
      if (val !== '') {
        formData.append(key, val);
      }
    });
    const res = await fetch(`http://localhost:5000/${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    return res.json();
  };

  const runRemoveBackground = async () => {
    setLoading(true);
    try {
      const data = await postImage('remove_background', file, {
        threshold_low: thresholdLow,
        threshold_high: thresholdHigh,
        kernel_size: kernelSize,
      });
      if (data) setResult((r) => ({ ...r, remove: data }));
    } finally {
      setLoading(false);
    }
  };

  const runBatchRemoveBackground = async () => {
    setLoading(true);
    try {
      const outputs = [];
      for (const imgFile of files) {
        const data = await postImage('remove_background', imgFile, {
          threshold_low: thresholdLow,
          threshold_high: thresholdHigh,
          kernel_size: kernelSize,
        });
        if (data) {
          outputs.push({ name: imgFile.name, data });
        }
      }
      setBatchResults(outputs);
    } finally {
      setLoading(false);
    }
  };

  const runDetectCorners = async () => {
    setLoading(true);
    try {
      const data = await postImage('detect_corners');
      if (data) setResult((r) => ({ ...r, corners: data }));
    } finally {
      setLoading(false);
    }
  };

  const runClassifyPiece = async () => {
    setLoading(true);
    try {
      const data = await postImage('classify_piece');
      if (data) setResult((r) => ({ ...r, type: data.type }));
    } finally {
      setLoading(false);
    }
  };

  const runEdgeDescriptors = async () => {
    setLoading(true);
    try {
      const data = await postImage('edge_descriptors');
      if (data) setResult((r) => ({ ...r, descriptors: data.metrics }));
    } finally {
      setLoading(false);
    }
  };

  const runSegmentPieces = async () => {
    setLoading(true);
    try {
      const data = await postImage('segment_pieces');
      if (data && data.pieces) setPieces(data.pieces);
    } finally {
      setLoading(false);
    }
  };

  const runAdjustImage = async () => {
    setLoading(true);
    try {
      if (!file) return;
      const formData = new FormData();
      formData.append('image', file);
      formData.append('threshold', thresholdValue);
      formData.append('blur', blurValue);
      formData.append('color', colorValue);
      const res = await fetch('http://localhost:5000/adjust_image', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data && data.image) setAdjustedImage(`data:image/png;base64,${data.image}`);
    } finally {
      setLoading(false);
    }
  };

  const segmentSelected = async () => {
    setLoading(true);
    try {
      const segs = {};
      for (const idx of selected) {
        const form = new FormData();
        form.append('image', images[idx].file, images[idx].name);
        const res = await fetch('http://localhost:5000/segment_pieces', {
          method: 'POST',
          body: form,
        });
        const data = await res.json();
        if (data && data.pieces) {
          segs[idx] = data.pieces.map((p) => `data:image/png;base64,${p}`);
        }
      }
      setSegments((prev) => ({ ...prev, ...segs }));
      setSelected([]);
    } finally {
      setLoading(false);
    }
  };

  const extractSelected = async () => {
    setLoading(true);
    try {
      const outs = {};
      for (const idx of selected) {
        const form = new FormData();
        form.append('image', images[idx].file, images[idx].name);
        const res = await fetch('http://localhost:5000/extract_filtered_pieces', {
          method: 'POST',
          body: form,
        });
        const data = await res.json();
        if (data && data.contours) {
          outs[idx] = data.contours.map((c) => `data:image/png;base64,${c}`);
        }
      }
      setContours((prev) => ({ ...prev, ...outs }));
      setSelected([]);
    } finally {
      setLoading(false);
    }
  };

  const mergePieces = async (ids) => {
    const res = await fetch('http://localhost:5000/merge_pieces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ piece_ids: ids }),
    });
    const data = await res.json();
    if (data && data.id) {
      setCanvasItems((prev) => {
        const remaining = prev.filter((p) => !ids.includes(p.id));
        return [
          ...remaining,
          {
            id: data.id,
            image: data.image,
            x: data.x,
            y: data.y,
            type: 'group',
          },
        ];
      });
    }
  };

  const fetchSuggestions = async (id) => {
    const res = await fetch('http://localhost:5000/suggest_match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ piece_id: id, edge_index: 0 }),
    });
    const data = await res.json();
    if (data && data.matches) {
      setSuggestions(data.matches);
    }
  };

  const sendFeedback = async (state, action, reward) => {
    await fetch('http://localhost:5000/submit_feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state, action, reward }),
    });
  };

  const acceptSuggestion = (sug) => {
    if (!selectedItem) return;
    const match = canvasItems.find((it) => it.id === sug.piece_id);
    if (!match) return;
    const size = 100;
    const offsets = {
      0: [0, -size],
      1: [size, 0],
      2: [0, size],
      3: [-size, 0],
    };
    const [dx, dy] = offsets[sug.edge_index] || [0, 0];
    const nx = match.x + dx;
    const ny = match.y + dy;
    setCanvasItems((prev) =>
      prev.map((it) => (it.id === selectedItem ? { ...it, x: nx, y: ny } : it))
    );
    sendFeedback(
      { piece_id: selectedItem, target_id: sug.piece_id },
      'accept',
      1
    );
  };

  const rejectSuggestion = (sug) => {
    if (!selectedItem) return;
    sendFeedback(
      { piece_id: selectedItem, target_id: sug.piece_id },
      'reject',
      -1
    );
  };

  const handleDragStop = (id, x, y) => {
    let snapped = false;
    let finalX = x;
    let finalY = y;
    if (id === selectedItem && suggestions.length > 0) {
      for (const sug of suggestions) {
        const match = canvasItems.find((it) => it.id === sug.piece_id);
        if (!match) continue;
        const size = 100;
        const offsets = {
          0: [0, -size],
          1: [size, 0],
          2: [0, size],
          3: [-size, 0],
        };
        const [dx, dy] = offsets[sug.edge_index] || [0, 0];
        const tx = match.x + dx;
        const ty = match.y + dy;
        const dist = Math.hypot(tx - x, ty - y);
        if (dist < 30) {
          finalX = tx;
          finalY = ty;
          snapped = true;
          sendFeedback(
            { piece_id: id, target_id: sug.piece_id },
            'snap',
            1
          );
          break;
        }
      }
    }
    setCanvasItems((prev) =>
      prev.map((it) => (it.id === id ? { ...it, x: finalX, y: finalY } : it))
    );
    if (!snapped) {
      sendFeedback({ piece_id: id }, 'move', 0);
    }
  };

  const undoMerge = async () => {
    const res = await fetch('http://localhost:5000/undo_merge', { method: 'POST' });
    const data = await res.json();
    if (data && Array.isArray(data.items)) {
      setCanvasItems(data.items);
    }
  };

  const toggleSelect = (idx) => {
    setSelected((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
    );
  };

  useEffect(() => {
    return () => {
      images.forEach((img) => URL.revokeObjectURL(img.url));
    };
  }, [images]);

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>Codex Puzzle</h1>

      <div
        style={{
          position: 'relative',
          width: '800px',
          height: '600px',
          border: '1px solid #ccc',
          marginBottom: '1rem',
        }}
      >
        {canvasItems.map((item) => (
          <Draggable
            key={item.id}
            position={{ x: item.x, y: item.y }}
            onStop={(_, data) => handleDragStop(item.id, data.x, data.y)}
          >
            <img
              src={`data:image/png;base64,${item.image}`}
              alt={`item-${item.id}`}
              onClick={() => {
                setSelectedItem(item.id);
                fetchSuggestions(item.id);
              }}
              style={{
                position: 'absolute',
                width: '100px',
                cursor: 'move',
              }}
            />
          </Draggable>
        ))}
      </div>

      {selectedItem && suggestions.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Suggestions for piece {selectedItem}</h3>
          {suggestions.map((s, idx) => (
            <div key={idx} style={{ marginBottom: '0.5rem' }}>
              <span>
                Match piece {s.piece_id} edge {s.edge_index} (score{' '}
                {s.score.toFixed(2)})
              </span>
              <button
                onClick={() => acceptSuggestion(s)}
                style={{ marginLeft: '0.5rem' }}
              >
                Accept
              </button>
              <button
                onClick={() => rejectSuggestion(s)}
                style={{ marginLeft: '0.5rem' }}
              >
                Reject
              </button>
            </div>
          ))}
        </div>
      )}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          border: '2px dashed #ccc',
          padding: '2rem',
          textAlign: 'center',
          cursor: 'pointer',
        }}
      >
        <p>Drag & drop images here, or click to select</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleChange}
          style={{ display: 'none' }}
        />
      </div>

      <div className="buttons" style={{ marginTop: '1rem' }}>
        <button onClick={runRemoveBackground} disabled={loading}>Remove Background</button>
        <button onClick={runDetectCorners} disabled={loading}>Detect Corners</button>
        <button onClick={runClassifyPiece} disabled={loading}>Classify Piece</button>
        <button onClick={runEdgeDescriptors} disabled={loading}>Edge Descriptors</button>
        <button onClick={runBatchRemoveBackground} disabled={loading}>Batch Remove Background</button>
        <button onClick={runSegmentPieces} disabled={loading}>Segment Pieces</button>
        <button onClick={runAdjustImage} disabled={loading}>Manual Adjust</button>
      </div>

      <div style={{ marginTop: '1rem' }}>
        <label>
          Lower Threshold:
          <input
            type="number"
            value={thresholdLow}
            onChange={(e) => setThresholdLow(e.target.value)}
            style={{ marginLeft: '0.5rem', width: '60px' }}
          />
        </label>
        <label style={{ marginLeft: '1rem' }}>
          Upper Threshold:
          <input
            type="number"
            value={thresholdHigh}
            onChange={(e) => setThresholdHigh(e.target.value)}
            style={{ marginLeft: '0.5rem', width: '60px' }}
          />
        </label>
        <label style={{ marginLeft: '1rem' }}>
          Kernel Size:
          <input
            type="number"
            value={kernelSize}
            onChange={(e) => setKernelSize(e.target.value)}
            style={{ marginLeft: '0.5rem', width: '60px' }}
          />
        </label>
      </div>

      <div style={{ marginTop: '1rem' }}>
        <label>
          Threshold:
          <input
            type="number"
            value={thresholdValue}
            onChange={(e) => setThresholdValue(e.target.value)}
            style={{ marginLeft: '0.5rem', width: '60px' }}
          />
        </label>
        <label style={{ marginLeft: '1rem' }}>
          Blur:
          <input
            type="number"
            value={blurValue}
            onChange={(e) => setBlurValue(e.target.value)}
            style={{ marginLeft: '0.5rem', width: '60px' }}
          />
        </label>
        <label style={{ marginLeft: '1rem' }}>
          Color:
          <select
            value={colorValue}
            onChange={(e) => setColorValue(e.target.value)}
            style={{ marginLeft: '0.5rem' }}
          >
            <option value="red">Red</option>
            <option value="green">Green</option>
            <option value="blue">Blue</option>
          </select>
        </label>
      </div>

      {loading && <p style={{ marginTop: '1rem' }}>Processing...</p>}

      {result.remove && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Segmented Piece</h3>
          <img
            src={`data:image/png;base64,${result.remove.image}`}
            alt="segmented"
            style={{ maxWidth: '200px', marginRight: '1rem' }}
          />
          <img
            src={`data:image/png;base64,${result.remove.mask}`}
            alt="mask"
            style={{ maxWidth: '200px' }}
          />
        </div>
      )}

      {result.corners && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Corners</h3>
          <img
            src={`data:image/png;base64,${result.corners.image}`}
            alt="corners"
            style={{ maxWidth: '200px' }}
          />
        </div>
      )}

      {result.type && (
        <p style={{ marginTop: '1rem' }}>Piece Type: {result.type}</p>
      )}

      {result.descriptors && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Edge Descriptor Lengths</h3>
          <pre>{JSON.stringify(result.descriptors, null, 2)}</pre>
        </div>
      )}

      {adjustedImage && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Adjusted Image</h3>
          <img src={adjustedImage} alt="adjusted" style={{ maxWidth: '200px' }} />
        </div>
      )}

      {batchResults.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Batch Results</h3>
          {batchResults.map((res, idx) => (
            <div key={idx} style={{ marginBottom: '1rem' }}>
              <p>{res.name}</p>
              <img
                src={`data:image/png;base64,${res.data.image}`}
                alt="segmented"
                style={{ maxWidth: '200px', marginRight: '1rem' }}
              />
              <img
                src={`data:image/png;base64,${res.data.mask}`}
                alt="mask"
                style={{ maxWidth: '200px' }}
              />
            </div>
          ))}
        </div>
      )}

      {pieces.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Segmented Pieces</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap' }}>
            {pieces.map((p, idx) => (
              <img
                key={idx}
                src={`data:image/png;base64,${p}`}
                alt={`piece-${idx}`}
                style={{ maxWidth: '150px', marginRight: '1rem', marginBottom: '1rem' }}
              />
            ))}
          </div>
        </div>
      )}

      <button
        onClick={segmentSelected}
        disabled={loading || selected.length === 0}
        style={{ marginTop: '1rem' }}
      >
        Segment Selected
      </button>

      <button
        onClick={extractSelected}
        disabled={loading || selected.length === 0}
        style={{ marginTop: '1rem', marginLeft: '1rem' }}
      >
        Extract Pieces
      </button>

      <button
        onClick={() => mergePieces(selected)}
        disabled={loading || selected.length < 2}
        style={{ marginTop: '1rem', marginLeft: '1rem' }}
      >
        Merge Selected
      </button>

      <button onClick={undoMerge} style={{ marginTop: '1rem', marginLeft: '1rem' }}>
        Undo Merge
      </button>

      <div style={{ marginTop: '1rem' }}>
        {images.map((img, idx) => (
          <div key={idx} style={{ marginBottom: '2rem' }}>
            <img
              src={img.url}
              alt={img.name}
              onClick={() => toggleSelect(idx)}
              style={{
                width: '150px',
                height: '150px',
                objectFit: 'cover',
                marginRight: '1rem',
                marginBottom: '1rem',
                border: selected.includes(idx) ? '3px solid blue' : '1px solid #ccc',
                cursor: 'pointer',
              }}
            />
            {(segments[idx] || contours[idx]) && (
              <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                {segments[idx] &&
                  segments[idx].map((piece, pidx) => (
                    <img
                      key={`seg-${pidx}`}
                      src={piece}
                      alt={`piece-${pidx}`}
                      style={{
                        width: '100px',
                        height: '100px',
                        objectFit: 'contain',
                        marginRight: '0.5rem',
                        marginBottom: '0.5rem',
                        border: '1px solid #ccc',
                      }}
                    />
                  ))}
                {contours[idx] &&
                  contours[idx].map((piece, pidx) => (
                    <img
                      key={`cnt-${pidx}`}
                      src={piece}
                      alt={`contour-${pidx}`}
                      style={{
                        width: '100px',
                        height: '100px',
                        objectFit: 'contain',
                        marginRight: '0.5rem',
                        marginBottom: '0.5rem',
                        border: '1px solid #ccc',
                      }}
                    />
                  ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
