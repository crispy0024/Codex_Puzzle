import { useState, useEffect, useRef } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [files, setFiles] = useState([]);
  const [result, setResult] = useState({});
  const [batchResults, setBatchResults] = useState([]);
  const [pieces, setPieces] = useState([]);

  const [images, setImages] = useState([]);
  const [selected, setSelected] = useState([]);
  const [segments, setSegments] = useState({});
  const [loading, setLoading] = useState(false);
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

  const postImage = async (endpoint, imageFile = file) => {
    if (!imageFile) return null;
    const formData = new FormData();
    formData.append('image', imageFile);
    const res = await fetch(`http://localhost:5000/${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    return res.json();
  };

  const runRemoveBackground = async () => {
    setLoading(true);
    try {
      const data = await postImage('remove_background');
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
        const data = await postImage('remove_background', imgFile);
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
            {segments[idx] && (
              <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                {segments[idx].map((piece, pidx) => (
                  <img
                    key={pidx}
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
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
