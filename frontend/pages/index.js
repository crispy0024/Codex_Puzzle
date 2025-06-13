
import { useState } from 'react';

export default function Home() {

  const [images, setImages] = useState([]);
  const [selected, setSelected] = useState([]);
  const [segments, setSegments] = useState({});
  const inputRef = useRef(null);

  const handleFiles = (files) => {
    const newImages = Array.from(files).map((file) => ({
      url: URL.createObjectURL(file),
      name: file.name,
      file: file,
    }));
    setImages((prev) => [...prev, ...newImages]);

  };

  const runRemoveBackground = async () => {
    const data = await postImage('remove_background');
    if (data) setResult((r) => ({ ...r, remove: data }));
  };

  const runBatchRemoveBackground = async () => {
    const outputs = [];
    for (const imgFile of files) {
      const data = await postImage('remove_background', imgFile);
      if (data) {
        outputs.push({ name: imgFile.name, data });
      }
    }
    setBatchResults(outputs);
  };


  const segmentSelected = async () => {
    const form = new FormData();
    selected.forEach((idx) => {
      form.append('files', images[idx].file, images[idx].name);
    });
    const res = await fetch('http://localhost:8000/segment', {
      method: 'POST',
      body: form,
    });
    const data = await res.json();
    const segs = {};
    data.results.forEach((resItem, i) => {
      const idx = selected[i];
      segs[idx] = resItem.pieces.map((p) => `data:image/png;base64,${p}`);
    });
    setSegments((prev) => ({ ...prev, ...segs }));
    setSelected([]);
  };

  const toggleSelect = (idx) => {
    setSelected((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
    );

  };

  const runClassifyPiece = async () => {
    const data = await postImage('classify_piece');
    if (data) setResult((r) => ({ ...r, type: data.type }));
  };

  const runEdgeDescriptors = async () => {
    const data = await postImage('edge_descriptors');
    if (data) setResult((r) => ({ ...r, descriptors: data.metrics }));
  };

  const runSegmentPieces = async () => {
    const data = await postImage('segment_pieces');
    if (data && data.pieces) setPieces(data.pieces);
  };

  return (
    <div className="container">
      <h1>Codex Puzzle</h1>

      <input type="file" multiple accept="image/*" onChange={handleFileChange} />
      <div className="buttons" style={{ marginTop: '1rem' }}>
        <button onClick={runRemoveBackground}>Remove Background</button>
        <button onClick={runDetectCorners}>Detect Corners</button>
        <button onClick={runClassifyPiece}>Classify Piece</button>
        <button onClick={runEdgeDescriptors}>Edge Descriptors</button>
        <button onClick={runBatchRemoveBackground}>Batch Remove Background</button>
        <button onClick={runSegmentPieces}>Segment Pieces</button>
      </div>
      {result.remove && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Segmented Piece</h3>
          <img
            src={`data:image/png;base64,${result.remove.image}`}
            alt="segmented"
            style={{ maxWidth: '200px', marginRight: '1rem' }}


      <button
        onClick={segmentSelected}
        disabled={selected.length === 0}
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
                border: selected.includes(idx)
                  ? '3px solid blue'
                  : '1px solid #ccc',
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

