import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState({});

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const postImage = async (endpoint) => {
    if (!file) return null;
    const formData = new FormData();
    formData.append('image', file);
    const res = await fetch(`http://localhost:5000/${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    return res.json();
  };

  const runRemoveBackground = async () => {
    const data = await postImage('remove_background');
    if (data) setResult((r) => ({ ...r, remove: data }));
  };

  const runDetectCorners = async () => {
    const data = await postImage('detect_corners');
    if (data) setResult((r) => ({ ...r, corners: data }));
  };

  const runClassifyPiece = async () => {
    const data = await postImage('classify_piece');
    if (data) setResult((r) => ({ ...r, type: data.type }));
  };

  const runEdgeDescriptors = async () => {
    const data = await postImage('edge_descriptors');
    if (data) setResult((r) => ({ ...r, descriptors: data.metrics }));
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>Codex Puzzle</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <div style={{ marginTop: '1rem' }}>
        <button onClick={runRemoveBackground}>Remove Background</button>
        <button onClick={runDetectCorners} style={{ marginLeft: '0.5rem' }}>
          Detect Corners
        </button>
        <button onClick={runClassifyPiece} style={{ marginLeft: '0.5rem' }}>
          Classify Piece
        </button>
        <button onClick={runEdgeDescriptors} style={{ marginLeft: '0.5rem' }}>
          Edge Descriptors
        </button>
      </div>
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
    </div>
  );
}
