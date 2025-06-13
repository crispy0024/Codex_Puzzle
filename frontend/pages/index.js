
import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [files, setFiles] = useState([]);
  const [result, setResult] = useState({});
  const [batchResults, setBatchResults] = useState([]);
  const [pieces, setPieces] = useState([]);

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files);
    setFiles(selected);
    if (selected.length > 0) {
      setFile(selected[0]);
    }
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
    </div>
  );
}
