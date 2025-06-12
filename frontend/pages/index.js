import { useState, useEffect, useRef } from 'react';

export default function Home() {
  const [images, setImages] = useState([]);
  const [selected, setSelected] = useState([]);
  const inputRef = useRef(null);

  const handleFiles = (files) => {
    const newImages = Array.from(files).map((file) => ({
      url: URL.createObjectURL(file),
      name: file.name,
    }));
    setImages((prev) => [...prev, ...newImages]);
  };

  const handleChange = (e) => {
    handleFiles(e.target.files);
    e.target.value = null;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
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
      <p>Welcome to the puzzle application built with Next.js.</p>

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

      <div style={{ display: 'flex', flexWrap: 'wrap', marginTop: '1rem' }}>
        {images.map((img, idx) => (
          <img
            key={idx}
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
        ))}
      </div>
    </div>
  );
}
