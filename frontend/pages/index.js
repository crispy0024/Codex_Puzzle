import { useState } from 'react';

export default function Home() {
  const [images, setImages] = useState([]);

  const handleChange = (e) => {
    const files = Array.from(e.target.files);
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setImages((prev) => [...prev, ev.target.result]);
      };
      reader.readAsDataURL(file);
    });
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>Codex Puzzle</h1>
      <p>Welcome to the puzzle application built with Next.js.</p>
      <input type="file" accept="image/*" multiple onChange={handleChange} />
      <div style={{ display: 'flex', flexWrap: 'wrap', marginTop: '1rem' }}>
        {images.map((src, idx) => (
          <img
            key={idx}
            src={src}
            alt={`upload-${idx}`}
            style={{ maxWidth: '200px', marginRight: '1rem', marginBottom: '1rem' }}
          />
        ))}
      </div>
    </div>
  );
}
