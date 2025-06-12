import { useState, useEffect } from 'react';

export default function Home() {
  const [images, setImages] = useState([]);

  const handleChange = (e) => {
    const files = Array.from(e.target.files);
    setImages((prev) => [
      ...prev,
      ...files.map((file) => ({
        url: URL.createObjectURL(file),
        name: file.name,
      })),
    ]);
    e.target.value = null;
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
      <input type="file" accept="image/*" multiple onChange={handleChange} />
      <div style={{ display: 'flex', flexWrap: 'wrap', marginTop: '1rem' }}>
        {images.map((img, idx) => (
          <img
            key={idx}
            src={img.url}
            alt={img.name}
            style={{ maxWidth: '200px', marginRight: '1rem', marginBottom: '1rem' }}
          />
        ))}
      </div>
    </div>
  );
}
