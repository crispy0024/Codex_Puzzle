import { useState, useEffect } from 'react';

export default function Home() {
  const [images, setImages] = useState([]);

  const handleChange = async (e) => {
    const files = Array.from(e.target.files);
    const results = await Promise.all(
      files.map(async (file) => {
        const formData = new FormData();
        formData.append('image', file);
        const res = await fetch('http://localhost:5000/remove_background', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        return {
          url: `data:image/png;base64,${data.image}`,
          name: file.name,
        };
      })
    );
    setImages((prev) => [...prev, ...results]);
    e.target.value = null;
  };

  useEffect(() => {
    return () => {
      images.forEach((img) => {
        if (img.url.startsWith('blob:')) {
          URL.revokeObjectURL(img.url);
        }
      });
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
