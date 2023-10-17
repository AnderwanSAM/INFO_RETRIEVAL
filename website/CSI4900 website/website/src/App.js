import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);

  const handleImageUpload = (event) => {
    const uploadedImage = event.target.files[0];
    setImage(URL.createObjectURL(uploadedImage));
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Text and Image Information Retrieval</h1>
        <div>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
          />
        </div>
        {image && (
          <div>
            <h2>Input Image:</h2>
            <img src={image} alt="Uploaded" width="300" />
          </div>
        )}
      </header>
    </div>
  );
}

export default App;