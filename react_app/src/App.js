import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file');
      return;
    }

    const reader = new FileReader();
    reader.readAsDataURL(selectedFile);
    reader.onload = async () => {
      const imageData = reader.result.split(',')[1]; // Extract base64 data
      try {
        const response = await axios.post(
            'http://localhost:8080/api/predict',
            { image: imageData },
            {
              headers: {
                'Content-Type': 'application/json',
              },
            }
        );
        setResult(response.data);
      } catch (error) {
        console.error('Error uploading file:', error);
        setResult('Error uploading file');
      }
    };
    reader.onerror = (error) => {
      console.error('Error reading file:', error);
    };
  };

  return (
      <div style={{ textAlign: 'center', paddingTop: '20px' }}>
        <h1>Kidney Classification</h1>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginBottom: '20px' }}>
          <input type="file" onChange={handleFileChange} />
          <button onClick={handleUpload}>Predict</button>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'row' }}>
          <div style={{ width: '512px', height: '512px', backgroundColor: '#eee', marginBottom: '20px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {/* Placeholder for the image */}
            {selectedFile && <img src={URL.createObjectURL(selectedFile)} alt="Selected" style={{ maxWidth: '100%', maxHeight: '100%' }} />}
          </div>
          <div style={{ width: '300px', backgroundColor: '#eee', padding: '20px' , marginLeft:"20px" }}>
            {result && <p>Result: {result}</p>}
          </div>
        </div>
      </div>
  );
}

export default App;
