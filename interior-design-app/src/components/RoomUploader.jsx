import React, { useState } from 'react';
import './RoomUploader.css';

const RoomUploader = ({ onRoomUpload }) => {
  const [uploadType, setUploadType] = useState('image'); // 'image' or 'video'
  const [files, setFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
  };

  const handleUpload = async () => {
    if (files.length === 0) return;

    setIsProcessing(true);
    
    // Simulate processing time for the MVP
    setTimeout(() => {
      const roomData = {
        type: uploadType,
        files: files.map(file => URL.createObjectURL(file)),
        id: Date.now(),
        processed: true
      };
      
      onRoomUpload(roomData);
      setIsProcessing(false);
    }, 2000);
  };

  return (
    <div className="room-uploader">
      <h2>Upload Room Data</h2>
      
      <div className="upload-options">
        <label>
          <input
            type="radio"
            value="image"
            checked={uploadType === 'image'}
            onChange={(e) => setUploadType(e.target.value)}
          />
          Multiple Images
        </label>
        
        <label>
          <input
            type="radio"
            value="video"
            checked={uploadType === 'video'}
            onChange={(e) => setUploadType(e.target.value)}
          />
          360° Video
        </label>
      </div>

      <div className="file-upload-area">
        <input
          type="file"
          id="file-upload"
          multiple
          accept={uploadType === 'image' ? 'image/*' : 'video/*'}
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <label htmlFor="file-upload" className="upload-label">
          <div className="upload-icon">📁</div>
          <p>Click to upload {uploadType === 'image' ? 'images' : 'video'}</p>
          <p className="upload-hint">Supports JPG, PNG, MP4 formats</p>
        </label>
      </div>

      {files.length > 0 && (
        <div className="file-preview">
          <h3>Selected Files:</h3>
          <div className="preview-grid">
            {files.slice(0, 4).map((file, index) => (
              <div key={index} className="preview-item">
                {uploadType === 'image' ? (
                  <img src={URL.createObjectURL(file)} alt={`Preview ${index}`} />
                ) : (
                  <video controls>
                    <source src={URL.createObjectURL(file)} type="video/mp4" />
                  </video>
                )}
                <p>{file.name}</p>
              </div>
            ))}
            {files.length > 4 && (
              <div className="preview-item more">+{files.length - 4} more</div>
            )}
          </div>
        </div>
      )}

      <button 
        className="upload-button" 
        onClick={handleUpload} 
        disabled={files.length === 0 || isProcessing}
      >
        {isProcessing ? 'Processing Room...' : 'Process Room Data'}
      </button>
    </div>
  );
};

export default RoomUploader;