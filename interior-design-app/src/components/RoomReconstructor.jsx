import React, { useState, useEffect } from 'react';
import { neuralNetwork } from '../utils/neuralNetwork';
import './RoomReconstructor.css';

const RoomReconstructor = () => {
  const [image, setImage] = useState(null);
  const [reconstructedRoom, setReconstructedRoom] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Neural network model for room reconstruction
  const reconstructRoom = async (imageFile) => {
    // In a real implementation, we would pass the image data to the neural network
    // For this simulation, we'll use our neural network module
    return neuralNetwork.reconstructRoom(imageFile);
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setImage(URL.createObjectURL(file));
    setLoading(true);
    setError(null);

    try {
      const reconstructed = await reconstructRoom(file);
      setReconstructedRoom(reconstructed);
    } catch (err) {
      setError('Ошибка при реконструкции комнаты: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderReconstructedRoom = () => {
    if (!reconstructedRoom) return null;

    return (
      <div className="reconstructed-room-container">
        <h3>Реконструированная 2D комната</h3>
        <div className="room-info">
          <p><strong>Тип комнаты:</strong> {reconstructedRoom.roomType.replace('_', ' ')}</p>
          <p><strong>Доверие модели:</strong> {(reconstructedRoom.confidence * 100).toFixed(1)}%</p>
        </div>
        <div 
          className="reconstructed-room"
          style={{ 
            width: reconstructedRoom.dimensions.width, 
            height: reconstructedRoom.dimensions.height,
            position: 'relative',
            border: '2px solid #ccc',
            overflow: 'hidden'
          }}
        >
          {/* Render textures */}
          {reconstructedRoom.textures.map(texture => (
            <div
              key={texture.id}
              className={`texture ${texture.type}`}
              style={{
                position: 'absolute',
                left: texture.position.x,
                top: texture.position.y,
                width: texture.position.width,
                height: texture.position.height,
                backgroundColor: texture.color,
                border: texture.type === 'wall' ? 'none' : '1px solid #333'
              }}
            />
          ))}
          
          {/* Render objects */}
          {reconstructedRoom.objects.map(obj => (
            <div
              key={obj.id}
              className={`object ${obj.type}`}
              style={{
                position: 'absolute',
                left: obj.position.x,
                top: obj.position.y,
                width: obj.position.width,
                height: obj.position.height,
                backgroundColor: obj.type === 'sofa' ? '#8B4513' : 
                                obj.type === 'bed' ? '#CD5C5C' :
                                obj.type === 'nightstand' ? '#A0522D' :
                                obj.type === 'coffee_table' ? '#D2B48C' :
                                obj.type === 'tv' ? '#333333' :
                                obj.type === 'counter' ? '#DDC488' :
                                obj.type === 'refrigerator' ? '#C0C0C0' :
                                obj.type === 'toilet' ? '#FFFFFF' :
                                obj.type === 'sink' ? '#ADD8E6' :
                                obj.type === 'bathtub' ? '#87CEEB' : '#A0522D',
                border: '1px solid #333'
              }}
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="room-reconstructor">
      <h2>Нейросеть реконструкции комнаты</h2>
      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          disabled={loading}
        />
        <p>Загрузите фото комнаты для 2D реконструкции</p>
      </div>
      
      {loading && <div className="loading">Обработка изображения нейросетью...</div>}
      {error && <div className="error">{error}</div>}
      {image && (
        <div className="original-image">
          <h3>Оригинальное изображение:</h3>
          <img src={image} alt="Оригинальная комната" style={{ maxWidth: '400px', maxHeight: '300px' }} />
        </div>
      )}
      {reconstructedRoom && renderReconstructedRoom()}
    </div>
  );
};

export default RoomReconstructor;