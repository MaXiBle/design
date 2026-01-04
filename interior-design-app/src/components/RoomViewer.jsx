import React, { useState, useRef, useEffect } from 'react';
import './RoomViewer.css';

const RoomViewer = ({ roomData, objects, setObjects, selectedObject, setSelectedObject }) => {
  const canvasRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [viewMode, setViewMode] = useState('2d'); // '2d' or '3d'

  // Initialize canvas when room data changes
  useEffect(() => {
    if (roomData && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw room background
      drawRoomBackground(ctx, canvas.width, canvas.height);
      
      // Draw objects
      objects.forEach(obj => {
        drawObject(ctx, obj);
      });
    }
  }, [roomData, objects]);

  const drawRoomBackground = (ctx, width, height) => {
    // Draw a simple room representation
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, width, height);
    
    // Draw walls
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 2;
    ctx.strokeRect(50, 50, width - 100, height - 100);
    
    // Draw floor pattern
    ctx.fillStyle = '#e0e0e0';
    for (let x = 0; x < width; x += 40) {
      for (let y = 0; y < height; y += 40) {
        if ((x + y) % 80 === 0) {
          ctx.fillRect(x, y, 20, 20);
        }
      }
    }
  };

  const drawObject = (ctx, object) => {
    ctx.save();
    
    // Draw object based on its type
    switch (object.type) {
      case 'chair':
        drawChair(ctx, object.x, object.y, object.selected);
        break;
      case 'table':
        drawTable(ctx, object.x, object.y, object.selected);
        break;
      case 'sofa':
        drawSofa(ctx, object.x, object.y, object.selected);
        break;
      case 'lamp':
        drawLamp(ctx, object.x, object.y, object.selected);
        break;
      default:
        drawGenericObject(ctx, object.x, object.y, object.selected);
    }
    
    ctx.restore();
  };

  const drawChair = (ctx, x, y, isSelected) => {
    ctx.fillStyle = isSelected ? '#ff6b6b' : '#4ecdc4';
    ctx.fillRect(x - 15, y - 15, 30, 30);
    
    // Draw chair back
    ctx.fillRect(x - 15, y - 30, 30, 15);
    
    if (isSelected) {
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 18, y - 18, 36, 36);
    }
  };

  const drawTable = (ctx, x, y, isSelected) => {
    ctx.fillStyle = isSelected ? '#ff6b6b' : '#45b7d1';
    ctx.fillRect(x - 40, y - 20, 80, 40);
    
    if (isSelected) {
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 43, y - 23, 86, 46);
    }
  };

  const drawSofa = (ctx, x, y, isSelected) => {
    ctx.fillStyle = isSelected ? '#ff6b6b' : '#96ceb4';
    ctx.fillRect(x - 50, y - 25, 100, 50);
    
    if (isSelected) {
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 53, y - 28, 106, 56);
    }
  };

  const drawLamp = (ctx, x, y, isSelected) => {
    // Lamp base
    ctx.fillStyle = isSelected ? '#ff6b6b' : '#feca57';
    ctx.fillRect(x - 8, y - 30, 16, 30);
    
    // Lamp shade
    ctx.beginPath();
    ctx.arc(x, y - 35, 15, 0, Math.PI * 2);
    ctx.fill();
    
    if (isSelected) {
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 11, y - 33, 22, 36);
    }
  };

  const drawGenericObject = (ctx, x, y, isSelected) => {
    ctx.fillStyle = isSelected ? '#ff6b6b' : '#9d9d9d';
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, Math.PI * 2);
    ctx.fill();
    
    if (isSelected) {
      ctx.strokeStyle = '#ff0000';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  };

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if clicked on an object
    const clickedObject = objects.find(obj => {
      const distance = Math.sqrt(Math.pow(obj.x - x, 2) + Math.pow(obj.y - y, 2));
      return distance < 30; // Adjust threshold as needed
    });
    
    if (clickedObject) {
      // If object is already selected, deselect it
      if (selectedObject && selectedObject.id === clickedObject.id) {
        setSelectedObject(null);
      } else {
        setSelectedObject(clickedObject);
      }
    } else {
      // Deselect if clicked on empty space
      setSelectedObject(null);
    }
  };

  const handleMouseDown = (e) => {
    if (!selectedObject) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if clicked on the selected object
    const distance = Math.sqrt(Math.pow(selectedObject.x - x, 2) + Math.pow(selectedObject.y - y, 2));
    
    if (distance < 30) {
      setIsDragging(true);
      setDragOffset({
        x: x - selectedObject.x,
        y: y - selectedObject.y
      });
    }
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !selectedObject) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const newX = x - dragOffset.x;
    const newY = y - dragOffset.y;
    
    // Update object position
    setObjects(prev => prev.map(obj => 
      obj.id === selectedObject.id 
        ? { ...obj, x: newX, y: newY }
        : obj
    ));
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const addObject = (type) => {
    const newObject = {
      id: Date.now(),
      type,
      x: 200 + Math.random() * 200, // Random position
      y: 200 + Math.random() * 200,
      selected: false
    };
    
    setObjects(prev => [...prev, newObject]);
    setSelectedObject(newObject);
  };

  const removeObject = () => {
    if (!selectedObject) return;
    
    setObjects(prev => prev.filter(obj => obj.id !== selectedObject.id));
    setSelectedObject(null);
  };

  return (
    <div className="room-viewer">
      <div className="viewer-header">
        <h3>Room Preview</h3>
        <div className="view-controls">
          <button 
            className={`view-mode-btn ${viewMode === '2d' ? 'active' : ''}`}
            onClick={() => setViewMode('2d')}
          >
            2D View
          </button>
          <button 
            className={`view-mode-btn ${viewMode === '3d' ? 'active' : ''}`}
            onClick={() => setViewMode('3d')}
            disabled
          >
            3D View (Coming Soon)
          </button>
        </div>
      </div>
      
      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={800}
          height={500}
          onClick={handleCanvasClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="room-canvas"
        />
      </div>
      
      <div className="viewer-controls">
        <button 
          className="control-btn remove-btn" 
          onClick={removeObject}
          disabled={!selectedObject}
        >
          Remove Selected Object
        </button>
        
        <div className="add-objects">
          <button className="control-btn" onClick={() => addObject('chair')}>Add Chair</button>
          <button className="control-btn" onClick={() => addObject('table')}>Add Table</button>
          <button className="control-btn" onClick={() => addObject('sofa')}>Add Sofa</button>
          <button className="control-btn" onClick={() => addObject('lamp')}>Add Lamp</button>
        </div>
      </div>
    </div>
  );
};

export default RoomViewer;