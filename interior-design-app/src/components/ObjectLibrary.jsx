import React, { useState } from 'react';
import './ObjectLibrary.css';

const ObjectLibrary = ({ objects, setObjects, selectedObject, setSelectedObject }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  
  // Predefined objects for the library
  const objectLibrary = [
    { id: 'chair1', name: 'Modern Chair', type: 'chair', category: 'seating', price: 199.99 },
    { id: 'chair2', name: 'Armchair', type: 'chair', category: 'seating', price: 299.99 },
    { id: 'table1', name: 'Coffee Table', type: 'table', category: 'tables', price: 149.99 },
    { id: 'table2', name: 'Dining Table', type: 'table', category: 'tables', price: 499.99 },
    { id: 'sofa1', name: '2-Seater Sofa', type: 'sofa', category: 'seating', price: 799.99 },
    { id: 'sofa2', name: 'L-Shaped Sofa', type: 'sofa', category: 'seating', price: 1299.99 },
    { id: 'lamp1', name: 'Floor Lamp', type: 'lamp', category: 'lighting', price: 89.99 },
    { id: 'lamp2', name: 'Table Lamp', type: 'lamp', category: 'lighting', price: 49.99 },
    { id: 'bed1', name: 'Queen Bed', type: 'bed', category: 'bedroom', price: 599.99 },
    { id: 'desk1', name: 'Office Desk', type: 'desk', category: 'office', price: 249.99 },
    { id: 'shelf1', name: 'Bookshelf', type: 'shelf', category: 'storage', price: 199.99 },
    { id: 'plant1', name: 'Indoor Plant', type: 'plant', category: 'decor', price: 39.99 },
  ];

  // Filter objects based on search and category
  const filteredLibrary = objectLibrary.filter(item => {
    const matchesSearch = item.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = categoryFilter === 'all' || item.category === categoryFilter;
    return matchesSearch && matchesCategory;
  });

  const categories = ['all', 'seating', 'tables', 'lighting', 'bedroom', 'office', 'storage', 'decor'];

  const addObjectToRoom = (object) => {
    const newObject = {
      id: `${object.id}-${Date.now()}`,
      type: object.type,
      name: object.name,
      x: 150 + Math.random() * 200, // Random position in room
      y: 150 + Math.random() * 200,
      price: object.price
    };
    
    setObjects(prev => [...prev, newObject]);
    setSelectedObject(newObject);
  };

  const removeObjectFromRoom = (objectId) => {
    setObjects(prev => prev.filter(obj => obj.id !== objectId));
    if (selectedObject && selectedObject.id === objectId) {
      setSelectedObject(null);
    }
  };

  return (
    <div className="object-library">
      <h3>Object Library</h3>
      
      <div className="library-controls">
        <input
          type="text"
          placeholder="Search objects..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        
        <select 
          value={categoryFilter} 
          onChange={(e) => setCategoryFilter(e.target.value)}
          className="category-filter"
        >
          {categories.map(category => (
            <option key={category} value={category}>
              {category.charAt(0).toUpperCase() + category.slice(1)}
            </option>
          ))}
        </select>
      </div>
      
      <div className="library-grid">
        {filteredLibrary.map(item => (
          <div key={item.id} className="library-item">
            <div className="item-icon">
              {item.type === 'chair' && '🪑'}
              {item.type === 'table' && '🪑'}
              {item.type === 'sofa' && '🛋️'}
              {item.type === 'lamp' && '💡'}
              {item.type === 'bed' && '🛏️'}
              {item.type === 'desk' && '🪑'}
              {item.type === 'shelf' && '📚'}
              {item.type === 'plant' && '🌿'}
            </div>
            <div className="item-info">
              <h4>{item.name}</h4>
              <p className="item-category">{item.category}</p>
              <p className="item-price">${item.price.toFixed(2)}</p>
            </div>
            <button 
              className="add-to-room-btn"
              onClick={() => addObjectToRoom(item)}
            >
              Add to Room
            </button>
          </div>
        ))}
      </div>
      
      {objects.length > 0 && (
        <div className="room-objects-section">
          <h4>Objects in Room ({objects.length})</h4>
          <div className="room-objects-list">
            {objects.map(obj => (
              <div key={obj.id} className={`room-object ${selectedObject && selectedObject.id === obj.id ? 'selected' : ''}`}>
                <span className="object-name">{obj.name || `${obj.type.charAt(0).toUpperCase() + obj.type.slice(1)}`}</span>
                <div className="object-actions">
                  <button 
                    className="remove-object-btn"
                    onClick={() => removeObjectFromRoom(obj.id)}
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ObjectLibrary;