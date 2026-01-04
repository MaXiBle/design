import './App.css';
import RoomUploader from './components/RoomUploader';
import RoomViewer from './components/RoomViewer';
import ObjectLibrary from './components/ObjectLibrary';
import RoomReconstructor from './components/RoomReconstructor';
import { useState } from 'react';

function App() {
  const [roomData, setRoomData] = useState(null);
  const [objects, setObjects] = useState([]);
  const [selectedObject, setSelectedObject] = useState(null);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Interior Design Studio</h1>
      </header>
      
      <main className="app-main">
        <div className="upload-section">
          <RoomUploader onRoomUpload={setRoomData} />
          <RoomReconstructor />
        </div>
        
        {roomData && (
          <div className="design-section">
            <div className="room-viewer-container">
              <RoomViewer roomData={roomData} objects={objects} setObjects={setObjects} selectedObject={selectedObject} setSelectedObject={setSelectedObject} />
            </div>
            
            <div className="object-library-container">
              <ObjectLibrary 
                objects={objects} 
                setObjects={setObjects} 
                selectedObject={selectedObject} 
                setSelectedObject={setSelectedObject} 
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
