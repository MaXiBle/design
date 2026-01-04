# Interior Design Studio

An interactive interior design application that allows users to upload room images or 360° videos and place virtual furniture objects in the space.

## Features

- Upload multiple images or a 360° video of a room
- Interactive 2D room viewer with drag-and-drop functionality
- Object library with various furniture items
- Ability to add, move, and remove objects in the room
- Responsive design that works on different screen sizes

## Project Structure

```
src/
├── components/
│   ├── RoomUploader.jsx      # Component for uploading room images/videos
│   ├── RoomViewer.jsx        # Interactive room viewer with canvas
│   └── ObjectLibrary.jsx     # Library of furniture objects
├── App.jsx                   # Main application component
├── main.jsx                  # Entry point
└── App.css                   # Global styles
```

## How It Works

1. **Upload**: Users can upload either multiple images of a room or a 360° video
2. **Process**: The application simulates processing the room data to create a 2D representation
3. **Design**: Users can add furniture from the object library to the room
4. **Interact**: Drag and drop objects to position them as desired
5. **Manage**: Remove objects or change their positions as needed

## Technical Implementation

This is an MVP (Minimum Viable Product) implementation with:

- React for the user interface
- HTML5 Canvas for the room visualization
- State management for tracking objects and room data
- Simulated neural network processing (in a real implementation, this would connect to actual AI services)

## Future Enhancements

- Integration with neural networks for realistic object insertion/removal
- 3D room visualization
- AR/VR support
- Real-time collaboration
- Advanced material and lighting controls
- Export options for design plans

## Getting Started

1. Install dependencies: `npm install`
2. Start the development server: `npm run dev`
3. Open your browser to the provided URL

## Development Notes

This MVP provides the UI framework for an interior design application. The neural network components for realistic object insertion and removal would be implemented in a production version by connecting to appropriate AI services or implementing the neural network processing on the backend.
