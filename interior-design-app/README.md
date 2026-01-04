# Interior Design Studio

An interactive interior design application that allows users to upload room images or 360° videos and place virtual furniture objects in the space. Includes a neural network component that reconstructs a 2D plan from a single room photo with textures and approximate dimensions.

## Features

- Upload multiple images or a 360° video of a room
- Interactive 2D room viewer with drag-and-drop functionality
- Object library with various furniture items
- Ability to add, move, and remove objects in the room
- **New feature**: Neural network room reconstruction - converts a single room photo into a 2D plan with textures and approximate dimensions
- Support for partial room images
- Room type detection and confidence scoring
- Responsive design that works on different screen sizes

## Project Structure

```
src/
├── components/
│   ├── RoomUploader.jsx      # Component for uploading room images/videos
│   ├── RoomViewer.jsx        # Interactive room viewer with canvas
│   ├── ObjectLibrary.jsx     # Library of furniture objects
│   └── RoomReconstructor.jsx # Neural network room reconstruction component
├── utils/
│   └── neuralNetwork.js      # Neural network logic with TensorFlow.js integration
├── App.jsx                   # Main application component
├── main.jsx                  # Entry point
└── App.css                   # Global styles
public/
└── tfjs-loader.js            # TensorFlow.js dynamic loading script
```

## How It Works

1. **Upload**: Users can upload either multiple images of a room or a 360° video
2. **Process**: The neural network analyzes the image and creates a 2D reconstruction with textures
3. **Design**: Users can add furniture from the object library to the reconstructed room
4. **Interact**: Drag and drop objects to position them as desired
5. **Manage**: Remove objects or change their positions as needed

## Neural Network Room Reconstruction

The `RoomReconstructor` component uses a neural network to convert a single room photo into a 2D reconstruction. It analyzes the image and creates a textured 2D representation of the room with approximate dimensions and furniture placement based on the room type. The system is designed with TensorFlow.js integration points for real neural network deployment.

Currently, the implementation includes a simulation that demonstrates the expected behavior. In a production environment, this would connect to an actual trained neural network model.

For implementation details, see [NEURAL_NETWORK.md](./NEURAL_NETWORK.md) and [NEURAL_NETWORK_INTEGRATION.md](./NEURAL_NETWORK_INTEGRATION.md).

## Technical Implementation

This is an MVP (Minimum Viable Product) implementation with:

- React for the user interface
- HTML5 Canvas for the room visualization
- State management for tracking objects and room data
- Simulated neural network processing (in a real implementation, this would connect to actual AI services like TensorFlow.js)
- Component-based architecture for maintainability

## Future Enhancements

- Integration with real neural networks for realistic object insertion/removal
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

This MVP provides the UI framework for an interior design application. The neural network components for realistic object insertion and removal would be implemented in a production version by connecting to appropriate AI services or implementing the neural network processing on the backend. The current implementation simulates neural network functionality for demonstration purposes.
