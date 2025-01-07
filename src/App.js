import React, { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";
import './style.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
          faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
          faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        ]);
      } catch (error) {
        setErrorMessage("Error loading face recognition models. Please check your /models directory.");
        console.error(error);
      }
    };

    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        setErrorMessage("Error accessing webcam. Please check your device permissions.");
        console.error(error);
      }
    };

    const setupFaceDetection = async () => {
      const labels = ["Virat", "Messi", "Prakash"];
      try {
        const labeledFaceDescriptors = await Promise.all(
          labels.map(async (label) => {
            const descriptions = [];
            for (let i = 1; i <= 2; i++) {
              const img = await faceapi.fetchImage(`./labels/${label}/${i}.png`);
              const detections = await faceapi
                .detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor();
              descriptions.push(detections.descriptor);
            }
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
          })
        );

        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);
        const displaySize = { width: videoRef.current.width, height: videoRef.current.height };
        faceapi.matchDimensions(canvasRef.current, displaySize);

        setInterval(async () => {
          const detections = await faceapi
            .detectAllFaces(videoRef.current)
            .withFaceLandmarks()
            .withFaceDescriptors();

          const resizedDetections = faceapi.resizeResults(detections, displaySize);

          const context = canvasRef.current.getContext("2d");
          context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

          const results = resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor));
          results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box;
            new faceapi.draw.DrawBox(box, { label: result.toString() }).draw(canvasRef.current);
          });
        }, 100);
      } catch (error) {
        setErrorMessage("Error setting up face detection.");
        console.error(error);
      }
    };

    loadModels().then(() => {
      startWebcam().then(() => {
        setupFaceDetection();
      });
    });
  }, []);

  return (
    <div className="App">
      <div className="navbar">
        <h2>Face Recognition App</h2>
        <a href="#instructions">How to Use</a>
      </div>
      <h1>Real-Time Face Recognition</h1>
      <p>This application uses advanced AI to recognize and identify faces in real-time.</p>
      {errorMessage && <p style={{ color: "red" }}>{errorMessage}</p>}
      <div className="video-container">
        <video ref={videoRef} width="600" height="450" autoPlay></video>
        <canvas ref={canvasRef} className="overlay" width="600" height="450"></canvas>
      </div>
      <div id="instructions" className="instructions">
        <h2>How to Use</h2>
        <ol>
          <li>Ensure you have placed the face detection models in the <code>/models</code> directory.</li>
          <li>Provide labeled images for recognition in the <code>/labels</code> folder.</li>
          <li>Allow access to your webcam when prompted by the browser.</li>
          <li>Watch as the app detects and identifies faces in real-time!</li>
        </ol>
      </div>
      <div className="footer">
        <p>© 2025 Face Recognition App | Developed with ❤️ by CHIRANJEEVULU CHENNEBOYINA </p>
        <p>
          For more information, visit our <a href="https://github.com/Chiru2123/FaceRecog-master" target="_blank" style={{ color: 'white', textDecoration: 'underline' }}>documentation</a>.
        </p>
      </div>
    </div>
  );
}

export default App;
