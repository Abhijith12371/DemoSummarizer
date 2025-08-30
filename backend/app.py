import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_video, write_jpeg
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class VideoResNetAnalyzer:
    def __init__(self, model_type='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the ResNet video analyzer
        
        Args:
            model_type: Type of ResNet model ('resnet18', 'resnet50', etc.)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = self._load_model(model_type)
        self.preprocess = self._get_preprocess()
        
        # Load ImageNet class labels
        with open('imagenet_classes.txt') as f:
            self.labels = [line.strip() for line in f.readlines()]
    
    def _load_model(self, model_type):
        """Load the pre-trained ResNet model"""
        if model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_type == 'resnet34':
            model = models.resnet34(pretrained=True)
        elif model_type == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model_type == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif model_type == 'resnet152':
            model = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        return model
    
    def _get_preprocess(self):
        """Get the image preprocessing pipeline"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def select_roi(self, video_path, frame_idx=0):
        """
        Select a region of interest from a video frame
        
        Args:
            video_path: Path to the video file
            frame_idx: Index of the frame to use for ROI selection
            
        Returns:
            roi: Selected region of interest (x, y, width, height)
        """
        # Read the video
        video_frames, _, _ = read_video(video_path, output_format='TCHW')
        frame = video_frames[frame_idx].permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Display the frame and let user select ROI
        roi = cv2.selectROI("Select Region of Interest", frame)
        cv2.destroyAllWindows()
        
        return roi
    
    def analyze_video_roi(self, video_path, roi, output_file=None, frame_interval=10):
        """
        Analyze a specific region of interest in a video
        
        Args:
            video_path: Path to the video file
            roi: Region of interest (x, y, width, height)
            output_file: Optional path to save analysis results
            frame_interval: Analyze every nth frame
            
        Returns:
            results: List of analysis results for each frame
        """
        # Read the video
        video_frames, _, _ = read_video(video_path, output_format='TCHW')
        results = []
        
        for i in range(0, len(video_frames), frame_interval):
            frame = video_frames[i].permute(1, 2, 0).numpy().astype(np.uint8)
            
            # Extract ROI from frame
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]
            
            # Analyze the ROI
            result = self.analyze_frame(roi_frame)
            results.append({
                'frame_idx': i,
                'predictions': result
            })
            
            print(f"Frame {i}: Top prediction - {result[0]['label']} ({result[0]['confidence']:.2f})")
        
        # Save results if output file is specified
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def analyze_frame(self, frame):
        """
        Analyze a single frame using ResNet
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            predictions: List of top predictions with labels and confidence
        """
        # Convert frame to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Preprocess the image
        input_tensor = self.preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        top5_prob = top5_prob.cpu().numpy()
        top5_idx = top5_idx.cpu().numpy()
        
        # Format results
        predictions = []
        for i in range(5):
            predictions.append({
                'label': self.labels[top5_idx[i]],
                'confidence': top5_prob[i]
            })
        
        return predictions
    
    def _save_results(self, results, output_file):
        """Save analysis results to a file"""
        with open(output_file, 'w') as f:
            for result in results:
                f.write(f"Frame {result['frame_idx']}:\n")
                for pred in result['predictions']:
                    f.write(f"  {pred['label']}: {pred['confidence']:.4f}\n")
                f.write("\n")
    
    def visualize_analysis(self, video_path, roi, results, frame_indices=None):
        """
        Visualize the analysis results on the video frames
        
        Args:
            video_path: Path to the video file
            roi: Region of interest
            results: Analysis results
            frame_indices: Specific frames to visualize (default: all analyzed frames)
        """
        if frame_indices is None:
            frame_indices = [r['frame_idx'] for r in results]
        
        # Read the video
        video_frames, _, _ = read_video(video_path, output_format='TCHW')
        
        for result in results:
            if result['frame_idx'] not in frame_indices:
                continue
                
            frame = video_frames[result['frame_idx']].permute(1, 2, 0).numpy().astype(np.uint8)
            
            # Draw ROI rectangle
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add prediction text
            top_pred = result['predictions'][0]
            text = f"{top_pred['label']}: {top_pred['confidence']:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            plt.figure(figsize=(10, 8))
            plt.imshow(frame)
            plt.title(f"Frame {result['frame_idx']}")
            plt.axis('off')
            plt.show()

def main():
    # Initialize the analyzer
    analyzer = VideoResNetAnalyzer(model_type='resnet50')
    
    # Path to your video file
    video_path = "your_video.mp4"  # Replace with your video path
    
    try:
        # Select a region of interest from the first frame
        print("Select a region of interest in the video frame...")
        roi = analyzer.select_roi(video_path, frame_idx=0)
        print(f"Selected ROI: {roi}")
        
        # Analyze the video with the selected ROI
        print("Analyzing video...")
        results = analyzer.analyze_video_roi(
            video_path, 
            roi, 
            output_file="analysis_results.txt",
            frame_interval=30  # Analyze every 30th frame
        )
        
        # Visualize the results for the first few analyzed frames
        print("Visualizing results...")
        analyzer.visualize_analysis(
            video_path, 
            roi, 
            results, 
            frame_indices=[r['frame_idx'] for r in results[:3]]  # Show first 3 analyzed frames
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()