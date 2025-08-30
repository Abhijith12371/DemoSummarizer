import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class VideoResNetAnalyzer:
    def __init__(self, model_type='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the ResNet video analyzer
        """
        self.device = device
        self.model = self._load_model(model_type)
        self.preprocess = self._get_preprocess()
        self.labels = self._get_imagenet_labels()
    
    def _get_imagenet_labels(self):
        """Get ImageNet class labels"""
        try:
            if not os.path.exists('imagenet_classes.txt'):
                import urllib.request
                url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                urllib.request.urlretrieve(url, "imagenet_classes.txt")
            
            with open('imagenet_classes.txt', 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        except:
            return [f"class_{i}" for i in range(1000)]
    
    def _load_model(self, model_type):
        """Load the pre-trained ResNet model"""
        weights = {
            'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
            'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
            'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
            'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,
            'resnet152': models.ResNet152_Weights.IMAGENET1K_V1
        }
        
        if model_type not in weights:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model = getattr(models, model_type)(weights=weights[model_type])
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_preprocess(self):
        """Get the image preprocessing pipeline"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def play_video_and_select_roi(self, video_path):
        """
        Play the video and allow user to pause and select ROI
        
        Args:
            video_path: Path to the video file
            
        Returns:
            roi: Selected region of interest (x, y, width, height)
            paused_frame: The frame where ROI was selected
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        paused = False
        roi_selected = False
        roi = None
        current_frame = None
        
        print("Video Controls:")
        print("Press SPACE to pause/resume")
        print("Press 's' to select ROI when paused")
        print("Press 'q' to quit")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                current_frame = frame.copy()
                
                # Display frame with instructions
                display_frame = frame.copy()
                cv2.putText(display_frame, "Press SPACE to pause", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'q' to quit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Select ROI - Play Video (SPACE=pause, s=select ROI, q=quit)', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(25) & 0xFF
            
            if key == ord(' '):  # Space bar to pause/resume
                paused = not paused
                if paused:
                    print("Video paused. Press 's' to select ROI")
                    cv2.putText(display_frame, "PAUSED - Press 's' to select ROI", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif paused and key == ord('s'):  # Select ROI when paused
                print("Select ROI with mouse, then press SPACE or ENTER to confirm")
                roi = cv2.selectROI("Select Region of Interest", current_frame)
                cv2.destroyWindow("Select Region of Interest")
                
                if roi != (0, 0, 0, 0):
                    roi_selected = True
                    paused_frame = current_frame.copy()
                    # Draw the selected ROI on the display frame
                    x, y, w, h = roi
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "ROI Selected! Press 'c' to continue", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"ROI selected: {roi}")
            
            elif key == ord('c') and roi_selected:  # Continue after ROI selection
                break
            
            elif key == ord('q'):  # Quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not roi_selected:
            raise ValueError("No ROI was selected")
        
        return roi, paused_frame
    
    def analyze_video_roi(self, video_path, roi, output_file=None, frame_interval=30, max_frames=50):
        """
        Analyze the selected ROI in the video
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Analyzing {min(max_frames, total_frames//frame_interval)} frames...")
        
        results = []
        frame_count = 0
        analyzed_count = 0
        
        while analyzed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                x, y, w, h = roi
                
                # Ensure ROI is within bounds
                if (x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]):
                    print(f"ROI out of bounds in frame {frame_count}")
                    frame_count += 1
                    continue
                
                roi_frame = frame[y:y+h, x:x+w]
                
                if roi_frame.size == 0:
                    frame_count += 1
                    continue
                
                try:
                    result = self.analyze_frame(roi_frame)
                    results.append({
                        'frame_idx': frame_count,
                        'timestamp': frame_count / fps,
                        'predictions': result
                    })
                    
                    top_pred = result[0]
                    print(f"Frame {frame_count}: {top_pred['label']} ({top_pred['confidence']:.3f})")
                    analyzed_count += 1
                    
                except Exception as e:
                    print(f"Error analyzing frame {frame_count}: {e}")
            
            frame_count += 1
            if frame_count >= total_frames:
                break
        
        cap.release()
        
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def analyze_frame(self, frame):
        """Analyze a single frame"""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        input_tensor = self.preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        predictions = []
        for i in range(5):
            predictions.append({
                'label': self.labels[top5_idx[i].item()],
                'confidence': float(top5_prob[i].item())
            })
        
        return predictions
    
    def _save_results(self, results, output_file):
        """Save results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"Frame {result['frame_idx']} (Time: {result['timestamp']:.2f}s):\n")
                for i, pred in enumerate(result['predictions']):
                    f.write(f"  {i+1}. {pred['label']}: {pred['confidence']:.4f}\n")
                f.write("\n")
    
    def show_analysis_summary(self, results):
        """Show a summary of analysis results"""
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        # Count occurrences of top predictions
        prediction_counts = {}
        for result in results:
            top_label = result['predictions'][0]['label']
            prediction_counts[top_label] = prediction_counts.get(top_label, 0) + 1
        
        # Display most common predictions
        print("\nMost common predictions:")
        sorted_predictions = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_predictions[:5]:
            percentage = (count / len(results)) * 100
            print(f"  {label}: {count} frames ({percentage:.1f}%)")
        
        print(f"\nTotal frames analyzed: {len(results)}")

def main():
    # Initialize analyzer
    analyzer = VideoResNetAnalyzer(model_type='resnet50')
    
    # Video path - CHANGE THIS TO YOUR VIDEO PATH
    video_path = "Video.mp4"  # Replace with your actual video path
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        print("Please provide a valid video file path.")
        return
    
    try:
        # Step 1: Play video and select ROI
        print("Playing video...")
        roi, paused_frame = analyzer.play_video_and_select_roi(video_path)
        
        # Step 2: Analyze the video with selected ROI
        print("\nStarting analysis...")
        results = analyzer.analyze_video_roi(
            video_path=video_path,
            roi=roi,
            output_file="video_analysis_results.txt",
            frame_interval=30,  # Analyze every 30th frame
            max_frames=50       # Analyze maximum 50 frames
        )
        
        # Step 3: Show summary
        analyzer.show_analysis_summary(results)
        
        print(f"\nAnalysis complete! Results saved to 'video_analysis_results.txt'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()