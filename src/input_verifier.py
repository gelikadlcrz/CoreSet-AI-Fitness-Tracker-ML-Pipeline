import os
import json
import torch
import numpy as np

def verify_data_consistency(sample_limit=10):
    # BASE_DIR ensures we look relative to where the command is run
    # Assuming your folders are INSIDE the project root
    json_root = "data/final_data/" 
    tensor_root = "data/flattened_tensors/"
    exercises = ["squat", "push_up", "bench_press", "pull_up"]
    
    print("🔍 --- Starting Pipeline Consistency Verification ---")
    print(f"Checking JSON path: {os.path.abspath(json_root)}")
    print(f"Checking Tensor path: {os.path.abspath(tensor_root)}")
    
    # 1. Count Files
    json_count = 0
    tensor_count = 0
    
    for ex in exercises:
        j_path = os.path.join(json_root, ex)
        t_path = os.path.join(tensor_root, ex)
        
        if os.path.exists(j_path):
            files = [f for f in os.listdir(j_path) if f.endswith('.json')]
            json_count += len(files)
        else:
            print(f"⚠️  Note: Exercise folder not found: {j_path}")
            
        if os.path.exists(t_path):
            tensor_count += len([f for f in os.listdir(t_path) if f.endswith('.pt')])
            
    print(f"📊 File Count Check: JSON: {json_count} | Tensors: {tensor_count}")

    if json_count == 0:
        print(f" Error: No JSON files found. Ensure your data is in: {json_root}/<exercise_name>/")
        return

    print(f"\nChecking up to {sample_limit} Samples for Numerical Consistency...")
    
    samples_checked = 0
    for ex in exercises:
        if samples_checked >= sample_limit: break
        
        j_dir = os.path.join(json_root, ex)
        if not os.path.exists(j_dir): continue
            
        filenames = [f for f in os.listdir(j_dir) if f.endswith('.json')]
        
        for fname in filenames:
            if samples_checked >= sample_limit: break
            
            json_path = os.path.join(json_root, ex, fname)
            tensor_path = os.path.join(tensor_root, ex, fname.replace('.json', '.pt'))
            
            if not os.path.exists(tensor_path):
                continue
                
            try:
                # --- Load JSON (final_data) ---
                with open(json_path, 'r') as f:
                    final_data = json.load(f)
                
                # Format detection (adhering to new pose_sequence format)
                if isinstance(final_data, dict) and 'pose_sequence' in final_data:
                    raw_frames = final_data['pose_sequence']
                    is_direct_sequence = True
                else:
                    # Fallback to old format if necessary
                    raw_frames = final_data.get('frames', [])
                    raw_frames = sorted(raw_frames, key=lambda x: x.get('frame_index', 0))
                    is_direct_sequence = False

                json_frames_list = []
                for frame in raw_frames:
                    landmarks = frame if is_direct_sequence else frame.get('landmarks', [])
                    flat_frame = []
                    for i in range(33): # Expected landmarks
                        lm = landmarks[i] if i < len(landmarks) else [0,0,0,0]
                        # Flatten to 4 features per landmark
                        for j in range(4):
                            val = lm[j] if j < len(lm) else 0.0
                            flat_frame.append(float(val or 0.0))
                    json_frames_list.append(flat_frame)
                
                json_np = np.array(json_frames_list, dtype=np.float32)
                
                # --- Load Processed Tensor ---
                loaded = torch.load(tensor_path)
                tensor_np = loaded['sequence'].numpy()
                rep_count = loaded.get('rep_count', 0)

                # --- Comparison ---
                if json_np.shape != tensor_np.shape:
                    print(f" Shape Mismatch [{fname}] JSON: {json_np.shape} vs Tensor: {tensor_np.shape}")
                elif np.allclose(json_np, tensor_np, atol=1e-4):
                    print(f"✅ Match: {fname} (Shape: {json_np.shape}, reps={rep_count})")
                else:
                    print(f" Value Drift [{fname}]: Max Diff: {np.abs(json_np - tensor_np).max()}")
                
                samples_checked += 1
                        
            except Exception as e:
                print(f" Error processing {fname}: {str(e)}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_data_consistency(sample_limit=10)