import os
import json
import torch
import numpy as np

def verify_data_consistency(sample_limit=10):
    json_root = "data/labeled_json/"
    tensor_root = "data/flattened_tensors/"
    exercises = ["squat", "push_up", "bench_press", "pull_up"]
    
    print("🔍 --- Starting Pipeline Consistency Verification ---")
    
    # 1. Count Files
    json_count = 0
    tensor_count = 0
    
    for ex in exercises:
        j_path = os.path.join(json_root, ex)
        t_path = os.path.join(tensor_root, ex)
        
        if os.path.exists(j_path):
            json_count += len([f for f in os.listdir(j_path) if f.endswith('.json')])
        if os.path.exists(t_path):
            tensor_count += len([f for f in os.listdir(t_path) if f.endswith('.pt')])
            
    print(f"📊 File Count Check: JSON: {json_count} | Tensors: {tensor_count}")

    if json_count == 0:
        print("❌ Error: No JSON files found. Check your json_root path.")
        return

    print(f"\n🧪 Checking up to {sample_limit} Samples for Numerical Consistency...")
    
    samples_checked = 0
    
    for ex in exercises:
        if samples_checked >= sample_limit:
            break
        
        j_dir = os.path.join(json_root, ex)
        if not os.path.exists(j_dir):
            continue
            
        filenames = [f for f in os.listdir(j_dir) if f.endswith('.json')]
        
        for fname in filenames:
            if samples_checked >= sample_limit:
                break
            
            json_path = os.path.join(json_root, ex, fname)
            tensor_path = os.path.join(tensor_root, ex, fname.replace('.json', '.pt'))
            
            if not os.path.exists(tensor_path):
                continue
                
            try:
                # --- Load JSON ---
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    raw_frames = data
                elif isinstance(data, dict):
                    raw_frames = data.get('frames', [])
                else:
                    raw_frames = []

                raw_frames = sorted(raw_frames, key=lambda x: x.get('frame_index', 0))
                
                json_frames_list = []
                expected_landmarks = 33
                
                for frame in raw_frames:
                    landmarks = frame.get('landmarks', [])
                    if landmarks is None:
                        landmarks = []
                        
                    flat_frame = []
                    
                    for i in range(expected_landmarks):
                        lm = landmarks[i] if i < len(landmarks) else None
                        
                        if lm is None:
                            flat_frame.extend([0.0, 0.0, 0.0, 0.0])
                        else:
                            for j in range(4):
                                val = lm[j] if j < len(lm) else 0.0
                                flat_frame.append(0.0 if val is None else val)
                    
                    if flat_frame:
                        json_frames_list.append(flat_frame)
                
                json_np = np.array(json_frames_list, dtype=np.float32)
                
                # --- FIXED PART ---
                loaded = torch.load(tensor_path)

                if not isinstance(loaded, dict):
                    print(f"❌ Unexpected format in {fname}")
                    continue

                tensor_np = loaded['sequence'].numpy()
                rep_count = loaded.get('rep_count', None)

                # --- Comparison ---
                if json_np.shape != tensor_np.shape:
                    print(f"❌ Shape Mismatch [{fname}]")
                    print(f"   JSON: {json_np.shape} | Tensor: {tensor_np.shape}")
                else:
                    if np.allclose(json_np, tensor_np, atol=1e-4):
                        print(f"✅ Match: {fname} (Shape: {json_np.shape}, reps={rep_count})")
                    else:
                        diff = np.abs(json_np - tensor_np).max()
                        print(f"⚠️ Value Drift [{fname}]: Max Diff: {diff}")
                
                samples_checked += 1
                        
            except Exception as e:
                print(f"❌ Error processing {fname}: {str(e)}")
                break

    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    verify_data_consistency(sample_limit=10)