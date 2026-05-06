import os
import torch
import litert_torch
from ai_edge_quantizer import quantizer, recipe

# Import your model
from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask

def convert_to_tflite_int8(checkpoint_path, output_path):
    print("1. Initializing PyTorch ST-GCN Model...")
    model = CoreSetSTGCN_MultiTask(in_channels=14, num_classes=4, max_frames=150, node_count=33)
    
    # Load PyTorch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input based on ST-GCN shape
    sample_input = torch.randn(1, 14, 150, 33, 1)

    print("2. Converting to standard Float LiteRT Model...")
    edge_model = litert_torch.convert(model, (sample_input,))
    
    # Save a temporary float32 version of the model
    temp_path = "checkpoint/temp_float.tflite"
    edge_model.export(temp_path)

    print("3. Compressing FlatBuffer to INT8 via AI Edge Quantizer...")
    # This compresses all the model weights to 8-bit integers 
    qt = quantizer.Quantizer(temp_path)
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    qt.quantize().export_model(output_path)
    
    # Clean up the temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    print(f"\nSuccess! INT8 TFLite model saved to {output_path}")

if __name__ == "__main__":
    chkpt = "checkpoint/best_stgcn_model.pth" 
    out_file = "checkpoint/stgcn_int8.tflite"
    
    convert_to_tflite_int8(chkpt, out_file)