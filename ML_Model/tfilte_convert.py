"""
Convert TFLite model to C array for embedding in STM32
"""
import os

def convert_tflite_to_c_array(tflite_path='activity_model_quantized.tflite',
                               output_h='model_data.h',
                               output_c='model_data.c'):
    """
    Convert TFLite model to C byte array
    """
    print(f"Converting {tflite_path} to C array...")
    
    # Lire le modèle TFLite
    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()
    
    model_size = len(tflite_model)
    print(f"Model size: {model_size} bytes ({model_size/1024:.2f} KB)")
    
    # Générer le fichier .h
    with open(output_h, 'w') as f:
        f.write("/**\n")
        f.write(" * TensorFlow Lite Model Data\n")
        f.write(" * Auto-generated from TFLite model\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define MODEL_SIZE {model_size}\n\n")
        f.write("extern const unsigned char model_data[];\n\n")
        f.write("#endif /* MODEL_DATA_H */\n")
    
    print(f"✓ Generated {output_h}")
    
    # Générer le fichier .c
    with open(output_c, 'w') as f:
        f.write("/**\n")
        f.write(" * TensorFlow Lite Model Data\n")
        f.write(" * Auto-generated from TFLite model\n")
        f.write(" */\n\n")
        f.write('#include "model_data.h"\n\n')
        f.write(f"const unsigned char model_data[MODEL_SIZE] = {{\n")
        
        # Écrire les bytes en format hexadécimal
        bytes_per_line = 12
        for i, byte in enumerate(tflite_model):
            if i % bytes_per_line == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}")
            if i < len(tflite_model) - 1:
                f.write(",")
                if (i + 1) % bytes_per_line == 0:
                    f.write("\n")
                else:
                    f.write(" ")
        
        f.write("\n};\n")
    
    print(f"✓ Generated {output_c}")
    print(f"\nTotal size: {model_size} bytes")
    print(f"\nNext steps:")
    print(f"  1. Copy {output_h} to your STM32 project Inc/ folder")
    print(f"  2. Copy {output_c} to your STM32 project Src/ folder")
    print(f"  3. Add TensorFlow Lite Micro library to your project")

if __name__ == "__main__":
    convert_tflite_to_c_array()