# 🤖 Real-Time Motion Classification on STM32
Real-time motion classification on STM32 using embedded AI - TinyML project

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-STM32L476JG-green.svg)](https://www.st.com)
[![TinyML](https://img.shields.io/badge/TinyML-Edge_AI-orange.svg)](https://tinyml.org)

An intelligent motion recognition system using embedded AI on STM32 microcontroller. Classifies 3 types of movements (Circular, Rectangular, Linear) in real-time with **97% accuracy**.
## ✨ Key Features

- 🧠 **Ultra-Compact CNN**: Only 875 parameters (8.5 KB)
- ⚡ **Real-Time Inference**: <100ms latency
- 📊 **High Accuracy**: 97% on validation set
- 💾 **Low Memory**: 29KB RAM / 124KB Flash
- 📡 **Live Monitoring**: USB CDC streaming
- 🎨 **Python Dashboard**: Real-time visualization

## 🛠️ Hardware
- **Development Board**: NUCLEO-F303RE
- **Sensor Board**: STM32 SensorTile (STEVAL-STLCS01V1)
- **MCU**: STM32F303RE (ARM Cortex-M4 @ 72MHz)
  - Flash: 512 KB
  - RAM: 64 KB
  - FPU: Yes (hardware floating-point)
- **Sensors**: LSM6DSM (6-axis IMU - Accelerometer + Gyroscope)
- **Configuration**: SensorTile mounted on NUCLEO expansion connector
## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97% |
| Inference Time | <100 ms |
| RAM Usage | 29 KB (23%) |
| Flash Usage | 124 KB (12%) |
| Model Size | 8.5 KB |
| Parameters | 875 |

## 🚀 Quick Start

### STM32 Setup
1. Open project in STM32CubeIDE
2. Build and flash to SensorTile
3. Connect via USB
4. 
## 🧠 Model Architecture

Input: 100 samples × 6 features (ax, ay, az, gx, gy, gz)
    ↓
Conv1D (32 filters, kernel=3) + ReLU
    ↓
MaxPooling1D (pool_size=2)
    ↓
Conv1D (64 filters, kernel=3) + ReLU
    ↓
MaxPooling1D (pool_size=2)
    ↓
Flatten
    ↓
Dense (64 neurons) + ReLU + Dropout(0.5)
    ↓
Dense (3 neurons) + Softmax
    ↓
Output: [Circular, Rectangular, Linear]
**Optimization**: Model quantized and optimized using X-CUBE-AI for embedded deployment.

1. **Open Project**
   STM32CubeIDE → File → Import → Existing Projects
   Select: STM32_Project folder
2. **Build**
   Project → Build All (Ctrl+B)
   Verify: 0 errors

3. **Flash to NUCLEO**
   Run → Debug (F11)
   Program will flash automatically via ST-LINK
   Press Resume (F8) to run
   Stop debugger after flash

### Step 2: Python Dashboard
cd Python_Dashboard
pip install -r requirements.txt
python dashboard.py
### Step 3: Usage
1. **Connect** NUCLEO to PC via USB
2. **Open dashboard** and select COM port
3. **Double-tap** the SensorTile to trigger classification
4. **Perform movement** (circular, rectangular, or linear)
5. **Wait 10 seconds** for data collection (100 samples @ 10Hz)
6. **See result** displayed with confidence percentage!

## 📁 Project Structure
STM32-Motion-Classification/
├── STM32_Project/          # Embedded C code for STM32F303RE
│   ├── Inc/               # Header files
│   ├── Src/               # Source files (main.c, etc.)
│   └── X-CUBE-AI/App/     # AI model files (network.c, etc.)
├── Python_Dashboard/       # Real-time visualization GUI
│   ├── dashboard.py
│   └── requirements.txt
├── ML_Model/              # Model training scripts
│   └── activity_model_quantized.tflite       
        
## 🔬 Technical Details

### Data Processing Pipeline

1. **Data Collection**: 
   - Accelerometer + Gyroscope data at 10 Hz
   - Window size: 100 samples (10 seconds)
   - 6 features per sample (ax, ay, az, gx, gy, gz)

2. **Preprocessing**:
   - Z-score normalization per channel
   - Real-time computation on MCU

3. **Inference**:
   - CNN forward pass using X-CUBE-AI runtime
   - Softmax activation for probability distribution
   - Argmax for final class prediction

4. **Output**:
   - Class label (Circular/Rectangular/Linear)
   - Confidence percentage
   - Transmitted via USB CDC

### Memory Optimization

- **Static allocation**: All buffers pre-allocated
- **Quantization**: Model weights quantized to int8
- **Code optimization**: -O2 compiler flags
- **HAL drivers**: Minimal configuration

### Learning Outcomes

- ✅ End-to-end TinyML pipeline development
- ✅ Embedded systems programming (STM32)
- ✅ Deep learning model optimization
- ✅ Real-time signal processing
- ✅ Hardware-software integration
- ✅ Edge computing implementation

## 📚 Documentation

- [Complete Setup Guide](Documentation/SETUP_GUIDE.md)
- [STM32F303RE Reference Manual](https://www.st.com/resource/en/reference_manual/dm00043574-stm32f303xb-c-d-e-stm32f303x6-8-stm32f328x8-stm32f358xc-stm32f398xe-advanced-armbased-mcus-stmicroelectronics.pdf)
- [X-CUBE-AI Documentation](https://www.st.com/en/embedded-software/x-cube-ai.html)
- [SensorTile User Manual](https://www.st.com/resource/en/user_manual/dm00310969-getting-started-with-the-stevalstlcs01v1-sensortile-integrated-development-platform-stmicroelectronics.pdf)

⭐ **Star this repo if you find it useful!**

💡 **Interested in TinyML and Edge AI? Let's connect!**

**Légende** :
```
Hardware Setup: NUCLEO-F303RE with SensorTile expansion board
