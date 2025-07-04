<h1 align="center">🌿 Attention-Enhanced GCN with SPP for UAV-Captured Plant Imagery</h1>

In precision agriculture, early and accurate detection of crop diseases is vital for increasing yield and minimizing losses. Traditional deep learning models such as CNNs often overlook spatial and topological relationships in plant structures, especially in high-resolution UAV-captured images.

This project proposes a novel pipeline that integrates **Attention-enhanced Graph Convolutional Networks (GCNs)** with **Spatial Pyramid Pooling (SPP)** to improve plant disease classification from drone-captured soybean leaf images. The proposed model leverages:

- **Graph-based representation of leaf images**, converting superpixel segments into nodes and using their connectivity as edges,
- **Multi-head attention** to enhance feature extraction by focusing on critical disease patterns,
- **SPP blocks** to capture multi-scale spatial features, ensuring robustness across different plant image resolutions.
---

## 🧾 Project Overview
<div style="width:100%; overflow-x:auto; margin:1em 0; font-family:Arial, sans-serif;" align = "center">

| Attribute       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| **📚 Framework** | PyTorch, PyTorch Geometric, Scikit-learn , Numpy, Pandas, Seaborn, Matplotlib                                  |
| **🧠 Models Used** | Graph Convolutional Network (GCN), Ensenble CNN (ShuffleNet_V2, EfficientNet_B0)                           |
| **📷 Input Format** | Drone-captured Images (~256×256 px)                                      |
| **🎯 Output**   | 4-Class Image Classification (Healthy, Mosaic, Rust, Pest)                                              |
| **🧪 File**     | `gcn-code.ipynb`                                                           |
| **📁 Dataset Source** | [Soybean Leaf Dataset](https://data.mendeley.com/datasets/hkbgh5s3b7/1) |

</div>

---
## 🌿 Workflow Diagram
The model begins by taking an input image and extracting features using two lightweight convolutional neural network backbones: EfficientNet-B0 and ShuffleNetV2, which generate feature vectors of dimensions 1280 *1024 respectively. These features are then combined through concatenation to form a single fused feature vector. This fused vector is passed through a Spatial Pyramid Pooling (SPP) layer, which transforms it into a fixed-size feature representation. The resulting vector is fed into the first layer of a Graph Convolutional Network (GCN), which computes intermediate feature representations. These intermediate features are further refined using an attention mechanism that highlights the most important information. In the second GCN layer, the model combines the attention-enhanced features with the earlier intermediate features using element-wise operations, activation functions, and a skip connection to retain useful information from previous layers. The final set of features is passed to a fully connected layer with a softmax activation function to classify the image into one of four categories: Healthy, Pest Attack, Rust, or Mosaic.


<p align="center">
  <img src="https://github.com/SayanDhar10/GCN-SPP-UAV/blob/6083104a21a965a0ff7c2ee43ed661ece25ec8f9/System%20Architecture%20Images/Flow_Diagram.png" height="350px" width="50%">
</p>



---

## 🧠 Model Architecture
The presented model architecture integrates Spatial Pyramid Pooling (SPP), Graph Convolutional Networks (GCN), and an Attention Module to effectively classify input features and produce output images. Initially, input features are processed through SPP to extract multi-scale spatial information and generate a fixed-size feature vector. This vector is passed into GCN Layer 1, which comprises multiple hidden layers with ReLU activation and outputs intermediate features (H₁). These intermediate features are then fed into an attention module that assigns weights, resulting in weighted features (A). Both H₁ and A are combined and processed through GCN Layer 2, which also uses ReLU-activated hidden layers to produce enhanced features (H₂). Simultaneously, a skip path directly transfers H₁ to the final stage, where it is added to H₂ to retain essential information. The enhanced output is passed through a fully connected layer followed by softmax activation to generate the final classified output image. This architecture benefits from multi-scale spatial context, attention-driven feature weighting, and skip connections for improved performance and robustness.


<p align="center">
  <img src="https://github.com/SayanDhar10/GCN-SPP-UAV/blob/6083104a21a965a0ff7c2ee43ed661ece25ec8f9/System%20Architecture%20Images/Model_Architecture.png" style="max-width: 90%; height: auto;" />
</p>



---

## <div align="center">🗂 Dataset Overview</div>

The **MH-SoyaHealthVision** dataset is a comprehensive resource for integrated crop health assessment in soybean farming. It combines **ground-level leaf images** and **UAV-captured aerial images** from soybean fields in the Maharashtra region. This dual-perspective approach enables effective disease and pest detection using deep learning.

### <div align="center">📊 Class Distribution</div>

<table align="center">
  <tr>
    <td align="center"><img src="https://github.com/SayanDhar10/GCN-SPP-UAV/blob/6083104a21a965a0ff7c2ee43ed661ece25ec8f9/Input_Images/Healthy.jpg" width="120px"></td>
    <td align="center"><img src="https://github.com/SayanDhar10/GCN-SPP-UAV/blob/6083104a21a965a0ff7c2ee43ed661ece25ec8f9/Input_Images/Rust.jpg" width="120px"></td>
    <td align="center"><img src="https://github.com/SayanDhar10/GCN-SPP-UAV/blob/6083104a21a965a0ff7c2ee43ed661ece25ec8f9/Input_Images/Mosaic.jpg" width="120px"></td>
    <td align="center"><img src="https://github.com/SayanDhar10/GCN-SPP-UAV/blob/6083104a21a965a0ff7c2ee43ed661ece25ec8f9/Input_Images/Pest.jpg" width="120px"></td>
  </tr>
  <tr>
    <td align="center"><b>Healthy</b></td>
    <td align="center"><b>Rust</b></td>
    <td align="center"><b>Mosaic</b></td>
    <td align="center"><b>Pest</b></td>
  </tr>
</table>

The dataset includes:

- **High-resolution images of soybean leaves** showing signs of rust, mosaic virus, septoria brown spot, frog-eye leaf spot, and pest attacks (e.g., caterpillars and semiloopers).
- **UAV-based aerial images** capturing large-scale field patterns of rust, mosaic virus, and pest infestations.

These images are categorized into respective folders for each class, making it suitable for both classification and segmentation tasks.

📦 **Total Images:** 5,680  
🖼 **Format:** `.jpg`  
🗃 **Structure:** Class-wise folders  
🧪 **Preprocessing:** Resizing, patch extraction, graph construction  
📥 **Download Link:** [MH-SoyaHealthVision Dataset on Mendeley](https://data.mendeley.com/datasets/hkbgh5s3b7/1)


---

## 🧪 Notebook Workflow
<div>
  
| Step                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| 📥 Data Loading      | Dataset parsed and structured into graph format                             |
| 🧼 Preprocessing      | Image → Superpixels → Graph (nodes/edges)                                  |
| 🧠 Model              | R-GCN architecture using PyTorch Geometric                                  |
| 🔁 Training           | CrossEntropyLoss + Adam optimizer                                           |
| 📊 Evaluation         | Accuracy, Confusion Matrix, ROC-AUC, Graph Visuals                          |
| 🔍 Inference          | Predict class for new images using trained GCN                             |

</div>

---

<h2>⚙️ Model Configuration &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 📈 Performance Summary</h2>
<div>
<table>
  <tr>
    <td>

<!-- Left Table -->
<b>⚙️ Model Configuration</b>

<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Graph Layers</td><td>R-GCN (2-layer)</td></tr>
  <tr><td>Input Features</td><td>16</td></tr>
  <tr><td>Hidden Features</td><td>32</td></tr>
  <tr><td>Output Classes</td><td>4</td></tr>
  <tr><td>Optimizer</td><td>Adam</td></tr>
  <tr><td>Loss Function</td><td>CrossEntropy</td></tr>
  <tr><td>Batch Size</td><td>16</td></tr>
</table>

  </td>
  <td style="width: 40px;"></td>
  <td>

<!-- Right Table -->
<b>📈 Performance Summary</b>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>✅ Final Accuracy</td><td><b>94.50%</b></td></tr>
  <tr><td>🔍 ROC-AUC</td><td><b>0.96 (macro avg)</b></td></tr>
  <tr><td>📉 Final Loss</td><td><b>0.217</b></td></tr>
  <tr><td>📊 Confusion Matrix</td><td>4×4</td></tr>
</table>

  </td>
  </tr>
</table>
</div>

---

## 📊 Model Outputs

<p >
  <img src="Output_Images/Accuracy_Curve.png" height="220px" style="margin-right: 10px;">
  <img src="Output_Images/Loss_Curve.png" height="220px">
</p>

<p >
  <img src="Output_Images/Confusion_Matrix.png" height="220px" style="margin-right: 10px;">
  <img src="Output_Images/ROC_Curve.png" height="220px">
</p>

<p >
  <img src="Output_Images/Graph_Visualization.png" height="220px" style="margin-right: 10px;">
  <img src="https://github.com/SayanDhar10/GCN-SPP-UAV/blob/6083104a21a965a0ff7c2ee43ed661ece25ec8f9/System%20Architecture%20Images/Flow_Diagram.png" height="220px">
</p>

---

## ✅ Conclusion

This work shows that **Graph Neural Networks** — specifically **Relational GCNs** — can effectively model spatial and topological relationships within drone-captured crop images, achieving over **94% accuracy** in detecting:

- 🟢 Healthy
- 🍂 Rust
- 🧬 Mosaic
- 🐛 Pest

GCNs provide a compact, explainable architecture for crop health monitoring and precision agriculture.

---

## 🔮 Future Enhancements

- 🛰️ **Integrate remote sensing data** (NDVI, multispectral)  
- 🌐 **Real-time GCN inference** using ONNX or TensorRT  
- 🔍 **Model Explainability** using Graph Attention or Grad-CAM  
- 🚀 **Deploy with Flask/Streamlit** for mobile/web-based monitoring  
- 🧪 Test **GAT / GIN / ChebNet** for improved disease segmentation  

---

## 🚀 Getting Started

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SayanDhar10/GCN-SPP-UAV.git
cd GCN-SPP-UAV
```

# Install project-specific packages
```bash
pip install -r requirements.txt
```

