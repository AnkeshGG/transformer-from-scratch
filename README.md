# 🤖 Transformer-from-Scratch — Java Transformer Implementation

A complete Java implementation of the **encoder-decoder Transformer architecture** from *Attention Is All You Need* (Vaswani et al., 2017).

Built from the ground up using **pure Java 17**, without ML-framework dependencies. The project focuses on making the internal mathematics and architecture of a Transformer explicit, testable, and easy to study.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Java](https://img.shields.io/badge/Java-17-orange?logo=oracle)](https://openjdk.org/projects/jdk/17/)
[![Maven](https://img.shields.io/badge/Maven-3.6%2B-orange.svg)](https://maven.apache.org)

---

## 🧠 Core Architectural Paradigms

The project recreates the original 2017 Transformer architecture at the component level rather than relying on a pre-built Transformer or ML framework.

* **Pure Java Transformer Core**: Matrix and tensor operations, attention mechanisms, embeddings, normalization, feed-forward layers, and model composition are implemented directly in Java.
* **Encoder-Decoder Architecture**: Source tokens pass through stacked encoder layers, while target tokens pass through masked self-attention, encoder-decoder cross-attention, and feed-forward layers.
* **Explicit Tensor Dimensions**: Operations use clearly defined `[B × S × dModel]`-style shapes to make data flow and dimension handling easier to understand.
* **Original Transformer Design**: Uses fixed sinusoidal positional encoding and Post-LayerNorm residual blocks to stay aligned with the original paper.
* **Numerically Stable Computation**: Softmax and cross-entropy calculations use stable formulations to reduce overflow and underflow issues.
* **Deterministic Experiments**: Configurable random seeds make initialization and experiments reproducible.
* **Forward-Pass Focus**: The current implementation supports inference and greedy autoregressive generation; backpropagation and model training are planned extensions.

---

## 🚀 Key Features

### 🧩 Transformer Architecture
- Full encoder-decoder forward pass.
- Configurable encoder and decoder layer counts.
- Multi-head scaled dot-product attention.
- Encoder-decoder cross-attention.
- Residual connections with Post-LayerNorm.
- Position-wise feed-forward networks.

### 🧠 Embeddings & Attention
- Token embedding lookup with `√dModel` scaling.
- Fixed sinusoidal positional encoding.
- Causal masking for decoder self-attention.
- Padding masking for encoder self-attention and decoder cross-attention.
- Configurable ReLU/GELU activation support.

### 📝 Tokenization & Inference
- Simple word-level tokenizer.
- `PAD`, `BOS`, `EOS`, and `UNK` special tokens.
- Greedy autoregressive token generation.
- Numerically stable softmax and log-softmax operations.

### 🧪 Testing & Correctness
- Unit and integration tests for matrix and tensor operations.
- Attention and masking tests.
- Component-level model tests.
- Mathematical correctness tests with known numerical values.
- Explicit dimension and error-handling checks.

---

## 🖼️ Transformer Architecture

The implementation follows the encoder-decoder flow of the original Transformer:

```text
Source Tokens [B × S]
       │
       ▼
┌─────────────────────────────────────┐
│              ENCODER                │
│                                     │
│ Token Embedding × √dModel           │
│ + Positional Encoding               │
│                                     │
│ ┌───────────────────────────────┐   │
│ │ Encoder Layer × N              │   │
│ │ Multi-Head Self-Attention      │   │
│ │ → Add & LayerNorm              │   │
│ │ Feed Forward                   │   │
│ │ → Add & LayerNorm              │   │
│ └───────────────────────────────┘   │
└───────────────────┬─────────────────┘
                    │ Encoder Output
                    │ [B × S × dModel]
                    │
Target Tokens [B × T]
       │            │
       ▼            │
┌─────────────────────────────────────┐
│              DECODER                │
│                                     │
│ Token Embedding × √dModel           │
│ + Positional Encoding               │
│                                     │
│ ┌───────────────────────────────┐   │
│ │ Decoder Layer × N              │   │
│ │ Masked Self-Attention          │   │
│ │ → Add & LayerNorm              │   │
│ │ Cross-Attention                │◄──┘
│ │ → Add & LayerNorm              │
│ │ Feed Forward                   │
│ │ → Add & LayerNorm              │
│ └───────────────────────────────┘   │
└───────────────────┬─────────────────┘
                    │ Decoder Output
                    │ [B × T × dModel]
                    ▼
              Linear Projection
              dModel → vocabSize
                    │
                    ▼
                  Softmax
          [B × T × vocabSize]
```

---

## 📐 Key Equations

| Component | Formula |
| :--- | :--- |
| **Scaled Dot-Product Attention** | `Attention(Q,K,V) = softmax(QKᵀ / √dk + mask) · V` |
| **Multi-Head Attention** | `MultiHead = Concat(head₁,…,headₕ) · Wₒ` |
| **Positional Encoding** | `PE(pos,2i)=sin(pos/10000^(2i/d))`, `PE(pos,2i+1)=cos(pos/10000^(2i/d))` |
| **Layer Normalisation** | `LayerNorm(x)ᵢ = γᵢ(xᵢ − μ)/√(σ² + ε) + βᵢ` |
| **Feed Forward** | `FFN(x) = W₂ · ReLU(W₁x + b₁) + b₂` |
| **Cross-Entropy** | `CE = −logSoftmax(logits)[target]` |

---

## 🛠️ Tech Stack & Dependencies

| Layer | Technology |
| :--- | :--- |
| **Language / Runtime** | Java 17 |
| **Build Tool** | Maven 3.6+ |
| **Numerical Core** | Custom Java `Matrix` and `Tensor3D` implementations |
| **Model Architecture** | Encoder-Decoder Transformer |
| **Testing** | Maven test suite |
| **External ML Frameworks** | None |
| **License** | MIT |

---

## 🏗️ Project Structure

```text
transformer-from-scratch/
├── src/
│   ├── main/java/transformer/
│   │   ├── Transformer.java              # Top-level model
│   │   ├── TransformerConfig.java        # Immutable model configuration
│   │   ├── Main.java                     # End-to-end demonstration
│   │   │
│   │   ├── tensor/
│   │   │   ├── Matrix.java               # 2-D matrix operations
│   │   │   └── Tensor3D.java             # 3-D tensor operations
│   │   │
│   │   ├── embedding/
│   │   │   ├── TokenEmbedding.java       # Token lookup and scaling
│   │   │   └── PositionalEncoding.java   # Sinusoidal positional encoding
│   │   │
│   │   ├── attention/
│   │   │   ├── ScaledDotProductAttention.java
│   │   │   ├── MultiHeadAttention.java
│   │   │   └── AttentionMask.java
│   │   │
│   │   ├── encoder/
│   │   │   ├── EncoderLayer.java
│   │   │   └── Encoder.java
│   │   │
│   │   ├── decoder/
│   │   │   ├── DecoderLayer.java
│   │   │   └── Decoder.java
│   │   │
│   │   ├── layers/
│   │   │   ├── Linear.java
│   │   │   ├── LayerNorm.java
│   │   │   ├── FeedForward.java
│   │   │   └── Dropout.java
│   │   │
│   │   ├── activation/
│   │   │   └── ActivationFunctions.java
│   │   │
│   │   ├── loss/
│   │   │   └── CrossEntropyLoss.java
│   │   │
│   │   ├── tokenizer/
│   │   │   └── SimpleTokenizer.java
│   │   │
│   │   └── utils/
│   │       ├── MathUtils.java
│   │       └── RandomUtils.java
│   │
│   └── test/java/transformer/
│       ├── MatrixTest.java
│       ├── Tensor3DTest.java
│       ├── AttentionTest.java
│       ├── ComponentTest.java
│       └── MathematicalCorrectnessTest.java
│
├── pom.xml
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## 📋 Prerequisites

- **JDK 17** or later
- **Maven 3.6+**
- **Git**

No external machine-learning framework is required.

---

## ⚡ Quick Start & Run

### Option A: Clone and Build

```bash
git clone https://github.com/AnkeshGG/transformer-from-scratch.git
cd transformer-from-scratch
```

### Compile the Project

```bash
mvn clean compile
```

### Run the Complete Test Suite

```bash
mvn test
```

### Run the Transformer Demo

```bash
mvn "-Dexec.mainClass=transformer.Main" exec:java
```

### Build the Standalone JAR

```bash
mvn package
```

Then:

```bash
java -jar target/transformer-from-scratch-1.0.0-jar-with-dependencies.jar
```

---

## ⚙️ Configuration

The model can be configured using `TransformerConfig.Builder`:

```java
TransformerConfig config = new TransformerConfig.Builder()
    .dModel(512)
    .numHeads(8)
    .dFF(2048)
    .numEncoderLayers(6)
    .numDecoderLayers(6)
    .vocabSize(32000)
    .maxSequenceLength(512)
    .randomSeed(42L)
    .debugMode(false)
    .build();
```

A smaller configuration is available for experiments and demonstrations:

```java
TransformerConfig config =
    TransformerConfig.tiny(vocabSize);
```

The original paper-style base configuration can also be created with:

```java
TransformerConfig config =
    TransformerConfig.paperBase(vocabSize);
```

---

## 💻 Usage

```java
TransformerConfig config = TransformerConfig.tiny(vocabSize);

Transformer model =
    new Transformer(config, PAD_ID, BOS_ID, EOS_ID);

// Teacher-forcing forward pass
int[][] src = {{3, 4, 5}};
int[][] tgt = {{1, 3, 4, 5}};

Tensor3D probabilities =
    model.forward(src, tgt);

// Greedy autoregressive generation
int[] generated =
    model.generate(new int[]{3, 4, 5}, 20);
```

The forward pass produces:

```text
[B × T × vocabSize]
```

where:

- `B` = batch size
- `T` = target sequence length
- `vocabSize` = number of tokens in the vocabulary

---

## 🧪 Running Tests

Run the complete test suite:

```bash
mvn clean test
```

The test suite covers:

- Matrix operations
- Tensor3D operations
- Attention calculations
- Causal and padding masks
- Layer components
- Model integration
- Mathematical correctness
- Error handling

The current project README reports **99 tests** covering these areas. fileciteturn0file0L168-L173

---

## 🔬 Design Decisions

| Decision | Choice | Notes |
| :--- | :--- | :--- |
| **Architecture** | Encoder-Decoder Transformer | Based on the 2017 Transformer |
| **LayerNorm Position** | Post-LN | LayerNorm is applied after the residual Add |
| **Activation** | ReLU by default | GELU is also available |
| **Positional Encoding** | Fixed sinusoidal | Non-trainable |
| **Weight Initialization** | Xavier uniform | Used for neural network weights |
| **Softmax** | Max-subtraction stabilised | Helps prevent exponential overflow |
| **Decoding** | Greedy / argmax | Beam search can be added later |
| **Training** | Not implemented | Current implementation focuses on forward inference |

---

## 🗺️ Roadmap

| Area | Status |
| :--- | :--- |
| Forward pass — encoder + decoder | ✅ Complete |
| Greedy autoregressive generation | ✅ Complete |
| Cross-entropy loss | ✅ Complete |
| Backpropagation / gradient computation | 🔲 Planned |
| Adam optimiser | 🔲 Planned |
| Mini-batch training loop | 🔲 Planned |
| Byte-pair encoding tokenizer | 🔲 Planned |
| Beam search | 🔲 Planned |
| Weight save / load | 🔲 Planned |

---

## 🤝 Contributing

Contributions are welcome! Improvements to mathematical correctness, tests, documentation, performance, and Transformer extensions are especially useful.

Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** before opening a pull request.

---

## 📚 Reference

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. *Advances in Neural Information Processing Systems*, 30.

Paper: https://arxiv.org/abs/1706.03762

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 About

**Ankesh Kumar**

> Built to study and demonstrate the internal architecture, mathematics, and data flow of the original Transformer model using pure Java, without relying on machine-learning frameworks.

### Connect

- 🌐 **GitHub**: [@AnkeshGG](https://github.com/AnkeshGG)
- 💼 **LinkedIn**: [Ankesh Kumar](https://www.linkedin.com/in/ankeshgg/)

---

## 🙏 Acknowledgements

- **Vaswani et al.** — Original *Attention Is All You Need* Transformer architecture.
- **OpenJDK** — Java runtime and standard library.
- **Maven** — Project build and dependency management.

---

## 📊 Project Status

🟢 **Active**

- **Version**: 1.0.0
- **Target JDK**: Java 17+
- **Build Tool**: Maven 3.6+
- **Architecture**: Encoder-Decoder Transformer
- **Training**: Forward pass and inference only

---

*Built from the mathematics up — one attention head at a time. 🤖*

[![Built with love](https://img.shields.io/badge/Built_with-%E2%80%9DAttention%E2%80%9D-red)](https://github.com/AnkeshGG/transformer-from-scratch)
