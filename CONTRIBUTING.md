# 🤝 Contributing to Transformer-from-Scratch

Thank you for your interest in contributing! Contributions are welcome across **mathematical correctness, Transformer components, testing, documentation, performance improvements, and future training capabilities**.

This project aims to keep the implementation understandable and close to the original Transformer architecture, so changes should preserve that focus.

---

## 📋 Getting Started

### Prerequisites

- **JDK 17** or higher
- **Maven 3.6+**
- **Git**

### Build & Test

```bash
git clone https://github.com/AnkeshGG/transformer-from-scratch.git
cd transformer-from-scratch
mvn clean test
```

### Run the Demo

```bash
mvn "-Dexec.mainClass=transformer.Main" exec:java
```

---

## 🏗️ Project Layout

```text
src/main/java/transformer/
├── Transformer.java              # Top-level encoder-decoder model
├── TransformerConfig.java        # Immutable model configuration
├── Main.java                     # End-to-end demonstration
├── tensor/                       # Matrix and Tensor3D primitives
├── embedding/                    # Token embedding and positional encoding
├── attention/                    # Scaled dot-product and multi-head attention
├── encoder/                      # Encoder layer stack
├── decoder/                      # Decoder layer stack
├── layers/                       # Linear, LayerNorm, FeedForward, Dropout
├── activation/                   # ReLU and GELU
├── loss/                         # Cross-entropy loss
├── tokenizer/                    # Word-level tokenizer
└── utils/                        # Mathematical utilities and random initialization

src/test/java/transformer/
├── MatrixTest.java
├── Tensor3DTest.java
├── AttentionTest.java
├── ComponentTest.java
└── MathematicalCorrectnessTest.java
```

The package structure mirrors the major Transformer components so contributors can make focused changes without placing unrelated functionality in the top-level model.

---

## 💡 How to Contribute

### Reporting Bugs

Open a GitHub Issue and include:

- What you expected to happen.
- What actually happened.
- Minimal reproduction steps.
- Relevant input tensor dimensions.
- The command used to reproduce the issue.
- Any stack trace or test failure output.

For mathematical bugs, include the expected calculation where possible.

### Suggesting Features

Open a GitHub Issue describing the proposed change and its motivation.

Useful future contributions include:

- Additional attention variants.
- Relative positional encodings.
- ALiBi-style positional bias.
- Flash-Attention-inspired optimisations.
- Beam-search decoding.
- Byte-pair encoding tokenization.
- Backpropagation and gradient computation.
- Adam optimisation.
- Mini-batch training.
- Weight serialization and checkpointing.

---

## 🔀 Submitting a Pull Request

1. Fork the repository.
2. Create a focused feature branch:

```bash
git checkout -b feature/your-feature-name
```

3. Implement the change.
4. Add or update tests for changed behaviour.
5. Run the complete test suite:

```bash
mvn clean test
```

6. Make sure all tests pass.
7. Keep commits focused and use clear commit messages.
8. Push your branch:

```bash
git push origin feature/your-feature-name
```

9. Open a Pull Request against `main`.

---

## 🧪 Testing Requirements

Every behavioural or mathematical change should include appropriate tests.

At minimum, tests should verify:

- Output shape.
- Dimension compatibility.
- Expected numerical behaviour.
- Error handling where applicable.
- Regression behaviour for existing functionality.

For a new mathematical layer, include at least one test using a small, manually verifiable example.

The project separates matrix, tensor, attention, component, and mathematical correctness tests, so new tests should follow the closest existing category.

---

## 🧮 Adding a New Transformer Layer or Component

When adding a new layer:

1. Place the class under the appropriate package.
2. Follow the existing `forward(Matrix)` / `forward(Tensor3D)` naming convention where applicable.
3. Document the mathematical operation.
4. Document input and output dimensions.
5. Add unit tests for shape and numerical behaviour.
6. Ensure invalid dimensions produce clear errors.
7. Keep the implementation independent of external ML frameworks.

For example, an attention-related component should clearly document shapes such as:

```text
Input:
[B × S × dModel]

After head split:
[B × numHeads × S × dHead]

Attention scores:
[B × numHeads × S × S]

Output:
[B × S × dModel]
```

---

## 📐 Code Style & Engineering Guidelines

- Follow standard Java conventions.
- Use clear, descriptive class and method names.
- Use 4-space indentation.
- Keep classes focused on a single responsibility.
- Add Javadoc to public classes and methods.
- Document mathematical operations where they are not immediately obvious.
- Include tensor dimensions in documentation for methods operating on `Matrix` or `Tensor3D`.
- Do not introduce ML-framework dependencies.
- Avoid silently accepting incompatible tensor dimensions.
- Prefer explicit validation and clear exceptions.
- Keep commits focused and easy to review.

---

## 🔢 Numerical & Mathematical Correctness

Transformer calculations are particularly sensitive to numerical mistakes.

When modifying attention or mathematical operations:

- Preserve the `1 / √dk` scaling in scaled dot-product attention.
- Use numerically stable softmax/log-softmax implementations.
- Apply causal masks before softmax.
- Preserve correct Q/K/V roles during cross-attention.
- Ensure `dModel % numHeads == 0`.
- Preserve residual connection dimensions.
- Keep LayerNorm across the intended embedding dimension.
- Avoid changing the original architecture unintentionally.

If changing an architectural choice, document the reason in the Pull Request.

---

## 📝 Documentation Changes

Documentation improvements are welcome.

When adding a new feature, update relevant documentation such as:

- `README.md`
- Architecture diagrams
- Mathematical explanations
- Usage examples
- Roadmap
- Project structure

Documentation should describe what the implementation actually supports and should not claim training or functionality that has not been implemented.

---

## 🗺️ Roadmap

| Area | Status |
| :--- | :--- |
| Forward pass — encoder + decoder | ✅ Complete |
| Greedy autoregressive generation | ✅ Complete |
| Cross-entropy loss | ✅ Complete |
| Backpropagation / gradient computation | 🔲 Planned |
| Adam optimiser | 🔲 Planned |
| Training loop with mini-batches | 🔲 Planned |
| Byte-pair encoding tokenizer | 🔲 Planned |
| Beam search | 🔲 Planned |
| Weight save / load | 🔲 Planned |

The roadmap is intentionally open to contributions that extend the implementation while preserving its educational and mathematical focus.

---

## 🔍 Pull Request Checklist

Before submitting a Pull Request, verify:

- [ ] The project compiles successfully.
- [ ] `mvn clean test` passes.
- [ ] New behaviour has tests.
- [ ] Tensor dimensions are documented.
- [ ] Mathematical changes are explained.
- [ ] No unnecessary ML-framework dependencies were added.
- [ ] Public APIs have appropriate documentation.
- [ ] README/documentation was updated when necessary.
- [ ] Commit history is focused and understandable.

---

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.
