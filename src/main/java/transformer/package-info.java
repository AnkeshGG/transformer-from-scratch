/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Root package for the transformer-from-scratch library.
 *
 * <p>This library implements the encoder-decoder Transformer architecture described in:
 * <em>Attention Is All You Need</em> — Vaswani et al., 2017 (arXiv:1706.03762).
 *
 * <h2>Package structure</h2>
 * <ul>
 *   <li>{@code transformer} — top-level model ({@link transformer.Transformer},
 *       {@link transformer.TransformerConfig})
 *   <li>{@code transformer.tensor} — 2-D and 3-D tensor primitives
 *   <li>{@code transformer.embedding} — token embedding and positional encoding
 *   <li>{@code transformer.attention} — scaled dot-product and multi-head attention
 *   <li>{@code transformer.encoder} — encoder layer and encoder stack
 *   <li>{@code transformer.decoder} — decoder layer and decoder stack
 *   <li>{@code transformer.layers} — Linear, LayerNorm, FeedForward, Dropout
 *   <li>{@code transformer.activation} — ReLU and GELU activation functions
 *   <li>{@code transformer.loss} — cross-entropy loss
 *   <li>{@code transformer.tokenizer} — word-level tokenizer
 *   <li>{@code transformer.utils} — math utilities and weight initialisation
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * TransformerConfig config = TransformerConfig.tiny(vocabSize);
 * Transformer model = new Transformer(config, PAD_ID, BOS_ID, EOS_ID);
 *
 * // Teacher-forcing forward pass
 * Tensor3D probs = model.forward(sourceTokenIds, targetTokenIds);
 *
 * // Greedy autoregressive generation
 * int[] output = model.generate(sourceTokenIds, maxLength);
 * }</pre>
 *
 * @see transformer.Transformer
 * @see transformer.TransformerConfig
 */
package transformer;
