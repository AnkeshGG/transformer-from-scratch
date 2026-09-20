/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Embedding layers.
 *
 * <ul>
 *   <li>{@link transformer.embedding.TokenEmbedding} — lookup table of shape
 *       {@code [vocabSize × dModel]}, scaled by {@code √dModel} at retrieval.
 *   <li>{@link transformer.embedding.PositionalEncoding} — fixed sinusoidal
 *       encoding added to the embedded sequence (not trainable).
 * </ul>
 */
package transformer.embedding;
