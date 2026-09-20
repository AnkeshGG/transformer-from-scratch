/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Encoder stack.
 *
 * <ul>
 *   <li>{@link transformer.encoder.EncoderLayer} — one encoder block:
 *       multi-head self-attention → Add&amp;Norm → feed-forward → Add&amp;Norm.
 *   <li>{@link transformer.encoder.Encoder} — token embedding, positional
 *       encoding, and {@code N} stacked encoder layers.
 * </ul>
 */
package transformer.encoder;
