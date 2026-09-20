/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Decoder stack.
 *
 * <ul>
 *   <li>{@link transformer.decoder.DecoderLayer} — one decoder block:
 *       masked self-attention → Add&amp;Norm → cross-attention → Add&amp;Norm
 *       → feed-forward → Add&amp;Norm.
 *   <li>{@link transformer.decoder.Decoder} — target embedding, positional
 *       encoding, and {@code N} stacked decoder layers.
 * </ul>
 */
package transformer.decoder;
