/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Attention mechanisms.
 *
 * <ul>
 *   <li>{@link transformer.attention.ScaledDotProductAttention} — computes
 *       {@code Attention(Q,K,V) = softmax(QKᵀ / √dk + mask) V}.
 *   <li>{@link transformer.attention.MultiHeadAttention} — projects Q/K/V into
 *       {@code h} subspaces, runs attention in each head, and concatenates results.
 *   <li>{@link transformer.attention.AttentionMask} — factory methods for causal
 *       (lower-triangular) masks and padding masks.
 * </ul>
 */
package transformer.attention;
