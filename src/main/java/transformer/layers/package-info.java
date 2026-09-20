/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Reusable neural-network layers.
 *
 * <ul>
 *   <li>{@link transformer.layers.Linear} — fully-connected layer:
 *       {@code y = xWᵀ + b}, Xavier-uniform initialisation.
 *   <li>{@link transformer.layers.LayerNorm} — layer normalisation:
 *       {@code γ(x − μ)/√(σ² + ε) + β}.
 *   <li>{@link transformer.layers.FeedForward} — position-wise two-layer MLP:
 *       {@code W₂ · act(W₁x + b₁) + b₂}.
 *   <li>{@link transformer.layers.Dropout} — pass-through at inference;
 *       throws at training time until backpropagation is implemented.
 * </ul>
 */
package transformer.layers;
