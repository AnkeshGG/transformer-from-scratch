/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.layers;

import transformer.tensor.Tensor3D;

/**
 * Dropout regularisation layer.
 *
 * <p>During inference ({@code training=false}), this layer is a pass-through
 * that returns its input unchanged — no activations are zeroed.
 *
 * <p>During training ({@code training=true}), each activation is independently
 * set to zero with probability {@code dropoutRate}, and the remaining values
 * are scaled by {@code 1/(1−p)} to preserve expected magnitude (inverted
 * dropout). <strong>Training mode is not yet implemented</strong>; calling
 * {@link #forward} with {@code training=true} throws
 * {@link UnsupportedOperationException}.
 *
 * <p>Dropout is applied after attention and after the FFN sublayer, before the
 * residual addition. The paper's base model uses a dropout rate of 0.1.
 */
public final class Dropout {

    /** Dropout probability p ∈ [0, 1). */
    private final double dropoutRate;

    /**
     * {@code false} = inference (pass-through).
     * {@code true}  = training mode (not yet implemented).
     */
    private final boolean training;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * @param dropoutRate probability of zeroing each activation; must be in [0, 1)
     * @param training    whether the model is in training mode
     * @throws IllegalArgumentException if {@code dropoutRate} is outside [0, 1)
     */
    public Dropout(double dropoutRate, boolean training) {
        if (dropoutRate < 0.0 || dropoutRate >= 1.0)
            throw new IllegalArgumentException(
                "dropoutRate must be in [0, 1), got: " + dropoutRate);
        this.dropoutRate = dropoutRate;
        this.training    = training;
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Apply dropout to {@code input}.
     *
     * <p>When {@code training=false} or {@code dropoutRate==0}, returns
     * {@code input} unchanged. Otherwise throws
     * {@link UnsupportedOperationException}.
     *
     * @param input input tensor [B × S × dModel]
     * @return output tensor (same shape as input)
     */
    public Tensor3D forward(Tensor3D input) {
        if (!training || dropoutRate == 0.0) {
            return input;
        }
        // Inverted dropout (training): scale surviving activations by 1/(1-p).
        // Not implemented — raises an explicit error rather than silently failing.
        throw new UnsupportedOperationException(
            "Dropout in training mode is not yet implemented. Use training=false for inference.");
    }
}
