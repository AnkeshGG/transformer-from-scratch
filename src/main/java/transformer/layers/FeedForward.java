/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.layers;

import transformer.activation.ActivationFunctions;
import transformer.tensor.Matrix;
import transformer.tensor.Tensor3D;

import java.util.Random;

/**
 * FeedForward — Position-wise Feed-Forward Network (FFN).
 *
 * <h2>What it does</h2>
 * <p>The FFN is applied identically and independently to each token position.
 * It consists of two linear transformations with an activation function in between.
 *
 * <h2>Mathematical operation</h2>
 * <pre>
 *   FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂
 *
 *   where:
 *     x  ∈ R^{dModel}  — input (one token's embedding)
 *     W₁ ∈ R^{dFF × dModel}  — first projection (expand)
 *     b₁ ∈ R^{dFF}           — first bias
 *     W₂ ∈ R^{dModel × dFF}  — second projection (contract)
 *     b₂ ∈ R^{dModel}        — second bias
 * </pre>
 *
 * <p>Dimensions flow:
 * <pre>
 *   dModel → [Linear] → dFF → [Activation] → dFF → [Linear] → dModel
 * </pre>
 *
 * <h2>Typical dimensions</h2>
 * <p>In the original paper: dModel = 512, dFF = 2048 (4× expansion).
 * The expanded inner layer acts as a kind of "memory bank" per token.
 *
 * <h2>Position-wise</h2>
 * <p>The same W₁, b₁, W₂, b₂ are applied to every position in the sequence.
 * This is equivalent to applying a shared 1-D convolution over the sequence.
 *
 * <h2>Activation</h2>
 * <p>Original paper uses ReLU. This implementation defaults to ReLU but
 * supports switching to GELU via the constructor.
 *
 * <h2>Dimensions for batched input</h2>
 * <pre>
 *   Input:  Tensor3D [B × S × dModel]
 *   Output: Tensor3D [B × S × dModel]  (same shape)
 * </pre>
 */
public final class FeedForward {

    /** First linear layer: dModel → dFF (expansion). */
    private final Linear linear1;

    /** Second linear layer: dFF → dModel (contraction). */
    private final Linear linear2;

    /** Activation function type used between the two linear layers. */
    private final ActivationFunctions.Type activationType;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct a FeedForward sublayer with ReLU activation.
     *
     * @param dModel model embedding dimension (input and output)
     * @param dFF    inner feed-forward dimension (typically 4 × dModel)
     * @param rng    seeded random generator for reproducible weight init
     */
    public FeedForward(int dModel, int dFF, Random rng) {
        this(dModel, dFF, ActivationFunctions.Type.RELU, rng);
    }

    /**
     * Construct a FeedForward sublayer with a configurable activation function.
     *
     * @param dModel     model embedding dimension
     * @param dFF        inner feed-forward dimension
     * @param activation activation function type (RELU or GELU)
     * @param rng        seeded random generator
     */
    public FeedForward(int dModel, int dFF,
                       ActivationFunctions.Type activation, Random rng) {
        this.linear1         = new Linear(dModel, dFF, rng);   // dModel → dFF
        this.linear2         = new Linear(dFF, dModel, rng);   // dFF → dModel
        this.activationType  = activation;
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Apply the FFN to a sequence represented as a Matrix.
     *
     * <p>Operation per token (row):
     * <pre>
     *   hidden = activation(input · W₁^T + b₁)   [S × dFF]
     *   output = hidden  · W₂^T + b₂              [S × dModel]
     * </pre>
     *
     * @param input Matrix of shape [S × dModel]
     * @return output Matrix of shape [S × dModel]
     */
    public Matrix forward(Matrix input) {
        // Step 1: first linear projection (expand): [S × dModel] → [S × dFF]
        Matrix hidden = linear1.forward(input);

        // Step 2: activation function applied element-wise.
        Matrix activated = applyActivation(hidden);

        // Step 3: second linear projection (contract): [S × dFF] → [S × dModel]
        return linear2.forward(activated);
    }

    /**
     * Apply the FFN to a batched 3-D tensor.
     *
     * <p>Each batch slice is processed independently:
     * <pre>
     *   Input:  Tensor3D [B × S × dModel]
     *   Output: Tensor3D [B × S × dModel]
     * </pre>
     *
     * @param input Tensor3D of shape [B × S × dModel]
     * @return output Tensor3D of shape [B × S × dModel]
     */
    public Tensor3D forward(Tensor3D input) {
        Tensor3D output = new Tensor3D(input.batch, input.rows, input.cols);
        for (int b = 0; b < input.batch; b++) {
            output.setMatrix(b, forward(input.getMatrix(b)));
        }
        return output;
    }

    // ── Activation application ───────────────────────────────────────────────

    /**
     * Apply the configured activation function element-wise to a Matrix.
     */
    private Matrix applyActivation(Matrix m) {
        return switch (activationType) {
            case RELU -> m.relu();
            case GELU -> m.gelu();
        };
    }

    @Override
    public String toString() {
        return String.format("FeedForward(%d → %d → %d, activation=%s)",
            linear1.inFeatures, linear1.outFeatures, linear2.outFeatures, activationType);
    }
}
