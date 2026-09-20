/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.layers;

import transformer.tensor.Matrix;
import transformer.tensor.Tensor3D;
import transformer.utils.RandomUtils;

/**
 * LayerNorm — Layer Normalisation as defined in Ba et al., 2016.
 *
 * <h2>What it does</h2>
 * <p>LayerNorm normalises the activations across the feature (embedding) dimension
 * for each token position independently. Unlike BatchNorm (which normalises
 * across the batch), LayerNorm normalises across features within a single example.
 *
 * <h2>Mathematical operation</h2>
 * <p>For a single token representation x ∈ R^{dModel}:
 * <pre>
 *   μ = (1/dModel) Σ_i x_i                    (mean across features)
 *   σ² = (1/dModel) Σ_i (x_i − μ)²           (variance across features)
 *   x̂_i = (x_i − μ) / sqrt(σ² + ε)           (normalise)
 *   y_i  = γ_i · x̂_i + β_i                   (scale and shift)
 * </pre>
 *
 * <p>γ (gamma) and β (beta) are learnable parameters of shape [dModel].
 * They are initialised to γ = 1, β = 0 so that initially LayerNorm
 * is approximately the identity function.
 *
 * <h2>Post-LN architecture (original paper)</h2>
 * <p>In the original "Attention Is All You Need" paper, LayerNorm is applied
 * <em>after</em> the residual connection:
 * <pre>
 *   output = LayerNorm(x + Sublayer(x))
 * </pre>
 * This is called "Post-LN". Modern architectures often use "Pre-LN"
 * (LayerNorm applied before the sublayer), but this implementation
 * follows the original paper.
 *
 * <h2>Dimensions</h2>
 * <pre>
 *   Input:  [B × S × dModel]   — batch of sequences
 *   Output: [B × S × dModel]   — same shape, each token normalised independently
 * </pre>
 *
 * <h2>Numerical stability</h2>
 * <p>A small ε (epsilon) is added to the variance before taking the square root
 * to prevent division by zero when the variance is very small.
 */
public final class LayerNorm {

    /** Scale parameter: γ (gamma), shape [dModel]. Initialised to 1.0. */
    public final float[] gamma;

    /** Shift parameter: β (beta), shape [dModel]. Initialised to 0.0. */
    public final float[] beta;

    /** Feature dimension (dModel). */
    private final int dModel;

    /**
     * Small constant added to variance for numerical stability.
     * Typical values: 1e-5 to 1e-6.
     */
    private final double epsilon;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct a LayerNorm for a given feature dimension.
     *
     * @param dModel  normalisation dimension (number of features)
     * @param epsilon small value for numerical stability
     */
    public LayerNorm(int dModel, double epsilon) {
        this.dModel   = dModel;
        this.epsilon  = epsilon;
        this.gamma    = new float[dModel];
        this.beta     = new float[dModel];
        // γ = 1, β = 0 → initially acts as identity.
        RandomUtils.oneInit(gamma);
        RandomUtils.zeroInit(beta);
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Apply Layer Normalisation to a 2-D Matrix.
     *
     * <p>Each row (token) is normalised independently across the cols (features).
     *
     * @param input Matrix of shape [S × dModel]
     * @return normalised Matrix of same shape
     */
    public Matrix forward(Matrix input) {
        if (input.cols != dModel)
            throw new IllegalArgumentException(
                "LayerNorm: input.cols (" + input.cols + ") ≠ dModel (" + dModel + ")");

        float[][] result = new float[input.rows][dModel];

        for (int r = 0; r < input.rows; r++) {
            // Step 1: compute mean across features for this token.
            double mean = input.rowMean(r);

            // Step 2: compute variance across features for this token.
            double variance = input.rowVariance(r, mean);

            // Step 3: normalise, scale, and shift.
            double invStd = 1.0 / Math.sqrt(variance + epsilon);
            for (int d = 0; d < dModel; d++) {
                double xHat = (input.data[r][d] - mean) * invStd;
                result[r][d] = (float) (gamma[d] * xHat + beta[d]);
            }
        }
        return new Matrix(result);
    }

    /**
     * Apply Layer Normalisation to a 3-D Tensor3D.
     *
     * <p>Each token at each batch index is normalised independently.
     *
     * @param input Tensor3D of shape [B × S × dModel]
     * @return normalised Tensor3D of same shape
     */
    public Tensor3D forward(Tensor3D input) {
        if (input.cols != dModel)
            throw new IllegalArgumentException(
                "LayerNorm (3D): input.cols (" + input.cols + ") ≠ dModel (" + dModel + ")");

        Tensor3D output = new Tensor3D(input.batch, input.rows, input.cols);
        for (int b = 0; b < input.batch; b++) {
            output.setMatrix(b, forward(input.getMatrix(b)));
        }
        return output;
    }

    @Override
    public String toString() {
        return String.format("LayerNorm(%d, eps=%.2e)", dModel, epsilon);
    }
}
