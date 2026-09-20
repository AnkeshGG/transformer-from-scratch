/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.layers;

import transformer.tensor.Matrix;
import transformer.tensor.Tensor3D;
import transformer.utils.RandomUtils;

import java.util.Random;

/**
 * Linear — Fully connected linear (affine) transformation layer.
 *
 * <h2>Mathematical operation</h2>
 * <pre>
 *   y = x · W^T + b
 *
 *   where:
 *     x ∈ R^{inFeatures}   — input vector
 *     W ∈ R^{outFeatures × inFeatures}  — weight matrix
 *     b ∈ R^{outFeatures}  — bias vector
 *     y ∈ R^{outFeatures}  — output vector
 * </pre>
 *
 * <p>In batched / sequence context:
 * <pre>
 *   Input:  [B × S × inFeatures]   → (flatten to [B·S × inFeatures])
 *   Output: [B × S × outFeatures]  → (reshape back to 3D)
 * </pre>
 *
 * <h2>Notation note</h2>
 * <p>We store weights as [outFeatures × inFeatures] and multiply x · W^T
 * so that each row of W is the weight vector for one output neuron.
 * This matches the convention used in PyTorch nn.Linear.
 *
 * <h2>Initialisation</h2>
 * <p>Weights are initialised with Xavier uniform:
 * <pre>
 *   W[i][j] ~ Uniform(−limit, +limit)
 *   limit = sqrt(6 / (inFeatures + outFeatures))
 * </pre>
 * Biases are initialised to zero (standard practice).
 */
public final class Linear {

    /**
     * Weight matrix W of shape [outFeatures × inFeatures].
     * Public for inspection; normally would be package-private.
     */
    public final float[][] weight;

    /**
     * Bias vector b of length outFeatures.
     * Public for inspection.
     */
    public final float[] bias;

    /** Input dimension. */
    public final int inFeatures;

    /** Output dimension. */
    public final int outFeatures;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct a Linear layer.
     *
     * @param inFeatures  dimension of the input
     * @param outFeatures dimension of the output
     * @param rng         seeded random generator for reproducibility
     */
    public Linear(int inFeatures, int outFeatures, Random rng) {
        this.inFeatures  = inFeatures;
        this.outFeatures = outFeatures;
        this.weight      = new float[outFeatures][inFeatures];
        this.bias        = new float[outFeatures];

        // Xavier uniform weight initialisation.
        RandomUtils.xavierUniform(weight, inFeatures, outFeatures, rng);
        // Bias initialised to zero.
        RandomUtils.zeroInit(bias);
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Apply the linear transformation to a 2-D Matrix.
     *
     * <p>Operation:
     * <pre>
     *   Input:  Matrix [S × inFeatures]     (S = sequence length or any batch dimension)
     *   Output: Matrix [S × outFeatures]
     *
     *   output[s][j] = Σ_k input[s][k] × weight[j][k]  +  bias[j]
     *                = (input · W^T)[s][j] + bias[j]
     * </pre>
     *
     * @param input 2-D matrix of shape [S × inFeatures]
     * @return output matrix of shape [S × outFeatures]
     */
    public Matrix forward(Matrix input) {
        if (input.cols != inFeatures)
            throw new IllegalArgumentException(
                "Linear forward: input cols (" + input.cols + ") ≠ inFeatures (" + inFeatures + ")\n" +
                "  input shape = " + input.shapeString());

        // Represent W as a Matrix for matmul convenience
        // output = input × W^T  →  [S × inFeatures] × [inFeatures × outFeatures] = [S × outFeatures]
        Matrix W = new Matrix(weight);
        Matrix WT = W.transpose();       // [inFeatures × outFeatures] ... actually W is [out × in], so WT is [in × out]

        // Hmm: W is [out × in], so W^T is [in × out].
        // input is [S × in].
        // input × W^T = [S × in] × [in × out] = [S × out]. ✓
        Matrix output = input.matmul(WT);         // [S × outFeatures]
        return output.addBias(bias);              // add bias vector to every row
    }

    /**
     * Apply the linear transformation to a 3-D Tensor3D by processing each
     * batch slice independently.
     *
     * <p>Operation:
     * <pre>
     *   Input:  Tensor3D [B × S × inFeatures]
     *   Output: Tensor3D [B × S × outFeatures]
     * </pre>
     *
     * @param input 3-D tensor of shape [B × S × inFeatures]
     * @return output tensor of shape [B × S × outFeatures]
     */
    public Tensor3D forward(Tensor3D input) {
        if (input.cols != inFeatures)
            throw new IllegalArgumentException(
                "Linear forward (3D): input.cols (" + input.cols + ") ≠ inFeatures (" + inFeatures + ")\n" +
                "  input shape = " + input.shapeString());

        Tensor3D output = new Tensor3D(input.batch, input.rows, outFeatures);
        for (int b = 0; b < input.batch; b++) {
            Matrix sliceOut = forward(input.getMatrix(b));  // [S × outFeatures]
            output.setMatrix(b, sliceOut);
        }
        return output;
    }

    @Override
    public String toString() {
        return String.format("Linear(%d → %d)", inFeatures, outFeatures);
    }
}
