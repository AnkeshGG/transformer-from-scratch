/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import transformer.TransformerConfig;
import transformer.attention.AttentionMask;
import transformer.attention.MultiHeadAttention;
import transformer.attention.ScaledDotProductAttention;
import transformer.tensor.Matrix;
import transformer.tensor.Tensor3D;
import transformer.utils.RandomUtils;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * AttentionTest — Unit tests for attention mechanisms.
 */
@DisplayName("Attention Mechanisms")
public class AttentionTest {

    private static final float TOL = 1e-4f;

    // ── Causal mask tests ─────────────────────────────────────────────────────

    @Test
    @DisplayName("causalMask: lower triangle is 0, upper triangle is -inf")
    void testCausalMask() {
        float[][] mask = AttentionMask.causalMask(4);
        assertEquals(4, mask.length);
        assertEquals(4, mask[0].length);

        // Lower triangle + diagonal = 0
        assertEquals(0.0f, mask[0][0], TOL);
        assertEquals(0.0f, mask[1][0], TOL);
        assertEquals(0.0f, mask[1][1], TOL);
        assertEquals(0.0f, mask[3][3], TOL);

        // Upper triangle = -inf
        assertEquals(Float.NEGATIVE_INFINITY, mask[0][1]);
        assertEquals(Float.NEGATIVE_INFINITY, mask[0][2]);
        assertEquals(Float.NEGATIVE_INFINITY, mask[0][3]);
        assertEquals(Float.NEGATIVE_INFINITY, mask[2][3]);
    }

    @Test
    @DisplayName("causalMask: size 1 = all zeros (trivial case)")
    void testCausalMaskSizeOne() {
        float[][] mask = AttentionMask.causalMask(1);
        assertEquals(0.0f, mask[0][0], TOL);
    }

    @Test
    @DisplayName("paddingMask: pads correct positions")
    void testPaddingMask() {
        int[] tokenIds = {3, 4, 0, 0};  // last 2 are PAD (ID=0)
        float[][] mask = AttentionMask.paddingMask(tokenIds, 0);
        // Columns j=2,3 are PAD → all rows should have -inf at j=2,3
        assertEquals(0.0f, mask[0][0], TOL);
        assertEquals(0.0f, mask[0][1], TOL);
        assertEquals(Float.NEGATIVE_INFINITY, mask[0][2]);
        assertEquals(Float.NEGATIVE_INFINITY, mask[0][3]);
        assertEquals(Float.NEGATIVE_INFINITY, mask[2][2]);
    }

    // ── Scaled dot-product attention ──────────────────────────────────────────

    @Test
    @DisplayName("ScaledDotProductAttention: output has correct shape")
    void testSdpaShape() {
        // Q: [3 × 4], K: [5 × 4], V: [5 × 6] → output: [3 × 6]
        Matrix Q = Matrix.filled(3, 4, 0.1f);
        Matrix K = Matrix.filled(5, 4, 0.1f);
        Matrix V = Matrix.filled(5, 6, 0.1f);
        Matrix out = ScaledDotProductAttention.compute(Q, K, V, null);
        assertEquals(3, out.rows, "Output rows should equal Q.rows (S_q=3)");
        assertEquals(6, out.cols, "Output cols should equal V.cols (d_v=6)");
    }

    @Test
    @DisplayName("ScaledDotProductAttention: attention weights sum to 1 per query")
    void testSdpaWeightsSumToOne() {
        Matrix Q = Matrix.filled(3, 4, 1.0f);
        Matrix K = Matrix.filled(3, 4, 1.0f);
        Matrix V = Matrix.filled(3, 4, 1.0f);

        ScaledDotProductAttention.AttentionResult result =
            ScaledDotProductAttention.computeWithWeights(Q, K, V, null);

        // Each row of attention weights should sum to ~1.0
        for (int r = 0; r < result.weights().rows; r++) {
            float sum = 0;
            for (int c = 0; c < result.weights().cols; c++)
                sum += result.weights().data[r][c];
            assertEquals(1.0f, sum, 1e-5f, "Attention weights row " + r + " should sum to 1");
        }
    }

    @Test
    @DisplayName("ScaledDotProductAttention: causal mask blocks future positions")
    void testSdpaCausalMask() {
        int seqLen = 4, dk = 4;
        // Use large identical Q and K so raw scores are large positive (≈ 1/sqrt(4)=0.5 each)
        // With causal mask, position 0 should only attend to position 0.
        Matrix Q = Matrix.filled(seqLen, dk, 1.0f);
        Matrix K = Matrix.filled(seqLen, dk, 1.0f);
        Matrix V = Matrix.identity(seqLen);  // identity: attending to position j gives row j

        float[][] causalMask = AttentionMask.causalMask(seqLen);
        ScaledDotProductAttention.AttentionResult result =
            ScaledDotProductAttention.computeWithWeights(Q, K, V, causalMask);

        // Row 0 of attention weights: position 0 should attend only to j=0 (j≥1 blocked)
        assertEquals(1.0f, result.weights().data[0][0], 1e-4f,
            "Position 0 should attend 100% to position 0 (causal mask)");
        for (int j = 1; j < seqLen; j++) {
            assertEquals(0.0f, result.weights().data[0][j], 1e-5f,
                "Position 0 should not attend to future position " + j);
        }
    }

    @Test
    @DisplayName("ScaledDotProductAttention: dimension mismatch throws")
    void testSdpaDimensionMismatch() {
        Matrix Q = Matrix.filled(3, 4, 0f);  // d_k=4
        Matrix K = Matrix.filled(3, 5, 0f);  // d_k=5 ← mismatch
        Matrix V = Matrix.filled(3, 4, 0f);
        assertThrows(IllegalArgumentException.class,
            () -> ScaledDotProductAttention.compute(Q, K, V, null));
    }

    // ── Multi-Head Attention ──────────────────────────────────────────────────

    @Test
    @DisplayName("MultiHeadAttention: output shape equals input shape")
    void testMultiHeadAttentionShape() {
        TransformerConfig config = TransformerConfig.tiny(20);
        // dModel=8, numHeads=2, dHead=4
        Random rng = RandomUtils.createRng(42);
        MultiHeadAttention mha = new MultiHeadAttention(config, rng);

        // Input: [1 × 5 × 8]
        Tensor3D x = Tensor3D.filled(1, 5, config.dModel, 0.5f);
        Tensor3D out = mha.selfAttention(x, null);

        assertEquals(1, out.batch);
        assertEquals(5, out.rows);
        assertEquals(config.dModel, out.cols);
    }

    @Test
    @DisplayName("MultiHeadAttention: cross-attention with different Q/KV lengths")
    void testCrossAttentionShape() {
        TransformerConfig config = TransformerConfig.tiny(20);
        Random rng = RandomUtils.createRng(42);
        MultiHeadAttention mha = new MultiHeadAttention(config, rng);

        // Decoder query: [1 × 4 × 8] (T=4)
        // Encoder key/value: [1 × 6 × 8] (S=6)
        Tensor3D decoderQ  = Tensor3D.filled(1, 4, config.dModel, 0.3f);
        Tensor3D encoderKV = Tensor3D.filled(1, 6, config.dModel, 0.3f);
        Tensor3D out = mha.crossAttention(decoderQ, encoderKV, null);

        assertEquals(1, out.batch, "batch should be 1");
        assertEquals(4, out.rows,  "output rows should be decoder length (T=4)");
        assertEquals(config.dModel, out.cols, "output cols should be dModel");
    }

    @Test
    @DisplayName("MultiHeadAttention: dModel not divisible by numHeads throws")
    void testMhaDimensionCheck() {
        // Manually create a bad config using builder (bypass tiny factory).
        assertThrows(IllegalArgumentException.class, () ->
            new TransformerConfig.Builder()
                .dModel(9)    // 9 not divisible by 2
                .numHeads(2)
                .dFF(16)
                .numEncoderLayers(1)
                .numDecoderLayers(1)
                .vocabSize(10)
                .build()
        );
    }
}
