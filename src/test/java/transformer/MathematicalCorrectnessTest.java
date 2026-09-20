/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import transformer.attention.AttentionMask;
import transformer.attention.ScaledDotProductAttention;
import transformer.embedding.PositionalEncoding;
import transformer.layers.FeedForward;
import transformer.layers.LayerNorm;
import transformer.layers.Linear;
import transformer.loss.CrossEntropyLoss;
import transformer.tensor.Matrix;
import transformer.tensor.Tensor3D;
import transformer.tokenizer.SimpleTokenizer;
import transformer.utils.MathUtils;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MathematicalCorrectnessTest — verifies specific, manually-computed values for
 * every layer and utility in the library.
 *
 * <p>These tests complement the API-level tests in {@link MatrixTest},
 * {@link Tensor3DTest}, {@link AttentionTest}, and {@link ComponentTest} by
 * checking exact numerical results rather than just output shapes.
 */
@DisplayName("Mathematical Correctness")
public class MathematicalCorrectnessTest {

    private static final float TOL   = 1e-4f;
    private static final float LOOSE = 1e-3f;

    // ── 1. Matrix arithmetic ──────────────────────────────────────────────────

    @Test
    @DisplayName("matmul [2×3]×[3×2]: exact values")
    void matmulExactValues() {
        Matrix A = new Matrix(new float[][]{{1,2,3},{4,5,6}});
        Matrix B = new Matrix(new float[][]{{7,8},{9,10},{11,12}});
        Matrix C = A.matmul(B);
        assertAll(
            () -> assertEquals(58f,  C.data[0][0], TOL),
            () -> assertEquals(64f,  C.data[0][1], TOL),
            () -> assertEquals(139f, C.data[1][0], TOL),
            () -> assertEquals(154f, C.data[1][1], TOL)
        );
    }

    @Test
    @DisplayName("addBias: bias added row-wise")
    void addBiasCorrect() {
        Matrix A = new Matrix(new float[][]{{1,2},{3,4}});
        Matrix R = A.addBias(new float[]{10, 20});
        assertAll(
            () -> assertEquals(11f, R.data[0][0], TOL),
            () -> assertEquals(22f, R.data[0][1], TOL),
            () -> assertEquals(13f, R.data[1][0], TOL),
            () -> assertEquals(24f, R.data[1][1], TOL)
        );
    }

    @Test
    @DisplayName("rowMean and rowVariance: known statistical values")
    void rowStatistics() {
        // Population [2,4,4,4,5,5,7,9] has mean=5, variance=4
        Matrix A = new Matrix(new float[][]{{2,4,4,4,5,5,7,9}});
        assertEquals(5.0, A.rowMean(0),               1e-5);
        assertEquals(4.0, A.rowVariance(0, 5.0), 1e-5);
    }

    @Test
    @DisplayName("split → concat is identity")
    void splitConcatRoundTrip() {
        Tensor3D t = new Tensor3D(new float[][][]{{{1,2,3,4,5,6,7,8}}});
        Tensor3D rt = Tensor3D.concatLastDim(t.splitLastDim(4));
        for (int c = 0; c < 8; c++)
            assertEquals(t.get(0, 0, c), rt.get(0, 0, c), TOL, "mismatch at col " + c);
    }

    // ── 2. Softmax ────────────────────────────────────────────────────────────

    @Test
    @DisplayName("softmax([0,1,2]): exact output values")
    void softmaxExact() {
        float[] probs = MathUtils.softmax(new float[]{0f, 1f, 2f});
        assertEquals(0.0900f, probs[0], LOOSE);
        assertEquals(0.6652f, probs[2], LOOSE);
    }

    @Test
    @DisplayName("softmax([1000,1000,1000]): numerically stable → uniform")
    void softmaxLargeInputStable() {
        float[] probs = MathUtils.softmax(new float[]{1000f, 1000f, 1000f});
        for (float p : probs) {
            assertFalse(Float.isNaN(p));
            assertEquals(1f/3, p, TOL);
        }
    }

    @Test
    @DisplayName("log-softmax: exp(logSoftmax) recovers softmax")
    void logSoftmaxConsistent() {
        float[] x    = {0f, 1f, 2f};
        float[] soft = MathUtils.softmax(x);
        float[] logS = MathUtils.logSoftmax(x);
        for (int i = 0; i < x.length; i++) {
            assertTrue(logS[i] <= 0f);
            assertEquals(soft[i], (float) Math.exp(logS[i]), TOL);
        }
    }

    // ── 3. Layer Normalization ────────────────────────────────────────────────

    @Test
    @DisplayName("LayerNorm([1,2,3,4]): output[0] matches formula")
    void layerNormFormula() {
        // mean=2.5, var=1.25; xhat[0]=(1−2.5)/√(1.25+ε); γ=1, β=0
        LayerNorm ln = new LayerNorm(4, 1e-8);
        Matrix out = ln.forward(new Matrix(new float[][]{{1f,2f,3f,4f}}));
        double expected = (1.0 - 2.5) / Math.sqrt(1.25 + 1e-8);
        assertEquals((float) expected, out.data[0][0], TOL);
    }

    @Test
    @DisplayName("LayerNorm: output mean ≈ 0 and variance ≈ 1")
    void layerNormStatistics() {
        LayerNorm ln = new LayerNorm(4, 1e-8);
        Matrix out   = ln.forward(new Matrix(new float[][]{{1f,2f,3f,4f}}));
        double mean  = out.rowMean(0);
        double var   = out.rowVariance(0, mean);
        assertEquals(0.0, mean, 1e-5);
        assertEquals(1.0, var,  TOL);
    }

    @Test
    @DisplayName("LayerNorm: constant input → zero output")
    void layerNormConstant() {
        LayerNorm ln = new LayerNorm(4, 1e-6);
        Matrix out = ln.forward(new Matrix(new float[][]{{5f,5f,5f,5f}}));
        for (float v : out.data[0]) assertEquals(0f, v, LOOSE);
    }

    // ── 4. Positional Encoding ────────────────────────────────────────────────

    @Test
    @DisplayName("PE(0,0)=sin(0)=0, PE(0,1)=cos(0)=1")
    void pePosition0() {
        PositionalEncoding pe = new PositionalEncoding(TransformerConfig.tiny(10));
        assertEquals(0f, pe.getValue(0, 0), 1e-6f);
        assertEquals(1f, pe.getValue(0, 1), 1e-6f);
    }

    @Test
    @DisplayName("PE(1,0)=sin(1)≈0.8415, PE(1,1)=cos(1)≈0.5403")
    void pePosition1() {
        PositionalEncoding pe = new PositionalEncoding(TransformerConfig.tiny(10));
        assertEquals(0.8415f, pe.getValue(1, 0), TOL);
        assertEquals(0.5403f, pe.getValue(1, 1), TOL);
    }

    @Test
    @DisplayName("PE(1,2) matches formula sin(pos / 10000^(2i/dModel))")
    void peDimFormula() {
        // dModel=8, i=1 → divTerm = 10000^(2/8) = 10000^0.25
        TransformerConfig cfg = TransformerConfig.tiny(10);
        PositionalEncoding pe = new PositionalEncoding(cfg);
        double divTerm = Math.pow(10000, 2.0 / cfg.dModel);
        assertEquals((float) Math.sin(1.0 / divTerm), pe.getValue(1, 2), 1e-5f);
    }

    @Test
    @DisplayName("PE values are bounded in [-1, 1]")
    void peBounded() {
        PositionalEncoding pe = new PositionalEncoding(TransformerConfig.tiny(10));
        for (int pos = 0; pos < 16; pos++)
            for (int d = 0; d < 8; d++)
                assertTrue(pe.getValue(pos, d) >= -1.001f && pe.getValue(pos, d) <= 1.001f,
                    "PE(" + pos + "," + d + ") out of [-1,1]");
    }

    // ── 5. Attention masks ────────────────────────────────────────────────────

    @Test
    @DisplayName("causalMask: correct structure")
    void causalMaskStructure() {
        float[][] m = AttentionMask.causalMask(4);
        // diagonal and lower triangle = 0
        assertEquals(0f, m[0][0], TOL);
        assertEquals(0f, m[3][0], TOL);
        assertEquals(0f, m[3][3], TOL);
        // upper triangle = -∞
        assertEquals(Float.NEGATIVE_INFINITY, m[0][1]);
        assertEquals(Float.NEGATIVE_INFINITY, m[1][3]);
        // row 2 has exactly 3 zeros
        int zeros = 0;
        for (float v : m[2]) if (v == 0f) zeros++;
        assertEquals(3, zeros);
    }

    @Test
    @DisplayName("paddingMask: PAD columns get -inf")
    void paddingMaskCorrect() {
        float[][] m = AttentionMask.paddingMask(new int[]{4, 5, 0, 0}, 0);
        assertEquals(0f,                    m[0][0]);
        assertEquals(Float.NEGATIVE_INFINITY, m[0][2]);
        assertEquals(Float.NEGATIVE_INFINITY, m[3][3]);
    }

    // ── 6. Scaled Dot-Product Attention ──────────────────────────────────────

    @Test
    @DisplayName("SDPA weights sum to 1 per query row")
    void sdpaWeightsSumToOne() {
        Matrix Q = Matrix.filled(3, 4, 1f);
        Matrix K = Matrix.filled(3, 4, 1f);
        Matrix V = Matrix.filled(3, 4, 1f);
        ScaledDotProductAttention.AttentionResult r =
            ScaledDotProductAttention.computeWithWeights(Q, K, V, null);
        for (int row = 0; row < 3; row++) {
            float sum = 0; for (float w : r.weights().data[row]) sum += w;
            assertEquals(1f, sum, 1e-5f);
        }
    }

    @Test
    @DisplayName("SDPA + causal mask: pos0 attends only to pos0")
    void sdpaCausalBlocksFuture() {
        int S = 4, dk = 4;
        // Strong self-affinity via identity-like Q and K
        Matrix Q = Matrix.filled(S, dk, 0f); Matrix K = Matrix.filled(S, dk, 0f);
        Matrix V = Matrix.identity(S);
        for (int i = 0; i < S; i++) { Q.set(i, i, 100f); K.set(i, i, 100f); }

        ScaledDotProductAttention.AttentionResult r =
            ScaledDotProductAttention.computeWithWeights(Q, K, V, AttentionMask.causalMask(S));
        assertEquals(1f, r.weights().data[0][0], TOL, "pos0 should attend 100% to itself");
        assertEquals(0f, r.weights().data[0][1], TOL, "pos0 must not attend to future pos1");
    }

    @Test
    @DisplayName("SDPA + causal mask vs no mask: uniform Q/K shows clear difference")
    void sdpaCausalVsNone() {
        int S = 4, dk = 4;
        Matrix Q = Matrix.filled(S, dk, 1f);
        Matrix K = Matrix.filled(S, dk, 1f);
        Matrix V = Matrix.identity(S);
        float[][] cm = AttentionMask.causalMask(S);

        float withMask    = ScaledDotProductAttention.computeWithWeights(Q, K, V, cm)
                               .weights().data[0][1];
        float withoutMask = ScaledDotProductAttention.computeWithWeights(Q, K, V, null)
                               .weights().data[0][1];

        assertEquals(0f, withMask,    1e-6f, "Masked: pos0→pos1 weight must be 0");
        assertTrue(withoutMask > 0.1f,       "Unmasked: pos0→pos1 weight must be positive (~0.25)");
    }

    // ── 7. Linear layer ───────────────────────────────────────────────────────

    @Test
    @DisplayName("Linear: y = xWᵀ + b with manually-set weights")
    void linearForwardExact() {
        Linear lin = new Linear(3, 2, new Random(0));
        // weight and bias arrays are final references; assign elements in-place
        lin.weight[0][0] = 1f; lin.weight[0][1] = 0f; lin.weight[0][2] = 0f;
        lin.weight[1][0] = 0f; lin.weight[1][1] = 1f; lin.weight[1][2] = 0f;
        lin.bias[0] = 10f; lin.bias[1] = 20f;

        Matrix y = lin.forward(new Matrix(new float[][]{{1f, 2f, 3f}}));
        assertEquals(11f, y.data[0][0], TOL);
        assertEquals(22f, y.data[0][1], TOL);
    }

    // ── 8. FeedForward ────────────────────────────────────────────────────────

    @Test
    @DisplayName("FeedForward: output shape preserves [S × dModel]")
    void ffnShapePreserved() {
        FeedForward ffn = new FeedForward(8, 16, new Random(0));
        Matrix out = ffn.forward(new Matrix(5, 8));
        assertEquals(5, out.rows);
        assertEquals(8, out.cols);
    }

    @Test
    @DisplayName("FeedForward: no NaN or Inf in output")
    void ffnNoNaN() {
        FeedForward ffn = new FeedForward(8, 16, new Random(42));
        Matrix out = ffn.forward(new Matrix(new float[][]{{1,-1,2,-2,0.5f,-0.5f,3,-3}}));
        for (float v : out.data[0])
            assertAll(
                () -> assertFalse(Float.isNaN(v)),
                () -> assertFalse(Float.isInfinite(v))
            );
    }

    // ── 9. Cross-entropy loss ─────────────────────────────────────────────────

    @Test
    @DisplayName("CE loss: perfect prediction → loss ≈ 0")
    void cePerfectPrediction() {
        CrossEntropyLoss ce = new CrossEntropyLoss(0);
        float loss = ce.compute(
            new float[][][]{{{-100f, 100f, -100f}, {-100f, -100f, 100f}}},
            new int[][]{{1, 2}});
        assertTrue(loss < 0.01f);
    }

    @Test
    @DisplayName("CE loss: uniform logits over V=3 → log(3)")
    void ceUniformLogits() {
        CrossEntropyLoss ce = new CrossEntropyLoss(0);
        float loss = ce.compute(new float[][][]{{{0f, 0f, 0f}}}, new int[][]{{1}});
        assertEquals((float) Math.log(3), loss, TOL);
    }

    @Test
    @DisplayName("CE loss: PAD target is excluded")
    void cePaddingExcluded() {
        CrossEntropyLoss ce = new CrossEntropyLoss(0);
        // Position 0 has target=PAD(0) → skipped; only position 1 contributes
        float loss = ce.compute(new float[][][]{{{0f,0f,0f},{0f,0f,0f}}},
                                new int[][]{{0, 1}});
        assertEquals((float) Math.log(3), loss, TOL);
    }

    @Test
    @DisplayName("perplexity = exp(CE)")
    void perplexityFormula() {
        float loss = (float) Math.log(3);
        assertEquals(3.0, CrossEntropyLoss.perplexity(loss), 0.01);
    }

    // ── 10. Tokenizer ─────────────────────────────────────────────────────────

    @Test
    @DisplayName("SimpleTokenizer: special token IDs are 0–3")
    void tokenizerSpecialIds() {
        SimpleTokenizer tok = new SimpleTokenizer(List.of("hello", "world"));
        assertAll(
            () -> assertEquals(0, SimpleTokenizer.PAD_ID),
            () -> assertEquals(1, SimpleTokenizer.BOS_ID),
            () -> assertEquals(2, SimpleTokenizer.EOS_ID),
            () -> assertEquals(3, SimpleTokenizer.UNK_ID)
        );
    }

    @Test
    @DisplayName("SimpleTokenizer: vocabulary words start at ID 4")
    void tokenizerVocabIds() {
        SimpleTokenizer tok = new SimpleTokenizer(List.of("hello", "world", "cats"));
        assertEquals(4, tok.wordToId("hello"));
        assertEquals(5, tok.wordToId("world"));
        assertEquals(6, tok.wordToId("cats"));
        assertEquals(SimpleTokenizer.UNK_ID, tok.wordToId("unknown"));
    }

    @Test
    @DisplayName("SimpleTokenizer: encode adds BOS/EOS; decode round-trips")
    void tokenizerEncodeDecodeRoundTrip() {
        SimpleTokenizer tok = new SimpleTokenizer(List.of("hello", "world"));
        java.util.List<Integer> ids = tok.encode("hello world");
        assertEquals(SimpleTokenizer.BOS_ID, ids.get(0));
        assertEquals(4, ids.get(1));
        assertEquals(5, ids.get(2));
        assertEquals(SimpleTokenizer.EOS_ID, ids.get(ids.size() - 1));

        String decoded = tok.decode(ids);
        assertTrue(decoded.contains("hello"));
        assertTrue(decoded.contains("world"));
    }

    // ── 11. Full Transformer ──────────────────────────────────────────────────

    @Test
    @DisplayName("Transformer forward: output probabilities sum to 1.0 at each position")
    void transformerProbabilitiesNormalised() {
        TransformerConfig cfg = new TransformerConfig.Builder()
            .dModel(8).numHeads(2).dFF(16)
            .numEncoderLayers(2).numDecoderLayers(2)
            .vocabSize(10).maxSequenceLength(16)
            .randomSeed(42).debugMode(false).build();

        Transformer model = new Transformer(cfg, 0, 1, 2);
        Tensor3D probs = model.forward(new int[][]{{3,4,5}}, new int[][]{{1,3,4,5}});

        for (int t = 0; t < probs.rows; t++) {
            float sum = 0; for (float p : probs.data[0][t]) sum += p;
            assertEquals(1.0f, sum, TOL, "Position " + t + " probabilities must sum to 1");
        }
    }

    @Test
    @DisplayName("Transformer forward: no NaN or Inf values")
    void transformerNoNaN() {
        TransformerConfig cfg = new TransformerConfig.Builder()
            .dModel(8).numHeads(2).dFF(16)
            .numEncoderLayers(2).numDecoderLayers(2)
            .vocabSize(10).maxSequenceLength(16)
            .randomSeed(42).debugMode(false).build();

        Transformer model = new Transformer(cfg, 0, 1, 2);
        Tensor3D probs = model.forward(new int[][]{{3,4,5}}, new int[][]{{1,3,4}});

        for (int b = 0; b < probs.batch; b++)
            for (int t = 0; t < probs.rows; t++)
                for (int v = 0; v < probs.cols; v++) {
                    float p = probs.get(b, t, v);
                    assertFalse(Float.isNaN(p),      "NaN at ["+b+","+t+","+v+"]");
                    assertFalse(Float.isInfinite(p), "Inf at ["+b+","+t+","+v+"]");
                }
    }

    @Test
    @DisplayName("Transformer generate: returns valid token IDs within vocab")
    void transformerGenerateValid() {
        TransformerConfig cfg = new TransformerConfig.Builder()
            .dModel(8).numHeads(2).dFF(16)
            .numEncoderLayers(2).numDecoderLayers(2)
            .vocabSize(10).maxSequenceLength(16)
            .randomSeed(42).debugMode(false).build();

        Transformer model = new Transformer(cfg, 0, 1, 2);
        int[] out = model.generate(new int[]{3, 4, 5}, 8);

        assertNotNull(out);
        assertTrue(out.length > 0);
        for (int id : out)
            assertTrue(id >= 0 && id < 10, "Generated ID " + id + " out of vocab range");
    }

    // ── 12. Dimension error detection ─────────────────────────────────────────

    @Test
    @DisplayName("matmul: incompatible shapes throw IllegalArgumentException")
    void matmulMismatch() {
        assertThrows(IllegalArgumentException.class, () ->
            new Matrix(new float[][]{{1,2}}).matmul(new Matrix(new float[][]{{1,2,3}})));
    }

    @Test
    @DisplayName("Tensor3D add: mismatched shapes throw IllegalArgumentException")
    void tensor3dAddMismatch() {
        assertThrows(IllegalArgumentException.class, () ->
            new Tensor3D(1,2,3).add(new Tensor3D(1,3,2)));
    }

    @Test
    @DisplayName("TransformerConfig: dModel not divisible by numHeads throws")
    void configHeadsDivisibility() {
        assertThrows(IllegalArgumentException.class, () ->
            new TransformerConfig.Builder()
                .dModel(9).numHeads(2).dFF(16)
                .numEncoderLayers(1).numDecoderLayers(1).vocabSize(10)
                .build());
    }

    @Test
    @DisplayName("ScaledDotProductAttention: Q/K dk mismatch throws")
    void sdpaDkMismatch() {
        assertThrows(IllegalArgumentException.class, () ->
            ScaledDotProductAttention.compute(
                Matrix.filled(2, 4, 0f),
                Matrix.filled(2, 5, 0f),   // dk=5 ≠ Q's dk=4
                Matrix.filled(2, 4, 0f), null));
    }

    @Test
    @DisplayName("LayerNorm: input feature size mismatch throws")
    void layerNormMismatch() {
        assertThrows(IllegalArgumentException.class, () ->
            new LayerNorm(4, 1e-6).forward(new Matrix(3, 5)));
    }

    @Test
    @DisplayName("Tensor3D.splitLastDim: non-divisible size throws")
    void splitNonDivisible() {
        assertThrows(IllegalArgumentException.class, () ->
            new Tensor3D(1, 2, 5).splitLastDim(3));
    }
}
