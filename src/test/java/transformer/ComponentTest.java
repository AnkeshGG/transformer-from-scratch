/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import transformer.TransformerConfig;
import transformer.decoder.Decoder;
import transformer.decoder.DecoderLayer;
import transformer.embedding.PositionalEncoding;
import transformer.encoder.Encoder;
import transformer.encoder.EncoderLayer;
import transformer.layers.FeedForward;
import transformer.layers.LayerNorm;
import transformer.tensor.Matrix;
import transformer.tensor.Tensor3D;
import transformer.utils.RandomUtils;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ComponentTest — Integration tests for high-level Transformer components.
 *
 * <p>Each test uses the tiny config: dModel=8, numHeads=2, dFF=16, numLayers=2.
 * This is fast to run while still exercising all the mathematical operations.
 */
@DisplayName("Transformer Components")
public class ComponentTest {

    private static final float TOL = 1e-4f;

    private TransformerConfig tinyConfig() {
        return TransformerConfig.tiny(20);
    }

    // ── Positional Encoding ───────────────────────────────────────────────────

    @Test
    @DisplayName("PositionalEncoding: PE(0, even) = sin(0) = 0")
    void testPEPosition0EvenDim() {
        TransformerConfig config = tinyConfig();
        PositionalEncoding pe = new PositionalEncoding(config);
        // PE(pos=0, dim=0) = sin(0 / 10000^0) = sin(0) = 0
        assertEquals(0.0f, pe.getValue(0, 0), TOL);
    }

    @Test
    @DisplayName("PositionalEncoding: PE(0, odd) = cos(0) = 1")
    void testPEPosition0OddDim() {
        TransformerConfig config = tinyConfig();
        PositionalEncoding pe = new PositionalEncoding(config);
        // PE(pos=0, dim=1) = cos(0 / 10000^0) = cos(0) = 1
        assertEquals(1.0f, pe.getValue(0, 1), TOL);
    }

    @Test
    @DisplayName("PositionalEncoding: output shape matches input shape")
    void testPEOutputShape() {
        TransformerConfig config = tinyConfig();
        PositionalEncoding pe = new PositionalEncoding(config);
        Tensor3D input = Tensor3D.filled(2, 6, config.dModel, 0.0f);
        Tensor3D output = pe.forward(input);
        assertEquals(2, output.batch);
        assertEquals(6, output.rows);
        assertEquals(config.dModel, output.cols);
    }

    @Test
    @DisplayName("PositionalEncoding: values are added to input embeddings")
    void testPEAddedToEmbeddings() {
        TransformerConfig config = tinyConfig();
        PositionalEncoding pe = new PositionalEncoding(config);
        // All-zero embeddings → output should be exactly the PE values
        Tensor3D zeros = Tensor3D.filled(1, 5, config.dModel, 0.0f);
        Tensor3D output = pe.forward(zeros);
        float expected0 = pe.getValue(0, 0);  // should be sin(0)=0
        float expected1 = pe.getValue(0, 1);  // should be cos(0)=1
        assertEquals(expected0, output.get(0, 0, 0), TOL);
        assertEquals(expected1, output.get(0, 0, 1), TOL);
    }

    @Test
    @DisplayName("PositionalEncoding: exceeding maxSeqLen throws")
    void testPESeqLenExceeded() {
        TransformerConfig config = tinyConfig();  // maxSeqLen=16
        PositionalEncoding pe = new PositionalEncoding(config);
        Tensor3D tooLong = Tensor3D.filled(1, 20, config.dModel, 0f);  // 20 > 16
        assertThrows(IllegalArgumentException.class, () -> pe.forward(tooLong));
    }

    // ── LayerNorm ─────────────────────────────────────────────────────────────

    @Test
    @DisplayName("LayerNorm: output has approximately zero mean per row")
    void testLayerNormZeroMean() {
        LayerNorm ln = new LayerNorm(8, 1e-6);
        // Random input matrix [3 × 8]
        float[][] data = {
            {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f},
            {-4f, -2f, 0f, 2f, 4f, 6f, 8f, 10f},
            {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}
        };
        Matrix input = new Matrix(data);
        // With gamma=1, beta=0 initialisation, output should be normalised.
        Matrix output = ln.forward(input);

        // Each row's mean should be ~0 (due to normalisation + beta=0)
        for (int r = 0; r < output.rows; r++) {
            double mean = output.rowMean(r);
            assertEquals(0.0, mean, 1e-5, "LayerNorm row " + r + " should have near-zero mean");
        }
    }

    @Test
    @DisplayName("LayerNorm: output has approximately unit variance per row")
    void testLayerNormUnitVariance() {
        LayerNorm ln = new LayerNorm(8, 1e-6);
        float[][] data = {{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}};
        Matrix input = new Matrix(data);
        Matrix output = ln.forward(input);
        double mean = output.rowMean(0);
        double var  = output.rowVariance(0, mean);
        assertEquals(1.0, var, 1e-4, "LayerNorm output should have variance ~1.0");
    }

    @Test
    @DisplayName("LayerNorm: constant input produces all-zero output")
    void testLayerNormConstantInput() {
        LayerNorm ln = new LayerNorm(4, 1e-6);
        // All same values → variance=0 → output = (x−mean)/sqrt(var+eps) * gamma + beta
        // With gamma=1, beta=0 and mean=x=const: output = 0 * 1/sqrt(eps) = ~0
        float[][] data = {{5f, 5f, 5f, 5f}};
        Matrix input = new Matrix(data);
        Matrix output = ln.forward(input);
        for (float v : output.data[0]) {
            assertEquals(0.0f, v, 1e-3f, "Constant input → LayerNorm output should be ~0");
        }
    }

    @Test
    @DisplayName("LayerNorm: 3D tensor shape preserved")
    void testLayerNorm3DShape() {
        LayerNorm ln = new LayerNorm(8, 1e-6);
        Tensor3D input = Tensor3D.filled(2, 5, 8, 3.0f);
        Tensor3D output = ln.forward(input);
        assertEquals(2, output.batch);
        assertEquals(5, output.rows);
        assertEquals(8, output.cols);
    }

    // ── FeedForward ───────────────────────────────────────────────────────────

    @Test
    @DisplayName("FeedForward: output shape equals input shape")
    void testFFNShape() {
        TransformerConfig config = tinyConfig();
        Random rng = RandomUtils.createRng(42);
        FeedForward ffn = new FeedForward(config.dModel, config.dFF, rng);

        Matrix input = new Matrix(5, config.dModel);  // [5 × 8]
        Matrix output = ffn.forward(input);
        assertEquals(5, output.rows, "FFN should preserve sequence length");
        assertEquals(config.dModel, output.cols, "FFN should preserve dModel");
    }

    @Test
    @DisplayName("FeedForward: 3D input/output shape preserved")
    void testFFN3DShape() {
        TransformerConfig config = tinyConfig();
        Random rng = RandomUtils.createRng(42);
        FeedForward ffn = new FeedForward(config.dModel, config.dFF, rng);

        Tensor3D input = Tensor3D.filled(2, 5, config.dModel, 0.5f);
        Tensor3D output = ffn.forward(input);
        assertEquals(2, output.batch);
        assertEquals(5, output.rows);
        assertEquals(config.dModel, output.cols);
    }

    // ── Encoder ──────────────────────────────────────────────────────────────

    @Test
    @DisplayName("EncoderLayer: output shape equals input shape")
    void testEncoderLayerShape() {
        TransformerConfig config = tinyConfig();
        Random rng = RandomUtils.createRng(42);
        EncoderLayer layer = new EncoderLayer(config, rng);

        Tensor3D input = Tensor3D.filled(1, 5, config.dModel, 0.1f);
        Tensor3D output = layer.forward(input, null);
        assertEquals(1, output.batch);
        assertEquals(5, output.rows);
        assertEquals(config.dModel, output.cols);
    }

    @Test
    @DisplayName("Encoder: forward pass produces correct shape")
    void testEncoderShape() {
        TransformerConfig config = tinyConfig();
        Random rng = RandomUtils.createRng(42);
        Encoder encoder = new Encoder(config, 0, rng);

        // Source tokens: batch=1, seqLen=4
        int[][] tokenIds = {{3, 4, 5, 6}};  // all valid IDs
        Tensor3D output = encoder.forward(tokenIds);

        assertEquals(1, output.batch,  "Encoder batch should be 1");
        assertEquals(4, output.rows,   "Encoder should preserve sequence length");
        assertEquals(config.dModel, output.cols, "Encoder should produce dModel-dim vectors");
    }

    @Test
    @DisplayName("Encoder: output contains no NaN or Inf values")
    void testEncoderNoNaN() {
        TransformerConfig config = tinyConfig();
        Random rng = RandomUtils.createRng(42);
        Encoder encoder = new Encoder(config, 0, rng);

        int[][] tokenIds = {{1, 2, 3, 4, 5}};
        Tensor3D output = encoder.forward(tokenIds);

        for (int b = 0; b < output.batch; b++)
            for (int r = 0; r < output.rows; r++)
                for (int c = 0; c < output.cols; c++) {
                    float v = output.get(b, r, c);
                    assertFalse(Float.isNaN(v), "NaN at [" + b + "," + r + "," + c + "]");
                    assertFalse(Float.isInfinite(v), "Inf at [" + b + "," + r + "," + c + "]");
                }
    }

    // ── Decoder ──────────────────────────────────────────────────────────────

    @Test
    @DisplayName("DecoderLayer: output shape equals decoder input shape")
    void testDecoderLayerShape() {
        TransformerConfig config = tinyConfig();
        Random rng = RandomUtils.createRng(42);
        DecoderLayer layer = new DecoderLayer(config, rng);

        // Decoder input: [1 × 4 × dModel], Encoder output: [1 × 3 × dModel]
        Tensor3D decoderInput  = Tensor3D.filled(1, 4, config.dModel, 0.1f);
        Tensor3D encoderOutput = Tensor3D.filled(1, 3, config.dModel, 0.1f);

        // Causal mask for T=4
        float[][] causal = transformer.attention.AttentionMask.causalMask(4);

        Tensor3D output = layer.forward(decoderInput, encoderOutput, causal, null);
        assertEquals(1, output.batch);
        assertEquals(4, output.rows);
        assertEquals(config.dModel, output.cols);
    }

    @Test
    @DisplayName("Decoder: forward pass produces correct shape")
    void testDecoderShape() {
        TransformerConfig config = tinyConfig();
        Random rng = RandomUtils.createRng(42);
        Decoder decoder = new Decoder(config, 0, rng);

        // Target tokens: [1 × 4], Encoder output: [1 × 3 × dModel]
        int[][] targetIds     = {{1, 3, 4, 5}};  // [BOS, I, love, cats]
        int[][] srcIds        = {{3, 4, 5}};       // [I, love, cats]
        Tensor3D encoderOutput = Tensor3D.filled(1, 3, config.dModel, 0.1f);

        Tensor3D output = decoder.forward(targetIds, encoderOutput, srcIds);
        assertEquals(1, output.batch);
        assertEquals(4, output.rows,         "Decoder should produce T=4 positions");
        assertEquals(config.dModel, output.cols);
    }

    @Test
    @DisplayName("Decoder: output contains no NaN or Inf values")
    void testDecoderNoNaN() {
        TransformerConfig config = tinyConfig();
        Random rng = RandomUtils.createRng(42);
        Decoder decoder = new Decoder(config, 0, rng);

        int[][] targetIds     = {{1, 3, 4}};
        int[][] srcIds        = {{3, 4, 5}};
        Tensor3D encoderOutput = Tensor3D.filled(1, 3, config.dModel, 0.1f);

        Tensor3D output = decoder.forward(targetIds, encoderOutput, srcIds);
        for (int b = 0; b < output.batch; b++)
            for (int r = 0; r < output.rows; r++)
                for (int c = 0; c < output.cols; c++) {
                    float v = output.get(b, r, c);
                    assertFalse(Float.isNaN(v), "NaN at [" + b + "," + r + "," + c + "]");
                    assertFalse(Float.isInfinite(v), "Inf at [" + b + "," + r + "," + c + "]");
                }
    }

    // ── Full Transformer ──────────────────────────────────────────────────────

    @Test
    @DisplayName("Transformer: full forward pass produces correct shape")
    void testTransformerShape() {
        TransformerConfig config = tinyConfig();
        Transformer model = new Transformer(config, 0, 1, 2);

        int[][] source = {{3, 4, 5}};        // "I love cats" (S=3)
        int[][] target = {{1, 3, 4, 5}};     // [BOS, I, love, cats] (T=4)

        Tensor3D probs = model.forward(source, target);
        assertEquals(1, probs.batch,  "Batch should be 1");
        assertEquals(4, probs.rows,   "Output should have T=4 positions");
        assertEquals(config.vocabSize, probs.cols, "Output should cover full vocab");
    }

    @Test
    @DisplayName("Transformer: output probabilities sum to 1 at each position")
    void testTransformerProbabilitiesSum() {
        TransformerConfig config = tinyConfig();
        Transformer model = new Transformer(config, 0, 1, 2);

        int[][] source = {{3, 4, 5}};
        int[][] target = {{1, 3, 4, 5}};

        Tensor3D probs = model.forward(source, target);

        for (int t = 0; t < probs.rows; t++) {
            float sum = 0;
            for (float p : probs.data[0][t]) sum += p;
            assertEquals(1.0f, sum, 1e-4f,
                "Probabilities at position " + t + " should sum to 1.0");
        }
    }

    @Test
    @DisplayName("Transformer: output contains no NaN or Inf")
    void testTransformerNoNaN() {
        TransformerConfig config = tinyConfig();
        Transformer model = new Transformer(config, 0, 1, 2);

        int[][] source = {{3, 4, 5}};
        int[][] target = {{1, 3, 4}};

        Tensor3D probs = model.forward(source, target);

        for (int b = 0; b < probs.batch; b++)
            for (int t = 0; t < probs.rows; t++)
                for (int v = 0; v < probs.cols; v++) {
                    float p = probs.get(b, t, v);
                    assertFalse(Float.isNaN(p), "NaN at [" + b + "," + t + "," + v + "]");
                    assertFalse(Float.isInfinite(p), "Inf at [" + b + "," + t + "," + v + "]");
                }
    }

    @Test
    @DisplayName("Transformer: generate() returns non-null, non-empty result")
    void testTransformerGenerate() {
        TransformerConfig config = tinyConfig();
        Transformer model = new Transformer(config, 0, 1, 2);

        int[] source = {3, 4, 5};  // [I, love, cats]
        int[] generated = model.generate(source, 5);

        assertNotNull(generated, "generate() should not return null");
        assertTrue(generated.length > 0, "generate() should produce at least one token");
        assertTrue(generated.length <= 5 + 1, "generate() should not exceed maxLength (+EOS)");
    }
}
