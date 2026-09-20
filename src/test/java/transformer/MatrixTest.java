/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import transformer.tensor.Matrix;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MatrixTest — Unit tests for the Matrix class.
 *
 * <p>Tests cover all mathematical operations that the Transformer relies upon.
 * Uses a small tolerance (1e-5) for floating-point comparisons.
 */
@DisplayName("Matrix Operations")
public class MatrixTest {

    private static final float TOLERANCE = 1e-5f;

    // ── Matrix multiplication ─────────────────────────────────────────────────

    @Test
    @DisplayName("matmul: [2×3] × [3×2] = [2×2]")
    void testMatmulBasic() {
        // [[1,2,3],[4,5,6]] × [[7,8],[9,10],[11,12]]
        // Row 0: 1×7+2×9+3×11=7+18+33=58, 1×8+2×10+3×12=8+20+36=64
        // Row 1: 4×7+5×9+6×11=28+45+66=139, 4×8+5×10+6×12=32+50+72=154
        Matrix A = new Matrix(new float[][]{{1,2,3},{4,5,6}});
        Matrix B = new Matrix(new float[][]{{7,8},{9,10},{11,12}});
        Matrix C = A.matmul(B);

        assertEquals(2, C.rows);
        assertEquals(2, C.cols);
        assertEquals(58f,  C.data[0][0], TOLERANCE);
        assertEquals(64f,  C.data[0][1], TOLERANCE);
        assertEquals(139f, C.data[1][0], TOLERANCE);
        assertEquals(154f, C.data[1][1], TOLERANCE);
    }

    @Test
    @DisplayName("matmul: dimension mismatch throws exception")
    void testMatmulDimensionMismatch() {
        Matrix A = new Matrix(new float[][]{{1,2},{3,4}});   // [2×2]
        Matrix B = new Matrix(new float[][]{{1,2,3}});       // [1×3]
        // [2×2] × [1×3] → cols(2) ≠ rows(1) → should throw
        assertThrows(IllegalArgumentException.class, () -> A.matmul(B));
    }

    @Test
    @DisplayName("matmul: identity matrix")
    void testMatmulIdentity() {
        Matrix A = new Matrix(new float[][]{{1,2,3},{4,5,6}});
        Matrix I = Matrix.identity(3);
        Matrix result = A.matmul(I);
        // A × I should equal A
        for (int r = 0; r < A.rows; r++)
            for (int c = 0; c < A.cols; c++)
                assertEquals(A.data[r][c], result.data[r][c], TOLERANCE);
    }

    // ── Transpose ────────────────────────────────────────────────────────────

    @Test
    @DisplayName("transpose: [2×3] → [3×2]")
    void testTranspose() {
        Matrix A = new Matrix(new float[][]{{1,2,3},{4,5,6}});
        Matrix T = A.transpose();

        assertEquals(3, T.rows);
        assertEquals(2, T.cols);
        assertEquals(1f, T.data[0][0], TOLERANCE);
        assertEquals(4f, T.data[0][1], TOLERANCE);
        assertEquals(2f, T.data[1][0], TOLERANCE);
        assertEquals(5f, T.data[1][1], TOLERANCE);
        assertEquals(3f, T.data[2][0], TOLERANCE);
        assertEquals(6f, T.data[2][1], TOLERANCE);
    }

    @Test
    @DisplayName("transpose: double transpose equals original")
    void testDoubleTranspose() {
        Matrix A = new Matrix(new float[][]{{1,2,3},{4,5,6},{7,8,9}});
        Matrix TT = A.transpose().transpose();
        assertEquals(A, TT);
    }

    // ── Addition ─────────────────────────────────────────────────────────────

    @Test
    @DisplayName("add: element-wise addition")
    void testAdd() {
        Matrix A = new Matrix(new float[][]{{1,2},{3,4}});
        Matrix B = new Matrix(new float[][]{{5,6},{7,8}});
        Matrix C = A.add(B);
        assertEquals(6f, C.data[0][0], TOLERANCE);
        assertEquals(8f, C.data[0][1], TOLERANCE);
        assertEquals(10f, C.data[1][0], TOLERANCE);
        assertEquals(12f, C.data[1][1], TOLERANCE);
    }

    @Test
    @DisplayName("add: shape mismatch throws exception")
    void testAddMismatch() {
        Matrix A = new Matrix(new float[][]{{1,2}});    // [1×2]
        Matrix B = new Matrix(new float[][]{{1},{2}});  // [2×1]
        assertThrows(IllegalArgumentException.class, () -> A.add(B));
    }

    // ── Scale ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("scale: multiply all elements by scalar")
    void testScale() {
        Matrix A = new Matrix(new float[][]{{1,2},{3,4}});
        Matrix B = A.scale(2.0f);
        assertEquals(2f, B.data[0][0], TOLERANCE);
        assertEquals(4f, B.data[0][1], TOLERANCE);
        assertEquals(6f, B.data[1][0], TOLERANCE);
        assertEquals(8f, B.data[1][1], TOLERANCE);
    }

    // ── Softmax ───────────────────────────────────────────────────────────────

    @Test
    @DisplayName("softmax: each row sums to approximately 1.0")
    void testSoftmaxRowSumsToOne() {
        Matrix A = new Matrix(new float[][]{{1,2,3},{4,1,2}});
        Matrix S = A.softmax();
        for (int r = 0; r < S.rows; r++) {
            float sum = 0;
            for (int c = 0; c < S.cols; c++) sum += S.data[r][c];
            assertEquals(1.0f, sum, 1e-5f, "Row " + r + " should sum to 1.0");
        }
    }

    @Test
    @DisplayName("softmax: all outputs are positive")
    void testSoftmaxAllPositive() {
        Matrix A = new Matrix(new float[][]{{-100,0,100}});
        Matrix S = A.softmax();
        for (float v : S.data[0]) assertTrue(v >= 0, "softmax outputs must be non-negative");
    }

    @Test
    @DisplayName("softmax: numerically stable with very large values")
    void testSoftmaxStability() {
        // Without the max subtraction trick, exp(1000) overflows to +infinity.
        // With the trick, we compute exp(1000-1000)=exp(0)=1 safely.
        Matrix A = new Matrix(new float[][]{{1000f, 1000f, 1000f}});
        Matrix S = A.softmax();
        // All equal values → uniform distribution
        for (float v : S.data[0]) {
            assertFalse(Float.isNaN(v), "softmax should not produce NaN for large inputs");
            assertFalse(Float.isInfinite(v), "softmax should not produce Inf for large inputs");
            assertEquals(1.0f / 3.0f, v, 1e-4f, "Equal logits should give uniform distribution");
        }
    }

    @Test
    @DisplayName("softmax: numerically stable with very negative values")
    void testSoftmaxNegativeStability() {
        Matrix A = new Matrix(new float[][]{{-1000f, 0f, 1000f}});
        Matrix S = A.softmax();
        assertFalse(Float.isNaN(S.data[0][0]));
        assertFalse(Float.isNaN(S.data[0][2]));
        // The last element (1000) should dominate → ≈ 1.0
        assertTrue(S.data[0][2] > 0.99f, "Largest value should dominate softmax");
    }

    // ── ReLU ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("relu: negatives become 0, positives unchanged")
    void testRelu() {
        Matrix A = new Matrix(new float[][]{{-3, -1, 0, 1, 3}});
        Matrix R = A.relu();
        assertEquals(0f, R.data[0][0], TOLERANCE);
        assertEquals(0f, R.data[0][1], TOLERANCE);
        assertEquals(0f, R.data[0][2], TOLERANCE);
        assertEquals(1f, R.data[0][3], TOLERANCE);
        assertEquals(3f, R.data[0][4], TOLERANCE);
    }

    // ── addBias ───────────────────────────────────────────────────────────────

    @Test
    @DisplayName("addBias: adds 1-D bias to every row")
    void testAddBias() {
        Matrix A = new Matrix(new float[][]{{1,2,3},{4,5,6}});
        float[] bias = {10, 20, 30};
        Matrix B = A.addBias(bias);
        assertEquals(11f, B.data[0][0], TOLERANCE);
        assertEquals(22f, B.data[0][1], TOLERANCE);
        assertEquals(33f, B.data[0][2], TOLERANCE);
        assertEquals(14f, B.data[1][0], TOLERANCE);
        assertEquals(25f, B.data[1][1], TOLERANCE);
        assertEquals(36f, B.data[1][2], TOLERANCE);
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    @Test
    @DisplayName("rowMean: correct mean for known values")
    void testRowMean() {
        Matrix A = new Matrix(new float[][]{{1f, 2f, 3f, 4f}});
        assertEquals(2.5, A.rowMean(0), 1e-6);
    }

    @Test
    @DisplayName("rowVariance: correct variance for known values")
    void testRowVariance() {
        // Values: [2, 4, 4, 4, 5, 5, 7, 9], mean=5, variance=4
        Matrix A = new Matrix(new float[][]{{2,4,4,4,5,5,7,9}});
        double mean = A.rowMean(0);
        double variance = A.rowVariance(0, mean);
        assertEquals(5.0, mean, 1e-5);
        assertEquals(4.0, variance, 1e-5);
    }
}
