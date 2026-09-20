/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import transformer.tensor.Tensor3D;
import transformer.tensor.Matrix;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tensor3DTest — Unit tests for the Tensor3D class.
 */
@DisplayName("Tensor3D Operations")
public class Tensor3DTest {

    private static final float TOL = 1e-5f;

    @Test
    @DisplayName("construction: zero-filled by default")
    void testZeroConstruction() {
        Tensor3D t = new Tensor3D(2, 3, 4);
        assertEquals(2, t.batch);
        assertEquals(3, t.rows);
        assertEquals(4, t.cols);
        assertEquals(0f, t.get(0, 0, 0), TOL);
        assertEquals(0f, t.get(1, 2, 3), TOL);
    }

    @Test
    @DisplayName("shapeString: correct format")
    void testShapeString() {
        Tensor3D t = new Tensor3D(1, 5, 8);
        assertEquals("[1 × 5 × 8]", t.shapeString());
    }

    @Test
    @DisplayName("add: element-wise addition")
    void testAdd() {
        Tensor3D a = Tensor3D.filled(1, 2, 3, 1.0f);
        Tensor3D b = Tensor3D.filled(1, 2, 3, 2.0f);
        Tensor3D c = a.add(b);
        assertEquals(3.0f, c.get(0, 0, 0), TOL);
        assertEquals(3.0f, c.get(0, 1, 2), TOL);
    }

    @Test
    @DisplayName("add: shape mismatch throws")
    void testAddMismatch() {
        Tensor3D a = new Tensor3D(1, 2, 3);
        Tensor3D b = new Tensor3D(1, 3, 2);
        assertThrows(IllegalArgumentException.class, () -> a.add(b));
    }

    @Test
    @DisplayName("transposeLastTwo: [B×R×C] → [B×C×R]")
    void testTransposeLastTwo() {
        // [1 × 2 × 3]: [[1,2,3],[4,5,6]]
        float[][][] data = {{{1,2,3},{4,5,6}}};
        Tensor3D t = new Tensor3D(data);
        Tensor3D tT = t.transposeLastTwo();  // [1 × 3 × 2]
        assertEquals(1, tT.batch);
        assertEquals(3, tT.rows);
        assertEquals(2, tT.cols);
        // tT[0][0][0] = t[0][0][0] = 1
        assertEquals(1f, tT.get(0, 0, 0), TOL);
        // tT[0][1][0] = t[0][0][1] = 2
        assertEquals(2f, tT.get(0, 1, 0), TOL);
        // tT[0][2][1] = t[0][1][2] = 6
        assertEquals(6f, tT.get(0, 2, 1), TOL);
    }

    @Test
    @DisplayName("splitLastDim: splits correctly into equal parts")
    void testSplitLastDim() {
        // [1 × 2 × 6]: split into 3 tensors each [1 × 2 × 2]
        float[][][] data = {{{1,2,3,4,5,6},{7,8,9,10,11,12}}};
        Tensor3D t = new Tensor3D(data);
        Tensor3D[] parts = t.splitLastDim(3);
        assertEquals(3, parts.length);
        for (Tensor3D p : parts) {
            assertEquals(1, p.batch);
            assertEquals(2, p.rows);
            assertEquals(2, p.cols);
        }
        // First part: cols 0-1
        assertEquals(1f, parts[0].get(0, 0, 0), TOL);
        assertEquals(2f, parts[0].get(0, 0, 1), TOL);
        // Second part: cols 2-3
        assertEquals(3f, parts[1].get(0, 0, 0), TOL);
        assertEquals(4f, parts[1].get(0, 0, 1), TOL);
        // Third part: cols 4-5
        assertEquals(5f, parts[2].get(0, 0, 0), TOL);
        assertEquals(6f, parts[2].get(0, 0, 1), TOL);
    }

    @Test
    @DisplayName("splitLastDim: indivisible size throws")
    void testSplitLastDimFails() {
        Tensor3D t = new Tensor3D(1, 2, 5);  // cols=5, not divisible by 2
        assertThrows(IllegalArgumentException.class, () -> t.splitLastDim(2));
    }

    @Test
    @DisplayName("concatLastDim: merges correctly")
    void testConcatLastDim() {
        Tensor3D a = Tensor3D.filled(1, 3, 2, 1.0f);
        Tensor3D b = Tensor3D.filled(1, 3, 3, 2.0f);
        Tensor3D c = Tensor3D.concatLastDim(new Tensor3D[]{a, b});
        assertEquals(1, c.batch);
        assertEquals(3, c.rows);
        assertEquals(5, c.cols);
        // First 2 cols from a (=1.0), next 3 from b (=2.0)
        assertEquals(1.0f, c.get(0, 0, 0), TOL);
        assertEquals(1.0f, c.get(0, 0, 1), TOL);
        assertEquals(2.0f, c.get(0, 0, 2), TOL);
        assertEquals(2.0f, c.get(0, 0, 4), TOL);
    }

    @Test
    @DisplayName("splitLastDim then concatLastDim round-trips correctly")
    void testSplitConcatRoundTrip() {
        Tensor3D original = new Tensor3D(new float[][][]{
            {{1,2,3,4,5,6,7,8}}
        });
        Tensor3D[] parts = original.splitLastDim(4);  // 4 parts of size 2
        Tensor3D reassembled = Tensor3D.concatLastDim(parts);
        assertEquals(original.batch, reassembled.batch);
        assertEquals(original.rows, reassembled.rows);
        assertEquals(original.cols, reassembled.cols);
        for (int c = 0; c < original.cols; c++) {
            assertEquals(original.get(0, 0, c), reassembled.get(0, 0, c), TOL,
                "Mismatch at col " + c);
        }
    }

    @Test
    @DisplayName("reshape: total elements preserved")
    void testReshape() {
        // [2 × 3 × 4] = 24 elements → [1 × 6 × 4] = 24 elements
        Tensor3D t = new Tensor3D(2, 3, 4);
        for (int b = 0; b < 2; b++)
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    t.set(b, r, c, b * 12 + r * 4 + c);

        Tensor3D reshaped = t.reshape(1, 6, 4);
        assertEquals(1, reshaped.batch);
        assertEquals(6, reshaped.rows);
        assertEquals(4, reshaped.cols);
        // First 12 elements from batch 0 appear in rows 0-2, next 12 from batch 1 in rows 3-5.
        assertEquals(0f, reshaped.get(0, 0, 0), TOL);
        assertEquals(23f, reshaped.get(0, 5, 3), TOL);
    }

    @Test
    @DisplayName("reshape: element count mismatch throws")
    void testReshapeMismatch() {
        Tensor3D t = new Tensor3D(2, 3, 4);  // 24 elements
        assertThrows(IllegalArgumentException.class, () -> t.reshape(1, 5, 4));  // 20 elements
    }

    @Test
    @DisplayName("batchMatmul: [B×M×K] × [B×K×N] = [B×M×N]")
    void testBatchMatmul() {
        // [2 × 2 × 3] × [2 × 3 × 2]
        Tensor3D A = new Tensor3D(new float[][][]{
            {{1,0,0},{0,1,0}},  // batch 0: identity-like [2×3]
            {{2,0,0},{0,2,0}}   // batch 1: scaled [2×3]
        });
        Tensor3D B = new Tensor3D(new float[][][]{
            {{1,2},{3,4},{5,6}},
            {{1,2},{3,4},{5,6}}
        });
        Tensor3D C = A.batchMatmul(B);
        assertEquals(2, C.batch);
        assertEquals(2, C.rows);
        assertEquals(2, C.cols);
        // batch 0: row 0 of A is [1,0,0], times B → [1,2]
        assertEquals(1f, C.get(0, 0, 0), TOL);
        assertEquals(2f, C.get(0, 0, 1), TOL);
        // batch 1: row 0 of A is [2,0,0], times B → [2,4]
        assertEquals(2f, C.get(1, 0, 0), TOL);
        assertEquals(4f, C.get(1, 0, 1), TOL);
    }
}
