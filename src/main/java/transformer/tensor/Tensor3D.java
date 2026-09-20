/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.tensor;

import java.util.Arrays;

/**
 * Tensor3D — a 3-D tensor with shape [batch × rows × cols] backed by float[][][].
 *
 * <p>In the Transformer, 3-D tensors appear everywhere:
 * <pre>
 *   [B × S × dModel]   — sequences of token embeddings
 *   [B × h × S × dHead] — (represented as a Tensor3D after head-splitting)
 *   [B × S × S]         — attention score matrices
 * </pre>
 *
 * <p>Head-related tensors use a reshaped Tensor3D with:
 * <pre>
 *   batch  = B × h   (batch and head dimensions are merged during computation)
 *   rows   = S
 *   cols   = dHead
 * </pre>
 *
 * <p>This avoids requiring a 4-D tensor class while keeping the math clear.
 *
 * <h2>Dimension notation used throughout</h2>
 * <pre>
 *   Shape: [batch × rows × cols]
 *   Access: data[b][r][c]
 * </pre>
 */
public final class Tensor3D {

    /** Raw data: data[batch][row][col]. */
    public final float[][][] data;

    /** Number of batches (first dimension). */
    public final int batch;

    /** Number of rows in each batch slice (second dimension). */
    public final int rows;

    /** Number of columns in each row (third dimension). */
    public final int cols;

    // ── Constructors ────────────────────────────────────────────────────────

    /**
     * Construct from existing 3-D float array.
     * The array is used directly (not copied).
     */
    public Tensor3D(float[][][] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("Tensor3D data must be non-null and non-empty.");
        this.data  = data;
        this.batch = data.length;
        this.rows  = data[0].length;
        this.cols  = data[0][0].length;
        // Validate consistency
        for (int b = 0; b < batch; b++) {
            if (data[b].length != rows)
                throw new IllegalArgumentException(
                    "Tensor3D batch slice " + b + " has " + data[b].length +
                    " rows but expected " + rows);
            for (int r = 0; r < rows; r++) {
                if (data[b][r].length != cols)
                    throw new IllegalArgumentException(
                        "Tensor3D [" + b + "][" + r + "] has " + data[b][r].length +
                        " cols but expected " + cols);
            }
        }
    }

    /** Construct a zero-filled Tensor3D with given dimensions. */
    public Tensor3D(int batch, int rows, int cols) {
        if (batch <= 0 || rows <= 0 || cols <= 0)
            throw new IllegalArgumentException(
                "All Tensor3D dimensions must be positive, got [" + batch +
                " × " + rows + " × " + cols + "].");
        this.batch = batch;
        this.rows  = rows;
        this.cols  = cols;
        this.data  = new float[batch][rows][cols];
    }

    // ── Shape utilities ──────────────────────────────────────────────────────

    /** Returns "[batch × rows × cols]" for error messages and debug printing. */
    public String shapeString() {
        return "[" + batch + " × " + rows + " × " + cols + "]";
    }

    // ── Slice access ─────────────────────────────────────────────────────────

    /**
     * Extract batch slice {@code b} as a Matrix [rows × cols].
     *
     * @param b batch index
     * @return Matrix view of the b-th batch slice
     */
    public Matrix getMatrix(int b) {
        if (b < 0 || b >= batch)
            throw new IndexOutOfBoundsException(
                "Batch index " + b + " out of bounds for batch size " + batch);
        return new Matrix(data[b]);
    }

    /**
     * Set batch slice {@code b} from a Matrix [rows × cols].
     */
    public void setMatrix(int b, Matrix m) {
        if (b < 0 || b >= batch)
            throw new IndexOutOfBoundsException("Batch index " + b + " out of bounds");
        if (m.rows != rows || m.cols != cols)
            throw new IllegalArgumentException(
                "setMatrix: matrix shape " + m.shapeString() +
                " does not match slice shape [" + rows + " × " + cols + "]");
        data[b] = m.data;
    }

    /**
     * Get a single scalar value.
     */
    public float get(int b, int r, int c) {
        return data[b][r][c];
    }

    /**
     * Set a single scalar value.
     */
    public void set(int b, int r, int c, float value) {
        data[b][r][c] = value;
    }

    // ── Mathematical operations ──────────────────────────────────────────────

    /**
     * Element-wise addition: this + other (same shape required).
     */
    public Tensor3D add(Tensor3D other) {
        checkSameShape("add", other);
        Tensor3D result = new Tensor3D(batch, rows, cols);
        for (int b = 0; b < batch; b++)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    result.data[b][r][c] = data[b][r][c] + other.data[b][r][c];
        return result;
    }

    /**
     * Element-wise subtraction: this - other.
     */
    public Tensor3D subtract(Tensor3D other) {
        checkSameShape("subtract", other);
        Tensor3D result = new Tensor3D(batch, rows, cols);
        for (int b = 0; b < batch; b++)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    result.data[b][r][c] = data[b][r][c] - other.data[b][r][c];
        return result;
    }

    /**
     * Scalar multiplication: every element multiplied by {@code scalar}.
     */
    public Tensor3D scale(float scalar) {
        Tensor3D result = new Tensor3D(batch, rows, cols);
        for (int b = 0; b < batch; b++)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    result.data[b][r][c] = data[b][r][c] * scalar;
        return result;
    }

    /**
     * Apply a function to every batch slice and return a new Tensor3D.
     * The function maps a Matrix [rows × cols] → Matrix [rows × cols].
     *
     * <p>Example: apply row-wise softmax to every batch slice.
     *
     * @param fn slice transformation
     * @return new Tensor3D
     */
    public Tensor3D applyToEachSlice(java.util.function.UnaryOperator<Matrix> fn) {
        Tensor3D result = new Tensor3D(batch, rows, cols);
        for (int b = 0; b < batch; b++) {
            Matrix out = fn.apply(getMatrix(b));
            result.setMatrix(b, out);
        }
        return result;
    }

    /**
     * Batch matrix multiplication: multiply each slice of this [rows × k]
     * by the corresponding slice of other [k × outCols].
     *
     * <p>Shape: [batch × rows × k] × [batch × k × outCols] → [batch × rows × outCols]
     *
     * @param other Tensor3D with this.cols == other.rows
     * @return new Tensor3D with shape [batch × this.rows × other.cols]
     */
    public Tensor3D batchMatmul(Tensor3D other) {
        if (this.batch != other.batch)
            throw new IllegalArgumentException(
                "batchMatmul: batch sizes differ: " + this.batch + " vs " + other.batch);
        if (this.cols != other.rows)
            throw new IllegalArgumentException(
                "batchMatmul dimension mismatch:\n" +
                "  left  = " + this.shapeString() + "\n" +
                "  right = " + other.shapeString() + "\n" +
                "  Required: left.cols (" + this.cols + ") == right.rows (" + other.rows + ")");

        Tensor3D result = new Tensor3D(batch, this.rows, other.cols);
        for (int b = 0; b < batch; b++) {
            result.setMatrix(b, getMatrix(b).matmul(other.getMatrix(b)));
        }
        return result;
    }

    // ── Reshape & split ──────────────────────────────────────────────────────

    /**
     * Reshape this Tensor3D to a new shape, keeping total elements the same.
     * Elements are read/written in row-major order (batch → row → col).
     *
     * @param newBatch new batch size
     * @param newRows  new rows
     * @param newCols  new cols
     * @return reshaped Tensor3D
     * @throws IllegalArgumentException if total element counts don't match
     */
    public Tensor3D reshape(int newBatch, int newRows, int newCols) {
        long totalOld = (long) batch * rows * cols;
        long totalNew = (long) newBatch * newRows * newCols;
        if (totalOld != totalNew)
            throw new IllegalArgumentException(
                "reshape: cannot reshape " + shapeString() +
                " (" + totalOld + " elements) to [" + newBatch + " × " + newRows + " × " + newCols +
                "] (" + totalNew + " elements)");

        float[][][] result = new float[newBatch][newRows][newCols];
        // Flatten source and refill
        int idx = 0;
        float[] flat = flatten();
        for (int b = 0; b < newBatch; b++)
            for (int r = 0; r < newRows; r++)
                for (int c = 0; c < newCols; c++)
                    result[b][r][c] = flat[idx++];
        return new Tensor3D(result);
    }

    /**
     * Flatten to a 1-D float array (row-major: batch → row → col).
     */
    public float[] flatten() {
        float[] flat = new float[batch * rows * cols];
        int idx = 0;
        for (int b = 0; b < batch; b++)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    flat[idx++] = data[b][r][c];
        return flat;
    }

    /**
     * Transpose the last two dimensions: [batch × rows × cols] → [batch × cols × rows].
     * Used in attention: K^T where K has shape [batch × S × dHead].
     */
    public Tensor3D transposeLastTwo() {
        Tensor3D result = new Tensor3D(batch, cols, rows);
        for (int b = 0; b < batch; b++)
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    result.data[b][c][r] = data[b][r][c];
        return result;
    }

    /**
     * Split along the last (cols) dimension.
     *
     * <p>Used to split [B × S × dModel] into h tensors each [B × S × dHead].
     *
     * @param numSplits number of equal-sized splits
     * @return array of numSplits tensors each with shape [batch × rows × (cols/numSplits)]
     * @throws IllegalArgumentException if cols is not divisible by numSplits
     */
    public Tensor3D[] splitLastDim(int numSplits) {
        if (cols % numSplits != 0)
            throw new IllegalArgumentException(
                "splitLastDim: cols (" + cols + ") not divisible by numSplits (" + numSplits + ")");
        int splitSize = cols / numSplits;
        Tensor3D[] parts = new Tensor3D[numSplits];
        for (int s = 0; s < numSplits; s++) {
            parts[s] = new Tensor3D(batch, rows, splitSize);
            int colStart = s * splitSize;
            for (int b = 0; b < batch; b++)
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < splitSize; c++)
                        parts[s].data[b][r][c] = data[b][r][colStart + c];
        }
        return parts;
    }

    /**
     * Concatenate multiple Tensor3Ds along the last (cols) dimension.
     *
     * <p>Used to merge h head outputs [B × S × dHead] → [B × S × dModel].
     *
     * @param tensors must all have same batch and rows; cols are summed
     * @return concatenated tensor
     */
    public static Tensor3D concatLastDim(Tensor3D[] tensors) {
        if (tensors == null || tensors.length == 0)
            throw new IllegalArgumentException("concatLastDim: tensors array is empty.");
        int batchSize = tensors[0].batch;
        int rowSize   = tensors[0].rows;
        int totalCols = 0;
        for (Tensor3D t : tensors) {
            if (t.batch != batchSize || t.rows != rowSize)
                throw new IllegalArgumentException(
                    "concatLastDim: all tensors must have same batch and rows.\n" +
                    "  Expected [" + batchSize + " × " + rowSize + " × ?]\n" +
                    "  Got      " + t.shapeString());
            totalCols += t.cols;
        }
        Tensor3D result = new Tensor3D(batchSize, rowSize, totalCols);
        for (int b = 0; b < batchSize; b++) {
            for (int r = 0; r < rowSize; r++) {
                int colOffset = 0;
                for (Tensor3D t : tensors) {
                    for (int c = 0; c < t.cols; c++) {
                        result.data[b][r][colOffset + c] = t.data[b][r][c];
                    }
                    colOffset += t.cols;
                }
            }
        }
        return result;
    }

    // ── Dimension checking ───────────────────────────────────────────────────

    private void checkSameShape(String op, Tensor3D other) {
        if (batch != other.batch || rows != other.rows || cols != other.cols)
            throw new IllegalArgumentException(
                op + " dimension mismatch:\n" +
                "  this  = " + shapeString() + "\n" +
                "  other = " + other.shapeString());
    }

    // ── Static factory ───────────────────────────────────────────────────────

    /**
     * Create a Tensor3D filled with a constant value.
     */
    public static Tensor3D filled(int batch, int rows, int cols, float value) {
        Tensor3D t = new Tensor3D(batch, rows, cols);
        for (int b = 0; b < batch; b++)
            for (float[] row : t.data[b])
                Arrays.fill(row, value);
        return t;
    }

    /**
     * Wrap a single Matrix as a Tensor3D with batch=1.
     */
    public static Tensor3D fromMatrix(Matrix m) {
        return new Tensor3D(new float[][][]{m.data});
    }

    /**
     * Unwrap a Tensor3D with batch=1 to a Matrix.
     */
    public Matrix toMatrix() {
        if (batch != 1)
            throw new IllegalStateException(
                "toMatrix() requires batch==1, but batch=" + batch);
        return getMatrix(0);
    }

    // ── Object overrides ─────────────────────────────────────────────────────

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor3D").append(shapeString()).append(":\n");
        for (int b = 0; b < batch; b++) {
            sb.append("  Batch ").append(b).append(":\n");
            for (int r = 0; r < rows; r++) {
                sb.append("    [");
                for (int c = 0; c < cols; c++) {
                    sb.append(String.format("%8.4f", data[b][r][c]));
                    if (c < cols - 1) sb.append(", ");
                }
                sb.append("]\n");
            }
        }
        return sb.toString();
    }
}
