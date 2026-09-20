/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.tensor;

import java.util.Arrays;

/**
 * Matrix — a 2-D tensor with shape [rows × cols] backed by a float[][].
 *
 * <p>This class implements all matrix operations required by the Transformer
 * without any external library dependency. Every public method validates
 * dimensions eagerly and throws a descriptive exception on mismatch.
 *
 * <h2>Design choices</h2>
 * <ul>
 *   <li>Stored row-major: {@code data[row][col]}.
 *   <li>Operations return new Matrix objects (immutable-style); the original is not modified.
 *   <li>Internal {@code float} precision is a pragmatic balance between clarity and performance;
 *       {@code double} arithmetic is used inside some operations for numerical stability.
 * </ul>
 *
 * <h2>Dimension notation</h2>
 * <pre>
 *   [M × K]  matmul  [K × N]  →  [M × N]
 *   [M × N]  transpose       →  [N × M]
 *   [M × N]  add     [M × N] →  [M × N]
 * </pre>
 */
public final class Matrix {

    /** Raw data: data[row][col]. */
    public final float[][] data;

    /** Number of rows. */
    public final int rows;

    /** Number of columns. */
    public final int cols;

    // ── Constructors ────────────────────────────────────────────────────────

    /**
     * Construct from an existing 2-D array.
     * The array is used directly (not copied) for performance; callers should
     * not mutate it after passing it here unless they own the Matrix object.
     */
    public Matrix(float[][] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("Matrix data must be non-null and non-empty.");
        this.data = data;
        this.rows = data.length;
        this.cols = data[0].length;
        // Validate all rows have the same width.
        for (int r = 1; r < rows; r++) {
            if (data[r].length != cols)
                throw new IllegalArgumentException(
                    "Matrix row " + r + " has length " + data[r].length +
                    " but expected " + cols + ".");
        }
    }

    /**
     * Construct a zero matrix with given dimensions.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0)
            throw new IllegalArgumentException(
                "Matrix dimensions must be positive, got [" + rows + " × " + cols + "].");
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols];  // Java zero-initialises float arrays
    }

    // ── Shape utilities ──────────────────────────────────────────────────────

    /** Returns "[rows × cols]" string for error messages and debug printing. */
    public String shapeString() {
        return "[" + rows + " × " + cols + "]";
    }

    // ── Core mathematical operations ─────────────────────────────────────────

    /**
     * Matrix multiplication: this [M × K] × other [K × N] → result [M × N].
     *
     * <p>Implements the standard O(M·K·N) algorithm.
     * Uses double accumulation to reduce floating-point drift.
     *
     * @param other right-hand matrix [K × N]
     * @return new Matrix [M × N]
     * @throws IllegalArgumentException if this.cols ≠ other.rows
     */
    public Matrix matmul(Matrix other) {
        if (this.cols != other.rows)
            throw new IllegalArgumentException(
                "Matrix multiplication dimension mismatch:\n" +
                "  left  = " + this.shapeString() + "\n" +
                "  right = " + other.shapeString() + "\n" +
                "  Required: left.cols (" + this.cols + ") == right.rows (" + other.rows + ")");

        int M = this.rows, K = this.cols, N = other.cols;
        float[][] result = new float[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += (double) this.data[i][k] * other.data[k][j];
                }
                result[i][j] = (float) sum;
            }
        }
        return new Matrix(result);
    }

    /**
     * Transpose: this [M × N] → result [N × M].
     * Swaps rows and columns.
     *
     * @return new transposed Matrix
     */
    public Matrix transpose() {
        float[][] result = new float[cols][rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[c][r] = data[r][c];
        return new Matrix(result);
    }

    /**
     * Element-wise addition: this [M × N] + other [M × N] → [M × N].
     *
     * @param other must have same shape as this
     * @return new Matrix with element-wise sums
     */
    public Matrix add(Matrix other) {
        checkSameShape("add", other);
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r][c] = data[r][c] + other.data[r][c];
        return new Matrix(result);
    }

    /**
     * Element-wise subtraction: this [M × N] - other [M × N] → [M × N].
     */
    public Matrix subtract(Matrix other) {
        checkSameShape("subtract", other);
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r][c] = data[r][c] - other.data[r][c];
        return new Matrix(result);
    }

    /**
     * Scalar multiplication: every element multiplied by {@code scalar}.
     *
     * @param scalar the scale factor
     * @return new scaled Matrix
     */
    public Matrix scale(float scalar) {
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r][c] = data[r][c] * scalar;
        return new Matrix(result);
    }

    /**
     * Element-wise multiplication (Hadamard product): this ⊙ other.
     * Same shape required.
     */
    public Matrix elementwiseMultiply(Matrix other) {
        checkSameShape("elementwiseMultiply", other);
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r][c] = data[r][c] * other.data[r][c];
        return new Matrix(result);
    }

    /**
     * Add a bias vector (row vector) to every row of this matrix.
     * Broadcasts bias [1 × cols] → [rows × cols].
     *
     * <p>Used in: {@code y = x·W + b} where b is a bias vector.
     *
     * @param bias a Matrix with shape [1 × cols] or [cols × 1] accepted as [1 × cols]
     * @return new Matrix with bias added to every row
     */
    public Matrix addBias(float[] bias) {
        if (bias.length != cols)
            throw new IllegalArgumentException(
                "addBias: bias length (" + bias.length + ") must equal cols (" + cols + ").");
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r][c] = data[r][c] + bias[c];
        return new Matrix(result);
    }

    /**
     * Apply a mask to this matrix by adding {@code maskValue} where mask is true.
     *
     * <p>In attention, we add −∞ (very large negative) to masked positions before softmax,
     * which causes softmax to output ≈ 0 for those positions.
     *
     * @param mask      boolean mask, same shape as this
     * @param maskValue value to add where mask[r][c] == true (typically Float.NEGATIVE_INFINITY)
     * @return new masked Matrix
     */
    public Matrix applyMask(boolean[][] mask, float maskValue) {
        if (mask.length != rows || mask[0].length != cols)
            throw new IllegalArgumentException(
                "applyMask: mask shape [" + mask.length + " × " + mask[0].length +
                "] does not match matrix shape " + shapeString());
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r][c] = mask[r][c] ? data[r][c] + maskValue : data[r][c];
        return new Matrix(result);
    }

    /**
     * Numerically stable row-wise softmax.
     *
     * <p>Standard softmax is:
     * <pre>
     *   softmax(x)_i = exp(x_i) / Σ exp(x_j)
     * </pre>
     *
     * <p>Numerically stable version (max subtraction trick):
     * <pre>
     *   x_stable = x - max(x)
     *   softmax(x)_i = exp(x_stable_i) / Σ exp(x_stable_j)
     * </pre>
     *
     * <p>Subtracting the row maximum before exponentiation prevents overflow
     * while not changing the output (the constant cancels in numerator and denominator).
     *
     * @return new Matrix with softmax applied row-wise
     */
    public Matrix softmax() {
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            // 1. Find row maximum for numerical stability.
            float rowMax = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++)
                if (data[r][c] > rowMax) rowMax = data[r][c];

            // 2. Exponentiate shifted values; accumulate sum.
            double sum = 0.0;
            for (int c = 0; c < cols; c++) {
                result[r][c] = (float) Math.exp(data[r][c] - rowMax);
                sum += result[r][c];
            }

            // 3. Normalise.
            for (int c = 0; c < cols; c++)
                result[r][c] /= (float) sum;
        }
        return new Matrix(result);
    }

    /**
     * Element-wise ReLU: max(0, x).
     */
    public Matrix relu() {
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[r][c] = Math.max(0f, data[r][c]);
        return new Matrix(result);
    }

    /**
     * Element-wise GELU approximation (Gaussian Error Linear Unit).
     * GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
     * This is the commonly used polynomial approximation.
     */
    public Matrix gelu() {
        float[][] result = new float[rows][cols];
        final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                double x = data[r][c];
                result[r][c] = (float) (0.5 * x *
                    (1.0 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x))));
            }
        return new Matrix(result);
    }

    // ── Statistics (used by LayerNorm) ───────────────────────────────────────

    /**
     * Compute the mean of a specific row.
     *
     * @param row row index
     * @return mean of that row's values
     */
    public double rowMean(int row) {
        double sum = 0.0;
        for (int c = 0; c < cols; c++) sum += data[row][c];
        return sum / cols;
    }

    /**
     * Compute the variance of a specific row given its mean.
     * Uses population variance (no Bessel correction), consistent with PyTorch LayerNorm.
     *
     * @param row  row index
     * @param mean precomputed mean
     * @return variance of that row's values
     */
    public double rowVariance(int row, double mean) {
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            double diff = data[row][c] - mean;
            sum += diff * diff;
        }
        return sum / cols;
    }

    // ── Structural operations ────────────────────────────────────────────────

    /**
     * Return a single row as a new [1 × cols] Matrix.
     */
    public Matrix getRow(int row) {
        if (row < 0 || row >= rows)
            throw new IndexOutOfBoundsException("Row " + row + " out of bounds for " + shapeString());
        return new Matrix(new float[][]{Arrays.copyOf(data[row], cols)});
    }

    /**
     * Get value at (row, col).
     */
    public float get(int row, int col) {
        return data[row][col];
    }

    /**
     * Set value at (row, col). Use sparingly; prefer functional style.
     */
    public void set(int row, int col, float value) {
        data[row][col] = value;
    }

    /**
     * Create a copy of this Matrix (deep copy of the float[][]).
     */
    public Matrix copy() {
        float[][] copy = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            copy[r] = Arrays.copyOf(data[r], cols);
        return new Matrix(copy);
    }

    // ── Dimension checking ───────────────────────────────────────────────────

    private void checkSameShape(String operation, Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols)
            throw new IllegalArgumentException(
                operation + " dimension mismatch:\n" +
                "  this  = " + this.shapeString() + "\n" +
                "  other = " + other.shapeString());
    }

    // ── Static factory methods ───────────────────────────────────────────────

    /**
     * Create a Matrix filled with a constant value.
     */
    public static Matrix filled(int rows, int cols, float value) {
        float[][] data = new float[rows][cols];
        for (float[] row : data) Arrays.fill(row, value);
        return new Matrix(data);
    }

    /**
     * Create an identity matrix of size n × n.
     */
    public static Matrix identity(int n) {
        float[][] data = new float[n][n];
        for (int i = 0; i < n; i++) data[i][i] = 1.0f;
        return new Matrix(data);
    }

    // ── Object overrides ─────────────────────────────────────────────────────

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Matrix").append(shapeString()).append(":\n");
        for (float[] row : data) {
            sb.append("  [");
            for (int c = 0; c < cols; c++) {
                sb.append(String.format("%8.4f", row[c]));
                if (c < cols - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Matrix)) return false;
        Matrix other = (Matrix) obj;
        if (rows != other.rows || cols != other.cols) return false;
        for (int r = 0; r < rows; r++)
            if (!Arrays.equals(data[r], other.data[r])) return false;
        return true;
    }

    @Override
    public int hashCode() {
        int result = rows * 31 + cols;
        for (float[] row : data) result = result * 31 + Arrays.hashCode(row);
        return result;
    }
}
