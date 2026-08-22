/**
 * rx.M — Amazon RXVM Matrix Math Module
 * Extracted from production Amazon page source, deobfuscated.
 *
 * CONFIRMED: rx.M is a metadata-only pass-through. It does NOT
 * rescale, dequantize, or transform weight values. The 8-bit
 * minifloat weights pass through unchanged.
 *
 * Original (minified):
 *   (i=>{let n=i.Math;function e(){for(let t=0;t<this.length;t+=1)this[t]=n.max(0,this[t])}function o(){for(let t=0;t<this.length;t+=1)this[t]=1/(1+n.exp(0-this[t]))}function r(t,n){var r=new i.Float32Array(t*n);return r.rows=t,r.columns=n,r.R=e,r.S=o,r}i.rx.M=function(t,n,r){return t.rows=n,t.columns=r,t.R=e,t.S=o,t},i.rx.M.zero=r,i.rx.M.mul=function(e,o,t){var n=e.rows,f=e.columns,u=o.columns,l=r(n,u);for(let i=0;i<n;i+=1)for(let r=0;r<u;r+=1){let n=t[r];for(let t=0;t<f;t+=1)n+=e[i*f+t]*o[t*u+r];l[i*u+r]=n}return l}})(window);
 */

// ReLU activation — applied in-place on typed arrays
// Called as: matrix.R()
function R() {
    for (let i = 0; i < this.length; i += 1)
        this[i] = Math.max(0, this[i]);
}

// Sigmoid activation — applied in-place on typed arrays
// Called as: matrix.S()
function S() {
    for (let i = 0; i < this.length; i += 1)
        this[i] = 1 / (1 + Math.exp(0 - this[i]));
}

// Create a zero-initialized matrix as a Float32Array with metadata
// M.zero(rows, cols) → Float32Array with .rows, .columns, .R, .S
rx.M.zero = function(rows, cols) {
    var mat = new Float32Array(rows * cols);
    mat.rows = rows;
    mat.columns = cols;
    mat.R = R;
    mat.S = S;
    return mat;
};

// Annotate an existing array with matrix metadata
// DOES NOT COPY OR TRANSFORM THE DATA — pure metadata attachment
// M(array, rows, cols) → same array with .rows, .columns, .R, .S
rx.M = function(array, rows, cols) {
    array.rows = rows;
    array.columns = cols;
    array.R = R;
    array.S = S;
    return array;  // returns the SAME array, unmodified values
};

// Matrix multiply with bias: result = input × weights + bias
// Standard row-major matmul: result[i][r] = bias[r] + Σ(input[i][t] × weights[t][r])
//
// Called by Dense.forward as: M.mul(input_matrix, weight_matrix, bias_vector)
// For single-sample inference (rows=1), reduces to:
//   output[r] = bias[r] + Σ(input[t] × weights[t * outCols + r])
rx.M.mul = function(input, weights, bias) {
    var rows = input.rows;         // input rows (1 for single sample)
    var inCols = input.columns;    // input columns = weight rows
    var outCols = weights.columns; // weight columns = output dimensions
    var result = rx.M.zero(rows, outCols);

    for (let i = 0; i < rows; i += 1)
        for (let r = 0; r < outCols; r += 1) {
            let sum = bias[r];
            for (let t = 0; t < inCols; t += 1)
                sum += input[i * inCols + t] * weights[t * outCols + r];
            result[i * outCols + r] = sum;
        }

    return result;
};
