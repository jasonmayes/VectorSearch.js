/**
 * A class to handle cosine similarity calculations using Google's TensorFlow.js.
 * Coded by Jason Mayes 2026.
 */
export class CosineSimilarity {
  constructor() {
    this.cachedMatrix = undefined;
  }

  /**
   * Resets the cached GPU matrix.
   */
  async deleteGPUVectorCache() {
    if (this.cachedMatrix) {
      this.cachedMatrix.dispose();
      this.cachedMatrix = undefined;
    }
  }

  /**
   * Calculates cosine similarity between two 1D tensors.
   * @param {tf.Tensor} emb1 First embedding vector.
   * @param {tf.Tensor} emb2 Second embedding vector.
   * @return {Promise<number>} Cosine similarity score.
   */
  async calculateCosineSimilarity(emb1, emb2) {
    // Cosine Similarity = (A . B) / (||A|| * ||B||)
    return tf.tidy(() => {
      // Squeeze for 1D vectors
      const V1 = emb1.squeeze();
      const V2 = emb2.squeeze();
      
      const DOT_PRODUCT = tf.dot(V1, V2);
      const NORM1 = tf.norm(V1);
      const NORM2 = tf.norm(V2);
      
      return DOT_PRODUCT.div(NORM1.mul(NORM2)).dataSync()[0];
    });
  }

  /**
   * Performs matrix-based cosine similarity search on the GPU.
   * @param {Array<Array<number>>} matrixData Array of embedding vectors.
   * @param {Array<number>} vectorData The query embedding vector.
   * @param {number} topK The number of top results to return.
   * @return {Promise<{values: Float32Array, indices: Int32Array}>} Top K similarity scores and their indices.
   */
  async cosineSimilarityTFJSGPUMatrix(matrixData, vectorData, topK) {
    // 1. Convert to Tensors
    if (!this.cachedMatrix) {
      console.log('Rebuilding GPU VectorDB Matrix');
      this.cachedMatrix = tf.tensor2d(matrixData); // Shape: [N, D]
    }
    
    let results = tf.tidy(() => {
      const vector = tf.tensor1d(vectorData); // Shape: [D]

      const epsilon = 1e-9;

      // 2. Normalize the Matrix (Row-wise)
      // l2 norm along the horizontal axis (1)
      const matrixNorms = this.cachedMatrix.norm(2, 1, true); 
      const normalizedMatrix = this.cachedMatrix.div(matrixNorms.add(epsilon));

      // 3. Normalize the Vector
      const vectorNorm = vector.norm(2);
      const normalizedVector = vector.div(vectorNorm.add(epsilon));

      // 4. Compute Dot Product
      // We can use matMul by reshaping vector to [D, 1]
      // [N, D] * [D, 1] -> [N, 1]
      try {
        const dotProduct = tf.matMul(
            normalizedMatrix, 
            normalizedVector.reshape([-1, 1])
        );
      } catch (error) {
        console.error('VectorDB you are trying to use was encoded using embedding model that generated different number of dimensions. Please re-encode DB or use correct Embedding Model', error);
      }
      return dotProduct.squeeze();
    });
    
    const topKResults = tf.topk(results, topK, false);
    const topValues = await topKResults.values.data();
    const topIndices = await topKResults.indices.data();

    return {values: topValues, indices: topIndices};
  }
}
