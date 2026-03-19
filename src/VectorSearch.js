import { VectorStore } from '/src/VectorStore.js';
import { CosineSimilarity } from '/src/CosineSimilarity.js';
import { EmbeddingModel } from '/src/EmbeddingModel.js';
import { Tokenizer } from '/src/Tokenizer.js';
import { VisualizeTokens } from '/src/VisualizeTokens.js';
import { VisualizeEmbedding } from '/src/VisualizeEmbedding.js';

/**
 * VectorSearch - A master orchestrator class for the RAG system.
 * Coded by Jason Mayes 2026.
 */
export class VectorSearch {
  /**
   * @param {string} modelUrl URL to the LiteRT model.
   * @param {string} tokenizerId ID for the Transformers.js tokenizer.
   * @param {number} seqLength Expected sequence length for the model.
   */
  constructor(modelUrl, tokenizerId, seqLength) {
    this.modelUrl = modelUrl;
    this.tokenizerId = tokenizerId;
    this.seqLength = seqLength;

    this.vectorStore = new VectorStore();
    this.cosineSimilarity = new CosineSimilarity();
    this.embeddingModel = new EmbeddingModel();
    this.tokenizer = new Tokenizer();
    this.visualizeTokens = new VisualizeTokens();
    this.visualizeEmbedding = new VisualizeEmbedding();

    this.allStoredData = undefined;
    this.lastDBName = '';
  }

  /**
   * Initializes the embedding model and tokenizer.
   * @return {Promise<void>}
   */
  async load() {
    await this.embeddingModel.load(this.modelUrl);
    await this.tokenizer.load(this.tokenizerId);
  }

  /**
   * Sets the current database name in the vector store.
   * @param {string} dbName
   */
  setDb(dbName) {
    this.vectorStore.setDb(dbName);
  }

  /**
   * Encodes text and generates an embedding.
   * @param {string} text
   * @return {Promise<{embedding: tf.Tensor, tokens: Array<number>}>}
   */
  async getEmbedding(text) {
    const tokens = await this.tokenizer.encode(text);
    const { embedding } = await this.embeddingModel.getEmbedding(tokens, this.seqLength);
    return { embedding, tokens };
  }

  /**
   * Renders tokens in the given container.
   * @param {Array<number>} tokens
   * @param {HTMLElement} containerEl
   */
  renderTokens(tokens, containerEl) {
    this.visualizeTokens.render(tokens, containerEl, this.seqLength);
  }

  /**
   * Renders an embedding visualization.
   * @param {tf.Tensor} tensor
   * @param {HTMLElement} vizEl
   * @param {HTMLElement} textEl
   */
  async renderEmbedding(tensor, vizEl, textEl) {
    await this.visualizeEmbedding.render(tensor, vizEl, textEl);
  }

  /**
   * Deletes the GPU vector cache.
   */
  async deleteGPUVectorCache() {
    await this.cosineSimilarity.deleteGPUVectorCache();
  }

  /**
   * Performs a vector search.
   * @param {tf.Tensor} queryEmbedding
   * @param {number} threshold
   * @param {string} selectedDB
   * @param {number} maxMatches
   * @return {Promise<{results: Array<Object>, bestScore: number, bestIndex: number}>}
   */
  async search(queryEmbedding, threshold, selectedDB, maxMatches = 10) {
    const QUERY_VECTOR = Array.from(await queryEmbedding.data());
    let matrixData = undefined;

    if (this.lastDBName !== selectedDB) {
      await this.deleteGPUVectorCache();
      this.lastDBName = selectedDB;
      this.allStoredData = await this.vectorStore.getAllVectors();

      if (this.allStoredData.length === 0) {
        return { results: [], bestScore: 0, bestIndex: 0 };
      }
      matrixData = this.allStoredData.map(item => item.embedding);
    } else {
      matrixData = this.allStoredData.map(item => item.embedding);
    }

    const { values, indices } = await this.cosineSimilarity.cosineSimilarityTFJSGPUMatrix(matrixData, QUERY_VECTOR, maxMatches);
    
    let topMatches = [];
    let bestIndex = 0;
    let bestScore = 0;

    for (let i = 0; i < values.length; i++) {
      if (values[i] >= threshold) {
        if (topMatches.length < maxMatches) {
          topMatches.push({
            id: this.allStoredData[indices[i]].id,
            score: values[i],
            vector: this.allStoredData[indices[i]].embedding
          });
          if (values[i] > bestScore) {
            bestIndex = topMatches.length - 1;
            bestScore = values[i];
          }
        }
      }
    }

    const results = [];
    for (const match of topMatches) {
      const text = await this.vectorStore.getTextByID(match.id);
      results.push({ ...match, text });
    }

    return { results, bestScore, bestIndex };
  }

  /**
   * Stores multiple items in the vector store.
   * @param {Array<{embedding: Array<number>, text: string}>} storagePayload
   */
  async storeBatch(storagePayload) {
    await this.vectorStore.storeBatch(storagePayload);
  }
}
