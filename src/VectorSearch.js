import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.js';
import * as LiteRT from 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/+esm';

import { VectorStore } from './VectorStore.js';
import { CosineSimilarity } from './CosineSimilarity.js';
import { EmbeddingModel } from './EmbeddingModel.js';
import { Tokenizer } from './Tokenizer.js';
import { VisualizeTokens } from './VisualizeTokens.js';
import { VisualizeEmbedding } from './VisualizeEmbedding.js';

/**
 * VectorSearch - A master orchestrator class for the RAG system.
 * Coded by Jason Mayes 2026.
 */
export class VectorSearch {
  /**
   * @param {string} modelConfig Config object for VectorSearch setup.
   */
  constructor(modelConfig) {
    this.modelUrl = modelConfig.url;
    this.modelRuntime = modelConfig.runtime;
    this.litertHostedWasmUrl = modelConfig.litertjsWasmUrl ? modelConfig.litertjsWasmUrl : 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/wasm/';
    this.tokenizerId = modelConfig.tokenizer;
    this.seqLength = modelConfig.sequenceLength;

    this.vectorStore = new VectorStore();
    this.cosineSimilarity = new CosineSimilarity();
    this.embeddingModel = new EmbeddingModel(this.modelRuntime);
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
  async load(STATUS_EL) {
    if (STATUS_EL) {
      STATUS_EL.innerText = 'Setting WebGPU Backend for TFJS...';
    }
    await tf.setBackend('webgpu');
    
    if (STATUS_EL) {
      STATUS_EL.innerText = 'Initializing Model Runtime...';
    }

    if (this.modelRuntime === 'litertjs') {
      const LITERTJS_WASM_PATH = this.litertHostedWasmUrl;
      await LiteRT.loadLiteRt(LITERTJS_WASM_PATH);
      const TF_BACKEND = tf.backend();
      LiteRT.setWebGpuDevice(TF_BACKEND.device);
    }

    if (STATUS_EL) {
      STATUS_EL.innerText = 'Loading Tokenizer & Embedding Model...';
    }
    await this.embeddingModel.load(this.modelUrl, this.modelRuntime);
    
    if (STATUS_EL) {
      STATUS_EL.innerText = 'Loading Tokenizer...';
    }
    if (this.modelRuntime === 'litertjs') {
      await this.tokenizer.load(this.tokenizerId);
    }
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
   * @return {Promise<{embedding: Array<number>, tokens: Array<number>}>}
   */
  async getEmbedding(text) {
    if (this.modelRuntime === 'litertjs') {
      const tokens = await this.tokenizer.encode(text);
      const { embedding } = await this.embeddingModel.getEmbeddingLiteRTJS(tokens, this.seqLength);
      const result = await embedding.array();
      embedding.dispose();
      return { embedding: result[0], tokens };
    } else {
      // Transformers.js (no tokens returned).
      const { embedding } = await this.embeddingModel.getEmbeddingTransformers(text);
      return { embedding: embedding };
    }
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
   * @param {Array<number>} data
   * @param {HTMLElement} vizEl
   * @param {HTMLElement} textEl
   */
  async renderEmbedding(data, vizEl, textEl) {
    await this.visualizeEmbedding.render(data, vizEl, textEl);
  }

  /**
   * Deletes the GPU vector cache.
   */
  async deleteGPUVectorCache() {
    await this.cosineSimilarity.deleteGPUVectorCache();
  }

  /**
   * Performs a vector search.
   * @param {Array<number>} queryVector
   * @param {number} threshold
   * @param {string} selectedDB
   * @param {number} maxMatches
   * @return {Promise<{results: Array<Object>, bestScore: number, bestIndex: number}>}
   */
  async search(queryVector, threshold, selectedDB, maxMatches = 5) {
    let matrixData = undefined;

    if (this.lastDBName !== selectedDB) {
      await this.deleteGPUVectorCache();
      this.lastDBName = selectedDB;
      this.allStoredData = await this.vectorStore.getAllVectors();
      matrixData = this.allStoredData.map(item => item.embedding);
    } else {
      matrixData = this.allStoredData.map(item => item.embedding);
    }

    if (matrixData.length === 0)  {
      console.warn('No data in chosen vector store. Store some data first before searching');
      return { results: [], bestScore: 0, bestIndex: 0 };
    }
    const { values, indices } = await this.cosineSimilarity.cosineSimilarityTFJSGPUMatrix(matrixData, queryVector, maxMatches);
    
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

  /**
   * Embeds and stores multiple texts.
   * @param {Array<string>} texts
   * @param {string} dbName
   * @param {Function} progressCallback
   */
  async storeTexts(texts, dbName, statusElement, batchSize = 2) {
    this.setDb(dbName);
    let textBatch = [];
    let tensorBatch = [];
    
    for (let i = 0; i < texts.length; i++) {
      if (statusElement) {
        statusElement.innerText = `Embedding paragraph ${i + 1} of ${texts.length}...`;
      }
      
      if (this.modelRuntime === 'litertjs') {
        const tokens = await this.tokenizer.encode(texts[i]);
        const { embedding } = await this.embeddingModel.getEmbeddingLiteRTJS(tokens, this.seqLength);
        tensorBatch.push(embedding);
        textBatch.push(texts[i]);
      
        if (tensorBatch.length >= batchSize || i === texts.length - 1) {
          const stackedTensors = tf.stack(tensorBatch);
          const allVectors = await stackedTensors.array();
        
          const storagePayload = allVectors.map((vector, index) => ({
            embedding: vector[0],
            text: textBatch[index]
          }));

          await this.vectorStore.storeBatch(storagePayload);

          tensorBatch.forEach(t => t.dispose());
          stackedTensors.dispose();
          tensorBatch = [];
          textBatch = [];
        }
      } else {
        // Using Transformers.js model.
        const { embedding } = await this.embeddingModel.getEmbeddingTransformers(texts[i]);
        const storagePayload = {
          embedding: embedding,
          text: texts[i]
        };

        await this.vectorStore.storeBatch([storagePayload]);
      }
    }
    await this.deleteGPUVectorCache();
  }
}
