import FileProxyCache from 'https://cdn.jsdelivr.net/gh/jasonmayes/web-ai-model-proxy-cache@main/FileProxyCache.min.js';
import * as LiteRT from 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/+esm';
import * as LiteRTInterop from 'https://cdn.jsdelivr.net/npm/@litertjs/tfjs-interop@1.0.1/+esm';
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

/**
 * A class to handle loading and using the EmbeddingGemma model with LiteRT.
 * Coded by Jason Mayes 2026.
 */
export class EmbeddingModel {
  constructor(modelRuntime) {
    this.model = undefined;
    this.runtime = modelRuntime;
  }

  /**
   * Loads and compiles the LiteRT model.
   * @param {string} modelUrl URL to the .tflite model file.
   * @return {Promise<void>}
   */
  async load(modelUrl) {
    // LiteRT initialization needs to happen before loading.
    // In this refactor, we assume LiteRT.loadLiteRt and setWebGpuDevice are handled 
    // in the main script or within this load method if needed.
    // However, the user asked to keep the logic similar, so we'll just handle 
    // model loading and compilation here.
    let dataUrl = await FileProxyCache.loadFromURL(modelUrl);
    
    if (this.runtime === 'litertjs') {
      this.model = await LiteRT.loadAndCompile(dataUrl, {
        accelerator: 'webgpu',
      });
    } else {
      // Transformers.js model.
      // Load the feature-extraction pipeline
      this.model = await pipeline('feature-extraction', dataUrl);
    }
  }

  /**
   * Returns the instruction prefix for the given task type.
   * @param {string|Object} [taskType='query'] The task type or task options.
   * @param {string} [title='none'] The document title (only used for 'document' task type).
   * @return {string} The prefix to prepend.
   */
  getTaskPrefix(taskType = 'query', title = 'none') {
    let task = 'query';
    let docTitle = title;
    if (typeof taskType === 'object' && taskType !== null) {
      task = taskType.type || 'query';
      docTitle = taskType.title || 'none';
    } else if (typeof taskType === 'string') {
      task = taskType;
    }
    task = task.toLowerCase();

    if (task === 'document' || task === 'search_document') {
      return `title: ${docTitle} | text: `;
    }

    const taskDescriptions = {
      query: 'search result',
      retrieval: 'search result',
      search_query: 'search result',
      classification: 'classification',
      clustering: 'clustering',
      similarity: 'sentence similarity',
      code_retrieval: 'code retrieval',
    };

    const desc = taskDescriptions[task] || 'search result';
    return `task: ${desc} | query: `;
  }

  /**
   * Generates an embedding for the given tokens.
   * @param {Array<number>} tokens Array of token IDs.
   * @param {number} seqLength Expected sequence length for the model.
   * @param {string|Object} [taskType='query'] The task type or task options.
   * @param {Object} [tokenizer=null] Optional tokenizer.
   * @param {string} [rawText=null] Optional raw text.
   * @return {Promise<{embedding: tf.Tensor, tokens: Array<number>}>} The generated embedding tensor.
   */
  async getEmbeddingLiteRTJS(tokens, seqLength, taskType = 'query', tokenizer = null, rawText = null) {
    if (!this.model) {
      throw new Error('Model not loaded. Call load() first.');
    }
    
    if (this.runtime === 'litertjs') {
      let finalTokens = tokens;
      if (tokenizer && rawText !== null) {
        const prefix = this.getTaskPrefix(taskType);
        finalTokens = await tokenizer.encode(prefix + rawText);
      }

      let inputTensor = tf.tensor1d(finalTokens, 'int32');
    
      // Ensure to fill to expected model token length else trim.
      if (finalTokens.length < seqLength) {
        inputTensor = inputTensor.pad([[0, seqLength - finalTokens.length]]);
      } else if (finalTokens.length > seqLength) {
        inputTensor = inputTensor.slice([0], [seqLength]);
      }
    
      const EXPANDED_INPUT = inputTensor.expandDims(0);
      const RESULTS = LiteRTInterop.runWithTfjsTensors(this.model, EXPANDED_INPUT);

      inputTensor.dispose();
      EXPANDED_INPUT.dispose();
      return {
        embedding: RESULTS[0], // Returns batch of 1.
        tokens: finalTokens
      };
    }
  }

  /**
   * Generates an embedding for the given tokens.
   * @param {string|Array<string>} textBatch Text or array of text to embed.
   * @param {string|Object} [taskType='query'] The task type or task options.
   * @return {Promise<{embedding: Array<number>}>} The generated embedding.
   */
  async getEmbeddingTransformers(textBatch, taskType = 'query') {
    if (this.runtime === 'transformersjs') {
      const prefix = this.getTaskPrefix(taskType);
      let formattedBatch;
      if (Array.isArray(textBatch)) {
        formattedBatch = textBatch.map(text => prefix + text);
      } else {
        formattedBatch = prefix + textBatch;
      }
      const queryResult = await this.model(formattedBatch, { pooling: 'mean', normalize: true });
      const queryVector = Array.from(queryResult.data);
      return {
        embedding: queryVector
      };
    }
  }
}
