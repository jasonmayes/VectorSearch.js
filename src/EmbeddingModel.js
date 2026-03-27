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
    let dataUrl = await FileProxyCache.loadFromURL(fileName);
    
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
   * Generates an embedding for the given tokens.
   * @param {Array<number>} tokens Array of token IDs.
   * @param {number} seqLength Expected sequence length for the model.
   * @return {Promise<{embedding: tf.Tensor, tokens: Array<number>}>} The generated embedding tensor.
   */
  async getEmbeddingLiteRTJS(tokens, seqLength) {
    if (!this.model) {
      throw new Error('Model not loaded. Call load() first.');
    }
    
    if (this.runtime === 'litertjs') {
      let inputTensor = tf.tensor1d(tokens, 'int32');
    
      // Ensure to fill to expected model token length else trim.
      if (tokens.length < seqLength) {
        inputTensor = inputTensor.pad([[0, seqLength - tokens.length]]);
      } else if (tokens.length > seqLength) {
        inputTensor = inputTensor.slice([0], [seqLength]);
      }
    
      const EXPANDED_INPUT = inputTensor.expandDims(0);
      const RESULTS = LiteRTInterop.runWithTfjsTensors(this.model, EXPANDED_INPUT);

      inputTensor.dispose();
      EXPANDED_INPUT.dispose();
      return {
        embedding: RESULTS[0], // Returns batch of 1.
        tokens: tokens
      };
    }
  }

  /**
   * Generates an embedding for the given tokens.
   * @param {string|Array<string>} textBatch Text or array of text to embed.
   * @return {Promise<{embedding: Array<number>}>} The generated embedding.
   */
  async getEmbeddingTransformers(textBatch) {
    if (this.runtime === 'transformersjs') {
      const queryResult = await this.model(textBatch, { pooling: 'mean', normalize: true });
      const queryVector = Array.from(queryResult.data);
      return {
        embedding: queryVector
      };
    }
  }
}
