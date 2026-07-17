# VectorSearch.js
### A library to perform semantic vector search, over millions of vectors in milliseconds, and can even visualize the tokens or embeddings too! Runs entirely client side in the web browser (custom Vector DB layer written on top of IndexDB) and currently supports Google's EmbeddingGemma (highest quality but 300Mb), or all-Mini-L6-v2 (fastest and only 30mb) embedding models via Web AI libraries with WebGPU acceleration for speed.

🦾 As it runs in the browser on YOUR hardware it's totally private, costs zero dollars to use (other than your own electricity), and super low latency. 

🤖 Powered by WebGPU for speed and builds upon no less than 3 popular Web ML Libraries and runtimes to get the best bits of all of them: [LiteRT.js](https://ai.google.dev/edge/litert/web/get_started), [Transformers.js](https://huggingface.co/docs/transformers.js/en/index), and [TensorFlow.js](https://www.tensorflow.org/js).

⭐ Give it a star on Github if you want me to keep evolving the code or have ideas. 

### Show me a demo that works already

Sure [check out my Codepen demo here](https://codepen.io/jasonmayes/pen/JoKMBmq)!

Here's a screen shot of it in action captured in real-time on an NVIDIA 1070 GPU running the larger EmbeddingGemma model:

![Screenshot of VectorSearch.js in action](https://github.com/jasonmayes/VectorSearch.js/blob/main/demo/demo.gif?raw=true)

### Got questions? 
[Reach out to me over on LinkedIn](https://www.linkedin.com/in/webai) or follow for updates on related client side Web AI projects.


## Usage

```javascript
import { VectorSearch } from 'https://cdn.jsdelivr.net/gh/jasonmayes/VectorSearch.js@main/VectorSearch-min.js';

// Embedding Model Configuration.
const MODEL_RUNTIME = 'litertjs'; // OR 'transformersjs'
const MODEL_URL = 'model/embeddinggemma-300M_seq1024_mixed-precision.tflite'; // OR 'Xenova/all-MiniLM-L6-v2' if transformersjs runtime.
const SEQ_LENGTH = 1024;
const TOKENIZER = 'onnx-community/embeddinggemma-300m-ONNX';
const EMBEDDING_MODEL_CONFIG = {
  runtime: MODEL_RUNTIME,
  url: MODEL_URL,
  sequenceLength: SEQ_LENGTH,
  tokenizer: TOKENIZER
};

// Instantiate VectorSearch Master Class.
const VECTOR_SEARCH = new VectorSearch(EMBEDDING_MODEL_CONFIG);

// Initiation and usage example.
async function init(statusDomElement) {
  // Actually load the chosen runtime and model so ready to use.
  await VECTOR_SEARCH.load(statusDomElement);

  await store(['I love Web AI', 'I like cats', 'Dogs are cool too', 'AI rocks', 'Birds can fly', 'Web AI is client side AI', 'Fish can swim', 'Robots are neat', 'JavaScript rocks too!', 'and so on']);
  await find('Likes animals', 0.25);
}

init();


// How to store text in client side VectorDB
async function store(someArrayOfStrings) {
  await VECTOR_SEARCH.storeTexts(someArrayOfStrings, 'DatabaseNameForThisData');
  
  // Optionally can specify callback to write status to a HTML DOM element:
  // await VECTOR_SEARCH.storeTexts(someArrayOfStrings, 'DatabaseNameForThisData', STATUS_EL);
  
  // You can also specify batch size to store array of strings much faster when using all-Mini-L6-v2 model only for now.
  // await VECTOR_SEARCH.storeTexts(someArrayOfStrings, 'DatabaseNameForThisData', undefined, 100);

  // Finally you can also specify embedding task type if using EmbeddingGemma. Can be one of the following strings:
  // document, query, retrieval, search_query, classification, clustering, similarity, code_retrieval
  // Store defaults to 'document' if you do not specify.
  // await VECTOR_SEARCH.storeTexts(someArrayOfStrings, 'DatabaseNameForThisData', undefined, undefined, 'document');
}


// Search example.
async function find(queryText, cosineSimilarityThreshold) {
  const {embedding: EMBEDDING_DATA, tokens: TOKENS} = await VECTOR_SEARCH.getEmbedding(queryText);

  /** Optional: Visualize embeddings and tokens for the search query text.
  if (TOKENS) {
    VECTOR_SEARCH.renderTokens(TOKENS, SOME_DOM_ELEMENT);
  }
  await VECTOR_SEARCH.renderEmbedding(EMBEDDING_DATA, SOME_DOM_ELEMENT_FOR_VISUAL, SOME_DOM_ELEMENT_FOR_TEXT);
  **/

  // Now actually search the vector database.
  const {results: RESULTS, bestScore: BEST_SCORE, bestIndex: BEST_INDEX} = await VECTOR_SEARCH.search(EMBEDDING_DATA, cosineSimilarityThreshold, 'DatabaseNameForThisData');

  if (RESULTS.length > 0) {  
    const BEST_MATCH_VECTOR = RESULTS[BEST_INDEX].vector;
    const BEST_MATCH_SCORE = RESULTS[BEST_INDEX].score;
    const BEST_MATCH_TEXT = RESULTS[BEST_INDEX].text;
    if (BEST_MATCH_TEXT) {
      console.log(BEST_MATCH_SCORE + ': ' + BEST_MATCH_TEXT);
      // Logs: 0.7519992589950562: I like cats.
    }
  } else {
    console.log('No matches found above threshold.');
  }
}
```

## Performance

[I tried to make this as fast as I could](https://www.linkedin.com/posts/webai_rag-litertjs-embeddinggemma-activity-7423026459201523712-IWiD?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE29dSoB2Q5rqrgken9VCQgyG_zQ-gVgvG8). I have tested with 100K vectors on my very old NVIDIA 1070 GPU and it can search those in tens of miliseconds using the more complex EmbeddingGemma model by Google. The largest cost is actually the embedding that takes around 200ms using the EmbeddingGemma model (high quality but large). You may want to swap this out for a leaner embedding model (e.g. all-MiniLM-L6-v2) for the ultimate client side speed of embedding - see example code above for how to change the config object to use that instead.

Currently it is designed to preload the IndexDB vector DB I wrote (yes even the vector DB is client side) into GPU memory to perform as fast as possible when calculating cosine similarity for your target text across all stored vectors. So that means the first search you perform will be slower as it has to transfer memory from CPU to GPU for the first time (suggest doing a dummy vector search on page load to warm up). This also means that it currently takes roughly the SAME time for 100K vectors searched vs 1K vectors due to leveraging the GPU. I have not yet found the upper bound, but there is obviously a limit here, depending on your GPU type, VRAM size etc. I will later need to refactor to load in chunks to avoid any issues for larger vector stores on client side.

I have verified this works on Intel integrated GPUs, NVIDIA, AMD, and Apple M GPUs in any web browser that supports WebGPU (most of them do now).

## Building and serving yourself

To build the minified version of the library from the src folder just run:

```
npm run build
```

Then to serve the demo folder to try it out on your own webserver run:

```
npm run demo
```

Please note that currently script.js in the demo/js folder imports the latest version of VectorSearch-min.js from this Github repo so change the import if you modify anything or want to host somewhere else.

Please also see below for things you need to host yourself to run on your own server.


## Things to be aware of before hosting and running yourself

This project depends on a few things that need to be setup to work.

### EmbeddingGemma model

This repo uses Google's EmbeddingGemma model for the embedding model by default for highest quality. Specifically this one: embeddinggemma-300M_seq1024_mixed-precision.tflite

This model is available to download from HuggingFace which you must do yourself manually:

[Download it yourself from HuggingFace](https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq1024_mixed-precision.tflite?download=true) so any applicable T&C accepted. You can then place this downloaded model into the demo/model folder. If you place it somewhere else update the code in script.js accordingly:

```javascript
// Embedding Model Configuration.
const MODEL_RUNTIME = 'litertjs';
const MODEL_URL = 'model/embeddinggemma-300M_seq1024_mixed-precision.tflite';
const SEQ_LENGTH = 1024;
const TOKENIZER = 'onnx-community/embeddinggemma-300m-ONNX';
const EMBEDDING_MODEL_CONFIG = {
  runtime: MODEL_RUNTIME,
  url: MODEL_URL,
  sequenceLength: SEQ_LENGTH,
  tokenizer: TOKENIZER
};

// Instantiate VectorSearch Master Class.
const VECTOR_SEARCH = new VectorSearch(EMBEDDING_MODEL_CONFIG);
```

For more details [see the model card page on HuggingFace](https://huggingface.co/litert-community/embeddinggemma-300m).

This is a LiteRT.js Web AI compatible EmbeddingGemma model using the tflite model format.

### all-MiniLM-L6-v2 embedding model

If you wish to use the all-MiniLM-L6-v2 embedding model instead for speed you can change the config object to be:

```javascript
// Embedding Model Configuration.
const MODEL_RUNTIME = 'transformersjs';
const MODEL_URL = 'Xenova/all-MiniLM-L6-v2';
const SEQ_LENGTH = 128;
const TOKENIZER = 'onnx-community/embeddinggemma-300m-ONNX';
const EMBEDDING_MODEL_CONFIG = {
  runtime: MODEL_RUNTIME,
  url: MODEL_URL,
  sequenceLength: SEQ_LENGTH,
  tokenizer: TOKENIZER
};

// Instantiate VectorSearch Master Class.
const VECTOR_SEARCH = new VectorSearch(EMBEDDING_MODEL_CONFIG);
```

However please note this model is faster for a few reasons:

1. The input text to be embedded can only be up to 128 tokens vs EmbeddingGemma's 1024 tokens.
2. The vector embedding produced has 384 dimensions vs EmbeddingGemma's 786 dimensions.

### LiteRT.js Wasm files (optional self host)

See the demo folder in this repo that contains a "wasm" sub folder with all the Web Assembly files needed for the LiteRT.js runtime. You can choose to serve these files yourself and update the config object if you do so but remember to enable CORS headers on your server so the files can be used if you do that. If you are curious to learn more about these files see the [official LiteRT.js documentation](https://ai.google.dev/edge/litert/web).

By default the library pulls in these Wasm files from JSDeliver CDN. 

If your hosted version is not in the same location update the config object to specify the new Wasm folder location on your webserver as follows:

```javascript
// Embedding Model Configuration.
const MODEL_RUNTIME = 'litertjs';
const MODEL_URL = 'model/embeddinggemma-300M_seq1024_mixed-precision.tflite';
const SEQ_LENGTH = 1024;
const TOKENIZER = 'onnx-community/embeddinggemma-300m-ONNX';
const EMBEDDING_MODEL_CONFIG = {
  runtime: MODEL_RUNTIME,
  litertjsWasmUrl: '/wasm', // Specify your path to your custom hosted Wasm files here!
  url: MODEL_URL,
  sequenceLength: SEQ_LENGTH,
  tokenizer: TOKENIZER
};

// Instantiate VectorSearch Master Class.
const VECTOR_SEARCH = new VectorSearch(EMBEDDING_MODEL_CONFIG);
```

Note when you call load you can also optionally specify a HTML element to render loading status updates to like this:

```javascript
await VECTOR_SEARCH.load(STATUS_EL);
```

## Shoutouts

This project was made by [Jason Mayes](https://www.linkedin.com/in/webai), and is possible by combining 3 amazing Web AI (client side AI) libraries and runtimes. 

Huge Kudos to:

1. [LiteRT.js](https://ai.google.dev/edge/litert/web/get_started) for the running of Google's EmbeddingGemma model.
2. [Transformers.js](https://huggingface.co/docs/transformers.js/en/index) for the running of the tokenizer.
3. [TensorFlow.js](https://www.tensorflow.org/js) for the WebGPU accelerated mathematics (yes Machine Learning libraries can be used to do Maths!) along with the pre/post processing of any Tensors that go into or come out of LiteRT.js for speed.
