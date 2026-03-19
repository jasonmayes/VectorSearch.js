# VectorSearch.js
Client side vector search using EmbeddingGemma with Web AI (LiteRT.js, TensorFlow.js, and Transformers.js)

## Usage

To build the minified version of the library from the src folder just run:

```
npm run build
```

Then to serve the demo folder to try it out on your own webserver run:

```
npm run demo
```

Please note that currently script.js in the dmeo/js folder imports the latest version of VectorSearch-min.js from this Github repo so change the import if you modify anything or want to host somewhere else.

Please also see below for things you need to host yourself to run on your own server.


## Things to be aware of before hosting and running yourself

This project depends on a few things that need to be setup to work.

### LiteRT.js Wasm files required

See the demo folder in this repo that contains a "wasm" sub folder with all the Web Assembly files needed for the LiteRT.js runtime. You will need to serve these files yourself to use the library. If you are curious to learn more about these files see the [official LiteRT.js documentation](https://ai.google.dev/edge/litert/web).

By default the library assumes this "wasm" folder exists in the www root at "wasm/". 

If your hosted version is not in the same location update the call to VECTOR_SEARCH.load() to specify the new Wasm folder location on your webserver as follows:

```javascript
await VECTOR_SEARCH.load('wasm/');
```

Note when you call load you can also optionally specify a HTML element to render loading status updates to like this:

```javascript
await VECTOR_SEARCH.load('wasm/', STATUS_EL);
```

### EmbeddingGemma model

This repo uses Google's EmbeddingGemma model for the embedding model. Specifically this one: embeddinggemma-300M_seq1024_mixed-precision.tflite

This model is available to download from HuggingFace which you must do yourself manually:

[Download it yourself from HuggingFace](https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq1024_mixed-precision.tflite?download=true) so any applicable T&C accepted. You can then place this downloaded model into the demo/model folder. If you place it somewhere else update the code in script.js accordingly:

```javascript
const MODEL_URL = 'model/embeddinggemma-300M_seq1024_mixed-precision.tflite';
```

For more details [see the model card page on HuggingFace](https://huggingface.co/litert-community/embeddinggemma-300m).

This is a LiteRT.js Web AI compatible EmbeddingGemma model using the tflite model format.

## Shoutouts

This project was made by [Jason Mayes](https://www.linkedin.com/in/webai), and is possible by combining 3 amazing Web AI (client side AI) libraries and runtimes. 

Huge Kudos to:

1. [LiteRT.js](https://ai.google.dev/edge/litert/web/get_started) for the running of Google's EmbeddingGemma model.
2. [Transformers.js](https://huggingface.co/docs/transformers.js/en/index) for the running of the tokenizer.
3. [TensorFlow.js](https://www.tensorflow.org/js) for the WebGPU accelerated mathematics (yes Machine Learning libraries can be used to do Maths!) along with the pre/post processing of any Tensors that go into or come out of LiteRT.js for speed.
