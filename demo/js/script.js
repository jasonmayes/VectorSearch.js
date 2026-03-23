import { VectorSearch } from 'https://cdn.jsdelivr.net/gh/jasonmayes/VectorSearch.js@main/VectorSearch-min.js';

// DOM references.
const DB_NAME_INPUT = document.getElementById('db-name-input');
const DB_SELECT = document.getElementById('db-select');
const STATUS_EL = document.getElementById('status');
const QUERY_EMBEDDING_TEXT = document.getElementById('query-embedding-text');
const QUERY_TOKENS_OUTPUT = document.getElementById('query-tokens-output');
const QUERY_EMBEDDING_VIZ = document.getElementById('query-embedding-viz');
const BEST_MATCH_EMBEDDING_VIZ = document.getElementById('best-match-embedding-viz');
const BEST_MATCH_EMBEDDING_TEXT = document.getElementById('best-match-embedding-text');
const INPUT_TEXT = document.getElementById('input-text');
const TARGET_TEXT = document.getElementById('target-text');
const STORE_BTN = document.getElementById('store-btn');
const PREDICT_BTN = document.getElementById('predict-btn');
const THRESHOLD_INPUT = document.getElementById('threshold-input');
const THRESHOLD_VALUE = document.getElementById('threshold-value');
const RESULTS_TEXT = document.getElementById('results-text');
const SIMILARITY_CONTAINER = document.getElementById('similarity-container');
const SIMILARITY_SCORE_EL = document.getElementById('similarity-score');
const SIMILARITY_LABEL_EL = document.getElementById('similarity-label');
const API_TRACE_OUTPUT = document.getElementById('api-trace-output');
const COPY_API_EXAMPLE_BTN = document.getElementById('copy-api-example-btn');


// Configuration.
const MODEL_URL = 'model/embeddinggemma-300M_seq1024_mixed-precision.tflite';
const TOKENIZER_ID = 'onnx-community/embeddinggemma-300m-ONNX';
const SEQ_LENGTH = 1024;
// Instantiate VectorSearch Master Class.
const VECTOR_SEARCH = new VectorSearch(MODEL_URL, TOKENIZER_ID, SEQ_LENGTH);
const TRACE_STATE = {
  store: '',
  search: ''
};


function quoteForCode(value) {
  return JSON.stringify(value);
}


function formatThresholdValue(value) {
  return Number.isFinite(value) ? parseFloat(value.toFixed(2)).toString() : '0.4';
}


function renderApiTrace() {
  const sections = [
    '// Initialization',
    'const vectorSearch = new VectorSearch(MODEL_URL, TOKENIZER_ID, SEQ_LENGTH);',
    "await vectorSearch.load('wasm/', statusEl);"
  ];

  if (TRACE_STATE.store) {
    sections.push('', '// Latest store call', TRACE_STATE.store);
  }

  if (TRACE_STATE.search) {
    sections.push('', '// Latest search call', TRACE_STATE.search);
  }

  API_TRACE_OUTPUT.textContent = sections.join('\n');
}


function updateStoreTrace(paragraphCount, dbName) {
  TRACE_STATE.store = `await vectorSearch.storeTexts(paragraphs /* ${paragraphCount} chunks */, ${quoteForCode(dbName)}, statusEl);`;
  renderApiTrace();
}


function updateSearchTrace(queryText, threshold, selectedDB, includeBestMatch = false) {
  const lines = [
    `vectorSearch.setDb(${quoteForCode(selectedDB)});`,
    `const { embedding, tokens } = await vectorSearch.getEmbedding(${quoteForCode(queryText)});`,
    'vectorSearch.renderTokens(tokens, queryTokensEl);',
    'await vectorSearch.renderEmbedding(embedding, queryEmbeddingVizEl, queryEmbeddingTextEl);',
    `const { results, bestScore, bestIndex } = await vectorSearch.search(embedding, ${formatThresholdValue(threshold)}, ${quoteForCode(selectedDB)});`
  ];

  if (includeBestMatch) {
    lines.push('await vectorSearch.renderEmbedding(results[bestIndex].vector, bestMatchEmbeddingVizEl, bestMatchEmbeddingTextEl);');
  }

  TRACE_STATE.search = lines.join('\n');
  renderApiTrace();
}


function buildStarterExample() {
  const dbName = DB_NAME_INPUT.value.trim() || 'MyVectorDB';
  const query = TARGET_TEXT.value.trim() || 'What should I search for?';
  const threshold = formatThresholdValue(parseFloat(THRESHOLD_INPUT.value) || 0.4);

  return `import { VectorSearch } from 'https://cdn.jsdelivr.net/gh/jasonmayes/VectorSearch.js@main/VectorSearch-min.js';

const MODEL_URL = ${quoteForCode(MODEL_URL)};
const TOKENIZER_ID = ${quoteForCode(TOKENIZER_ID)};
const SEQ_LENGTH = ${SEQ_LENGTH};
const DB_NAME = ${quoteForCode(dbName)};
const QUERY = ${quoteForCode(query)};
const THRESHOLD = ${threshold};

const vectorSearch = new VectorSearch(MODEL_URL, TOKENIZER_ID, SEQ_LENGTH);
await vectorSearch.load('wasm/');

const paragraphs = [
  'First paragraph to index',
  'Second paragraph to index'
];

await vectorSearch.storeTexts(paragraphs, DB_NAME);

vectorSearch.setDb(DB_NAME);
const { embedding, tokens } = await vectorSearch.getEmbedding(QUERY);

// Optional visualizers if you have DOM elements available:
// vectorSearch.renderTokens(tokens, queryTokensEl);
// await vectorSearch.renderEmbedding(embedding, queryEmbeddingVizEl, queryEmbeddingTextEl);

const { results, bestScore, bestIndex } = await vectorSearch.search(
  embedding,
  THRESHOLD,
  DB_NAME
);

console.log({ results, bestScore, bestIndex });`;
}


function setCopyButtonLabel(label) {
  COPY_API_EXAMPLE_BTN.innerText = label;
  window.setTimeout(() => {
    COPY_API_EXAMPLE_BTN.innerText = 'Copy Starter Example';
  }, 1600);
}


async function copyStarterExample() {
  const example = buildStarterExample();

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(example);
    } else {
      const TEXTAREA = document.createElement('textarea');
      TEXTAREA.value = example;
      TEXTAREA.setAttribute('readonly', '');
      TEXTAREA.style.position = 'absolute';
      TEXTAREA.style.left = '-9999px';
      document.body.appendChild(TEXTAREA);
      TEXTAREA.select();
      document.execCommand('copy');
      document.body.removeChild(TEXTAREA);
    }
    setCopyButtonLabel('Copied');
  } catch (e) {
    console.error('Failed to copy starter example:', e);
    setCopyButtonLabel('Copy failed');
  }
}


async function predictBtnClickHandler() {
  const QUERY_TEXT_VALUE = TARGET_TEXT.value;
  const THRESHOLD = parseFloat(THRESHOLD_INPUT.value) || 0.5;
  const SELECTED_DB = DB_SELECT.value;

  if (QUERY_TEXT_VALUE && SELECTED_DB) {
    VECTOR_SEARCH.setDb(SELECTED_DB);
    PREDICT_BTN.disabled = true;
    STATUS_EL.innerText = `Searching VectorDB (${SELECTED_DB})...`;
    const t0 = performance.now();
    await predict(QUERY_TEXT_VALUE, THRESHOLD);
    const t1 = performance.now();
    console.log(`Total search time (query embedding + vector search) took ${t1 - t0} milliseconds.`);
    STATUS_EL.innerText = 'Search complete';
    PREDICT_BTN.disabled = false;
  }
}


async function storeBtnClickHandler() {
  const text = INPUT_TEXT.value.trim();
  const dbName = DB_NAME_INPUT.value.trim();
  if (!text || !dbName) return;

  STORE_BTN.disabled = true;
  
  const paragraphs = text.split(/\n\s*\n/).map(p => p.trim()).filter(p => p.length > 0);
  updateStoreTrace(paragraphs.length, dbName);
  
  await VECTOR_SEARCH.storeTexts(paragraphs, dbName, STATUS_EL);

  STATUS_EL.innerText = `Stored ${paragraphs.length} paragraphs.`;
  STORE_BTN.disabled = false;
  INPUT_TEXT.value = '';
  
  await updateDbList();
} 


async function load() {
  try {
    await updateDbList();

    await VECTOR_SEARCH.load('wasm/', STATUS_EL);

    STATUS_EL.innerText = 'Ready to store and search';
    STORE_BTN.disabled = false;
    PREDICT_BTN.disabled = false;

    STORE_BTN.addEventListener('click', storeBtnClickHandler);
    PREDICT_BTN.addEventListener('click', predictBtnClickHandler);
    THRESHOLD_INPUT.addEventListener('input', () => {
      THRESHOLD_VALUE.innerText = THRESHOLD_INPUT.value;
    });
  } catch (e) {
    console.error(e);
    STATUS_EL.innerText = 'Error: ' + e.message;
  }
}


async function predict(queryText, threshold) {
  updateSearchTrace(queryText, threshold, DB_SELECT.value);

  // Visualize embeddings and tokens for the search query text.
  const { embedding: EMBEDDING_DATA, tokens: TOKENS } = await VECTOR_SEARCH.getEmbedding(queryText);
  VECTOR_SEARCH.renderTokens(TOKENS, QUERY_TOKENS_OUTPUT);
  await VECTOR_SEARCH.renderEmbedding(EMBEDDING_DATA, QUERY_EMBEDDING_VIZ, QUERY_EMBEDDING_TEXT);
  
  // Now actually search the vector database.
  const { results: RESULTS, bestScore: BEST_SCORE, bestIndex: BEST_INDEX } = await VECTOR_SEARCH.search(EMBEDDING_DATA, threshold, DB_SELECT.value);

  if (RESULTS.length > 0) {
    RESULTS_TEXT.value = RESULTS.map(m => `[Score: ${m.score.toFixed(4)}]\n${m.text}`).join('\n\n');
    updateSimilarityUI(BEST_SCORE);
    
    const bestMatchVector = RESULTS[BEST_INDEX].vector;
    if (bestMatchVector) {
      updateSearchTrace(queryText, threshold, DB_SELECT.value, true);
      await VECTOR_SEARCH.renderEmbedding(bestMatchVector, BEST_MATCH_EMBEDDING_VIZ, BEST_MATCH_EMBEDDING_TEXT);
    }
  } else {
    RESULTS_TEXT.value = "No matches found above threshold.";
    SIMILARITY_CONTAINER.classList.add('hidden');
    BEST_MATCH_EMBEDDING_VIZ.innerHTML = '';
    BEST_MATCH_EMBEDDING_TEXT.innerText = '';
  }
}


function updateSimilarityUI(score) {
  SIMILARITY_CONTAINER.classList.remove('hidden');
  SIMILARITY_SCORE_EL.innerText = score.toFixed(4);
  
  const HUE = Math.max(0, Math.min(120, score * 120));
  const BACKGROUND_COLOUR = `hsla(${HUE}, 70%, 20%, 0.4)`;
  const BORDER_COLOUR = `hsla(${HUE}, 70%, 50%, 0.6)`;
  
  SIMILARITY_CONTAINER.style.backgroundColor = BACKGROUND_COLOUR;
  SIMILARITY_CONTAINER.style.borderColor = BORDER_COLOUR;
  
  let label = 'Low Similarity';
  if (score > 0.8) {
    label = 'Very High Similarity';
  } else if (score > 0.6) {
    label = 'High Similarity';
  } else if (score > 0.4) {
    label = 'Moderate Similarity';
  }
  
  SIMILARITY_LABEL_EL.innerText = label;
}


async function updateDbList() {
  if (!window.indexedDB.databases) {
    console.warn('indexedDB.databases() is not supported in this browser.');
    return;
  }

  try {
    const dbs = await window.indexedDB.databases();
    const currentSelection = DB_SELECT.value;
    
    DB_SELECT.innerHTML = '';
    const currentInputName = DB_NAME_INPUT.value.trim();
    let names = dbs.map(db => db.name).filter(name => name !== undefined);
    
    if (currentInputName && !names.includes(currentInputName)) {
      names.push(currentInputName);
    }
    
    names.sort();
    
    names.forEach(name => {
      const option = document.createElement('option');
      option.value = name;
      option.text = name;
      if (name === currentSelection || (currentSelection === '' && name === currentInputName)) {
        option.selected = true;
      }
      DB_SELECT.appendChild(option);
    });
  } catch (e) {
    console.error('Error fetching databases:', e);
  }
}

renderApiTrace();
COPY_API_EXAMPLE_BTN.addEventListener('click', copyStarterExample);
load();
