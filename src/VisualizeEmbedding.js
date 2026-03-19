/**
 * A class to handle visual representation of embeddings in the DOM.
 * Coded by Jason Mayes 2026.
 */
export class VisualizeEmbedding {
  /**
   * Renders the given data as a heatmap and its full vector text.
   * @param {tf.Tensor|Array} data The embedding data to visualize.
   * @param {HTMLElement} vizEl The DOM element to render the heatmap into.
   * @param {HTMLElement} textEl The DOM element to render the full vector text into.
   */
  async render(data, vizEl, textEl) {
    const DATA = (data instanceof tf.Tensor) ? await data.data() : data;
    
    // Render grid viz (all dimensions).
    vizEl.innerHTML = '';

    for (let i = 0; i < DATA.length; i++) {
      const VALUE = DATA[i];
      const CELL = document.createElement('div');
      CELL.className = 'viz-cell';
      
      // Simple heatmap mapping.
      const INTENSITY = Math.max(0, Math.min(255, (VALUE + 0.1) * 1000 + 128)); 
      CELL.style.backgroundColor = `rgb(${INTENSITY}, ${INTENSITY/2}, 255)`;
      CELL.title = `Dim ${i}: ${VALUE.toFixed(4)}`;
      vizEl.appendChild(CELL);
    }
    
    // Show full vector.
    textEl.innerText = 'Full Vector: [' + Array.from(DATA).map(v => v.toFixed(4)).join(', ') + ']';
  }
}
