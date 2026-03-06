/**
 * Antigravity | Chrome Extension Content Script
 * Monitors DOM for news posts and injects verified overlays.
 */

const API_ENDPOINT = 'http://127.0.0.1:5000/analyze';
const PROCESSED_ATTR = 'data-antigravity-processed';

// Selectors for common social media post containers
const POST_SELECTORS = [
  'article',
  '[role="article"]',
  '.feed-item',
  '.Post',
  '[data-testid="tweet"]'
];

/**
 * Main function to scan for new posts
 */
function scanFeed() {
  POST_SELECTORS.forEach(selector => {
    const posts = document.querySelectorAll(`${selector}:not([${PROCESSED_ATTR}])`);
    posts.forEach(async (post) => {
      post.setAttribute(PROCESSED_ATTR, 'true');

      // Extract text content
      const textContent = post.innerText.trim();
      if (textContent.length < 50) return; // Skip short posts

      try {
        const response = await fetch(API_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: textContent })
        });

        if (response.ok) {
          const data = await response.json();
          injectOverlay(post, data);
        }
      } catch (error) {
        console.error('Antigravity API Error:', error);
      }
    });
  });
}

/**
 * Injects the UI overlay onto the post container
 * @param {HTMLElement} post 
 * @param {Object} data 
 */
function injectOverlay(post, data) {
  const isFake = data.classification.toLowerCase().includes('fake');
  const colorClass = isFake ? 'antigravity-fake' : 'antigravity-real';
  const badgeText = isFake ? 'SUSPICIOUS' : 'VERIFIED';

  // Create the overlay container
  const wrapper = document.createElement('div');
  wrapper.className = `antigravity-layer ${colorClass}`;

  // Create the badge
  const badge = document.createElement('div');
  badge.className = 'antigravity-badge';
  badge.innerHTML = `
        <span class="badge-icon">${isFake ? '⚠' : '✓'}</span>
        <span class="badge-text">${badgeText}</span>
        <div class="antigravity-tooltip">
            <strong>Truth Matrix: ${data.credibility_score}%</strong>
            <p>${data.decision_summary}</p>
            <div class="sensational-words">
                ${data.sensational_words.length > 0 ?
      `<span>Sensational terms: ${data.sensational_words.join(', ')}</span>` :
      ''}
            </div>
            <a href="http://127.0.0.1:5000/dashboard" target="_blank" class="view-report">Show Why →</a>
        </div>
    `;

  // Ensure post has relative positioning for overlay if needed
  if (getComputedStyle(post).position === 'static') {
    post.style.position = 'relative';
  }

  post.appendChild(badge);
  if (isFake) {
    post.classList.add('antigravity-glow');
  } else {
    post.classList.add('antigravity-border');
  }
}

// Initial Scan
scanFeed();

// Create MutationObserver to detect dynamic content loading
const observer = new MutationObserver((mutations) => {
  let shouldScan = false;
  mutations.forEach(mutation => {
    if (mutation.addedNodes.length > 0) shouldScan = true;
  });
  if (shouldScan) {
    // Debounce scan
    clearTimeout(window.antigravityScanTimeout);
    window.antigravityScanTimeout = setTimeout(scanFeed, 500);
  }
});

observer.observe(document.body, { childList: true, subtree: true });

console.log('🚀 Antigravity Layer Active');
