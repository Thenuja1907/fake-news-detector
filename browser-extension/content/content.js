// Grammarly-like detector for Fake News
console.log("Truth Guardian: Active and protecting...");

const INJECT_CSS = `
    .truth-guardian-marker {
        position: relative;
        display: inline-block;
        margin-left: 5px;
        cursor: pointer;
        vertical-align: middle;
        z-index: 10000;
        animation: truth-pop 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    @keyframes truth-pop {
        0% { transform: scale(0); }
        100% { transform: scale(1); }
    }

    .tg-tooltip {
        visibility: hidden;
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #1e293b;
        color: white;
        text-align: center;
        padding: 12px;
        border-radius: 12px;
        width: 220px;
        font-family: sans-serif;
        font-size: 13px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .truth-guardian-marker:hover .tg-tooltip {
        visibility: visible;
        opacity: 1;
    }

    .tg-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 99px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .tg-real { background: #10b981; }
    .tg-fake { background: #ef4444; }
    .tg-neutral { background: #3b82f6; }
`;

// Inject Styles
const styleEl = document.createElement('style');
styleEl.textContent = INJECT_CSS;
document.head.appendChild(styleEl);

// Function to analyze text
async function analyzeSnippet(text, element) {
  try {
    const response = await fetch('http://127.0.0.1:5000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: text,
        url: window.location.href
      })
    });

    const result = await response.json();
    injectIndicator(element, result);
  } catch (err) {
    console.error("Truth Guardian Analysis Error:", err);
  }
}

// Function to inject the indicator
function injectIndicator(element, data) {
  if (element.querySelector('.truth-guardian-marker')) return;

  const marker = document.createElement('span');
  marker.className = 'truth-guardian-marker';

  const isFake = data.classification === "Fake News";
  const color = isFake ? '#ef4444' : '#10b981';

  marker.innerHTML = `
        <svg viewBox="0 0 24 24" width="18" height="18" fill="${color}" style="filter: drop-shadow(0 0 2px ${color}40)">
            <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-1 14.41L7.7 12.11l1.41-1.41 1.89 1.89 4.48-4.48 1.41 1.41L11 15.41z"/>
        </svg>
        <div class="tg-tooltip">
            <span class="tg-badge ${isFake ? 'tg-fake' : 'tg-real'}">${data.classification}</span>
            <div style="margin-top: 5px;">
                <strong>Credibility:</strong> ${data.credibility_score}%
            </div>
            <div style="font-size: 11px; margin-top: 8px; color: #94a3b8; line-height: 1.4;">
                ${data.explanation}
            </div>
        </div>
    `;

  element.appendChild(marker);
}

// Scan for paragraphs
function scanPage() {
  const paragraphs = document.querySelectorAll('p, h1, h2, article');
  paragraphs.forEach(p => {
    const text = p.innerText.trim();
    if (text.length > 50 && text.length < 1000 && !p.dataset.tgAnalyzed) {
      p.dataset.tgAnalyzed = "true";
      // Random delay to mimic organic scanning and avoid overwhelming server
      setTimeout(() => analyzeSnippet(text, p), Math.random() * 5000);
    }
  });
}

// Debounced scan
let scanTimeout;
window.addEventListener('scroll', () => {
  clearTimeout(scanTimeout);
  scanTimeout = setTimeout(scanPage, 2000);
});

// Initial scan
setTimeout(scanPage, 3000);
