# Antigravity | Fake News Detection Extension

Premium AI-powered verification layer for social media. Seamlessly detects and highlights fake news in your Facebook, Twitter (X), and Reddit feeds using real-time neural analysis.

## ✨ Features
- **Neural Layer Integration**: Acts as a transparent 'Antigravity' UI layer over social posts.
- **Dynamic Monitoring**: Uses `MutationObserver` to scan newly loaded content without page refreshes.
- **Visual Intelligence**:
  - **Red Glow**: Indicates suspicious or fabricated content.
  - **Green Border**: Marks corroborated, verified reporting.
- **'Show Why' Tooltips**: Hover over the badge to see:
  - **Credibility Score**: Real-time 0-100% confidence index.
  - **Sensational Keywords**: Highlighted manipulative terms (e.g., 'shocking', 'exposed').
  - **Decision Summary**: Translucent explanation of the model's logic.

## 🚀 Installation
1. Open Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in the top right).
3. Click **Load unpacked**.
4. Select the `extension/` directory within this project.

## ⚙️ Requirements
- The **Fake News Detector API** (Flask server) must be running at `http://127.0.0.1:5000`.

## 🛡️ Privacy & Security
Analyses are performed on-the-fly. Personal data is not transmitted, only the textual content of news posts is processed for verification.
