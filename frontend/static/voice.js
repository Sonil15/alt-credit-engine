/**
 * VoiceProvider abstraction for multilingual psychometric assessment.
 * Default: Browser Web Speech API (free). Swap with BhashiniProvider later.
 */

const LANG_MAP = {
  en: "en-IN",
  hi: "hi-IN",
  bn: "bn-IN",
};

class BrowserVoiceProvider {
  constructor() {
    this.synth = window.speechSynthesis || null;
    this.recognition = this._initRecognition();
  }

  _initRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return null;
    const rec = new SpeechRecognition();
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    return rec;
  }

  isSupported() {
    return !!(this.synth && this.recognition);
  }

  speak(text, lang = "en") {
    if (!this.synth || !text) return Promise.resolve();
    return new Promise((resolve) => {
      this.synth.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.lang = LANG_MAP[lang] || LANG_MAP.en;
      utter.rate = 0.95;
      utter.onend = () => resolve();
      utter.onerror = () => resolve();
      this.synth.speak(utter);
    });
  }

  listen(lang = "en") {
    if (!this.recognition) {
      return Promise.reject(new Error("Speech recognition not supported in this browser"));
    }
    return new Promise((resolve, reject) => {
      this.recognition.lang = LANG_MAP[lang] || LANG_MAP.en;
      this.recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        resolve(transcript.trim());
      };
      this.recognition.onerror = (event) => reject(new Error(event.error || "recognition failed"));
      this.recognition.onend = () => {};
      this.recognition.start();
    });
  }
}

/** Placeholder for future Bhashini/AI4Bharat integration */
class BhashiniVoiceProvider {
  constructor(_apiKey) {
    this.apiKey = _apiKey;
  }
  isSupported() {
    return false;
  }
  speak() {
    return Promise.reject(new Error("Bhashini provider not configured"));
  }
  listen() {
    return Promise.reject(new Error("Bhashini provider not configured"));
  }
}

function createVoiceProvider() {
  const provider = new BrowserVoiceProvider();
  if (provider.isSupported()) return provider;
  return provider; // graceful text-only fallback handled by UI
}

window.VoiceProvider = {
  BrowserVoiceProvider,
  BhashiniVoiceProvider,
  createVoiceProvider,
  LANG_MAP,
};
