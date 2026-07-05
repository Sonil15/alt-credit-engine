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
    this._voices = [];
    if (this.synth) {
      const loadVoices = () => { this._voices = this.synth.getVoices() || []; };
      loadVoices();
      // Voices often populate asynchronously; refresh when they do.
      this.synth.addEventListener("voiceschanged", loadVoices);
    }
  }

  /** Best installed voice for a language, or null if the device has none. */
  _voiceFor(lang) {
    const target = (LANG_MAP[lang] || LANG_MAP.en).toLowerCase(); // e.g. "bn-in"
    const base = target.split("-")[0];                            // e.g. "bn"
    const voices = this._voices.length
      ? this._voices
      : (this.synth ? this.synth.getVoices() : []);
    return (
      voices.find((v) => v.lang && v.lang.toLowerCase() === target) ||
      voices.find((v) => v.lang && v.lang.toLowerCase().startsWith(base + "-")) ||
      voices.find((v) => v.lang && v.lang.toLowerCase() === base) ||
      null
    );
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
    const voice = this._voiceFor(lang);
    // No installed voice for this language → stay silent instead of letting a
    // mismatched voice read the script as garbage (e.g. an English voice
    // speaking Bengali "নমস্কার!" aloud as "exclamation point").
    if (!voice) return Promise.resolve();
    return new Promise((resolve) => {
      this.synth.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.voice = voice;
      utter.lang = voice.lang;
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
    // Re-initialise to avoid "already started" errors on repeated taps
    this.recognition = this._initRecognition();
    return new Promise((resolve, reject) => {
      let settled = false;
      const settle = (fn, val) => { if (!settled) { settled = true; fn(val); } };

      this.recognition.lang = LANG_MAP[lang] || LANG_MAP.en;
      this.recognition.continuous = false;
      this.recognition.interimResults = false;

      this.recognition.onresult = (event) => {
        settle(resolve, event.results[0][0].transcript.trim());
      };
      this.recognition.onerror = (event) => {
        const raw = event.error || "recognition failed";
        const msg = raw === "network"
          ? "network error — mic requires HTTPS or localhost, and microphone permission must be allowed"
          : raw === "not-allowed"
          ? "microphone permission denied — allow mic access in your browser settings"
          : raw === "no-speech"
          ? "no speech detected — please try again"
          : raw;
        settle(reject, new Error(msg));
      };
      // onend fires after every session; reject if nothing was resolved yet
      this.recognition.onend = () => settle(reject, new Error("no speech detected — please try again"));

      try {
        this.recognition.start();
      } catch (e) {
        settle(reject, e);
      }
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

/**
 * Records mic audio and sends it to the backend's /speech/transcribe endpoint
 * (Sarvam, tuned for Indian languages, with Gemini as fallback — see speech/
 * on the backend). Used instead of the browser's built-in recognition when a
 * server provider is configured, since it handles Hindi/Bengali accents more
 * reliably than the browser's own speech engine.
 */
class ServerSpeechRecorder {
  constructor() {
    this._recorder = null;
    this._chunks = [];
  }

  async isAvailable() {
    if (!navigator.mediaDevices || !window.MediaRecorder) return false;
    try {
      const resp = await fetch("/speech/config");
      if (!resp.ok) return false;
      const data = await resp.json();
      return !!data.stt_available;
    } catch {
      return false;
    }
  }

  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._chunks = [];
    this._recorder = new MediaRecorder(stream);
    this._recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this._chunks.push(e.data);
    };
    this._recorder.start();
  }

  /** Stops recording and returns the transcribed text. */
  stop(lang = "en") {
    return new Promise((resolve, reject) => {
      if (!this._recorder) return reject(new Error("Recording was never started"));
      const recorder = this._recorder;
      const tracks = recorder.stream.getTracks();
      recorder.onstop = async () => {
        tracks.forEach((t) => t.stop());
        const blob = new Blob(this._chunks, { type: recorder.mimeType || "audio/webm" });
        try {
          const form = new FormData();
          form.append("language", lang);
          form.append("audio", blob, "audio.webm");
          const resp = await fetch("/speech/transcribe", { method: "POST", body: form });
          if (!resp.ok) {
            const detail = (await resp.json().catch(() => ({}))).detail || "transcription failed";
            return reject(new Error(detail));
          }
          const data = await resp.json();
          resolve((data.text || "").trim());
        } catch (e) {
          reject(e);
        }
      };
      recorder.stop();
    });
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
  ServerSpeechRecorder,
  createVoiceProvider,
  LANG_MAP,
};
