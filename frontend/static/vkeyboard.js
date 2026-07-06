/* Reusable on-screen Hindi/Bengali/English keyboard.
 * Extracted from the assessment page's open-ended-answer keyboard so any free-text
 * field (business description, "other" purpose, etc.) can offer the same typing aid
 * without requiring the borrower's device to have an Indic system keyboard installed. */
(function () {
  const KEYBOARD_LAYOUTS = {
    en: [
      ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
      ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
      ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '?'],
      ['Space', 'Backspace', 'Clear']
    ],
    hi: [
      ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ'],
      ['क', 'ख', 'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ट', 'ठ'],
      ['ड', 'ढ', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब'],
      ['भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह'],
      ['ा', 'ि', 'ी', 'ु', 'ू', 'े', 'ै', 'ो', 'ौ', '्'],
      ['Space', 'Backspace', 'Clear']
    ],
    bn: [
      ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'এ', 'ঐ', 'ও', 'ঔ'],
      ['ক', 'খ', 'গ', 'ঘ', 'চ', 'ছ', 'জ', 'ঝ', 'ট', 'ঠ'],
      ['ড', 'ঢ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব'],
      ['ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়'],
      ['া', 'ি', 'ী', 'ু', 'ূ', 'ে', 'ৈ', 'ো', 'ৌ', '্'],
      ['Space', 'Backspace', 'Clear']
    ]
  };

  const CONTROL_LABELS = {
    Space: { en: 'Space', hi: 'स्थान', bn: 'স্পেস' },
    Backspace: { en: 'Backspace', hi: 'हटाएं', bn: 'মুছুন' },
    Clear: { en: 'Clear', hi: 'साफ़ करें', bn: 'পরিষ্কার' }
  };

  /**
   * Wires a keyboard-toggle button + panel to a text input/textarea.
   * @param {Object} opts
   * @param {string|HTMLElement} opts.toggleBtn - toggle button element or id
   * @param {string|HTMLElement} opts.panel - container element or id to render keys into
   * @param {string|HTMLElement} opts.input - the input/textarea element or id to type into
   * @param {function} [opts.getLanguage] - returns current language code; defaults to window.getCurrentLanguage()
   */
  function attachVirtualKeyboard(opts) {
    const toggleBtn = typeof opts.toggleBtn === 'string' ? document.getElementById(opts.toggleBtn) : opts.toggleBtn;
    const panel = typeof opts.panel === 'string' ? document.getElementById(opts.panel) : opts.panel;
    const input = typeof opts.input === 'string' ? document.getElementById(opts.input) : opts.input;
    const getLanguage = opts.getLanguage || (() => (window.getCurrentLanguage ? window.getCurrentLanguage() : 'en'));

    if (!toggleBtn || !panel || !input) return;

    function render() {
      panel.innerHTML = '';
      const layout = KEYBOARD_LAYOUTS[getLanguage()] || KEYBOARD_LAYOUTS.en;
      layout.forEach(row => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'keyboard-row';
        row.forEach(key => {
          const keyBtn = document.createElement('button');
          keyBtn.type = 'button';
          keyBtn.className = 'keyboard-key';
          keyBtn.textContent = key;
          if (CONTROL_LABELS[key]) {
            keyBtn.classList.add('control-key');
            keyBtn.textContent = CONTROL_LABELS[key][getLanguage()] || CONTROL_LABELS[key].en;
          }
          keyBtn.onclick = (e) => {
            e.preventDefault();
            handleKeyPress(key);
          };
          rowDiv.appendChild(keyBtn);
        });
        panel.appendChild(rowDiv);
      });
    }

    function handleKeyPress(key) {
      if (key === 'Space') {
        input.value += ' ';
      } else if (key === 'Backspace') {
        input.value = input.value.slice(0, -1);
      } else if (key === 'Clear') {
        input.value = '';
      } else {
        input.value += key;
      }
      input.focus();
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function toggle() {
      const isHidden = panel.classList.contains('hidden');
      if (isHidden) {
        panel.classList.remove('hidden');
        toggleBtn.classList.add('active');
        render();
        input.focus();
      } else {
        panel.classList.add('hidden');
        toggleBtn.classList.remove('active');
      }
    }

    toggleBtn.onclick = (e) => {
      e.preventDefault();
      toggle();
    };

    // Re-render live if the borrower switches language while the panel is open.
    window.addEventListener('altcreditLanguageChanged', () => {
      if (!panel.classList.contains('hidden')) render();
    });

    return { render, toggle };
  }

  window.attachVirtualKeyboard = attachVirtualKeyboard;
})();
